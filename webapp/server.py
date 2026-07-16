#!/usr/bin/env python3
# ============================================================================
#  Web UI for the classical plasma simulation.
# ============================================================================
#
#  A dependency-free (Python standard library only) HTTP server that lets you
#  drive the existing C simulator + visualize.py from a browser form:
#
#      Browser form  ->  POST /api/run   (validated + clamped, then queued)
#                    ->  background worker runs ./plasma_sim_parallel and
#                        python3 visualize.py in an isolated per-job directory
#                    ->  GET /api/status/<id>  (polled ~1 Hz for live progress)
#                    ->  GET /api/file/<id>/gif|plots  (the animation + plots)
#
#  Physics parameters (box, dt, temperature, B, E, softening, save-stride) are
#  passed to the C binary through the PLASMA_* environment variables it now
#  understands; N / nsteps / seed / nthreads are passed as positional args.
#
#  Only ONE simulation runs at a time (a single worker thread drains the queue)
#  so the shared compute node is never overloaded by concurrent O(N^2) runs.
#
#  Usage:
#      python3 webapp/server.py [--port 8000] [--host 0.0.0.0]
#
#  Then open  http://<this-host>:8000  in a browser (or tunnel with
#  `ssh -L 8000:localhost:8000 <host>` and open http://localhost:8000).
# ----------------------------------------------------------------------------

import argparse
import glob
import json
import os
import queue
import shutil
import subprocess
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# --- Paths -----------------------------------------------------------------
# Resolve everything relative to this file so the server works from any cwd.
WEBAPP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(WEBAPP_DIR)                 # the project directory
INDEX_HTML = os.path.join(WEBAPP_DIR, "index.html")
BINARY     = os.path.join(ROOT_DIR, "plasma_sim_parallel")
BINARY_SRC = os.path.join(ROOT_DIR, "plasma_sim_parallel.c")
JOBS_DIR   = os.path.join(ROOT_DIR, "output", "web_jobs")

NCPU        = os.cpu_count() or 4
MAX_JOBS_KEPT = 20            # prune old per-job output dirs beyond this many
LOG_TAIL_LINES = 60          # how many recent log lines to surface to the page

# --- Parameter validation spec ---------------------------------------------
# Each numeric parameter: (json_key, env_var, kind, min, max).  "int"/"float"
# values are clamped into [min, max]; a None bound means unbounded on that side.
# Positional args (N, nsteps, seed, nthreads) use env_var=None.
PARAM_SPEC = [
    ("N",       None,          "int",   1,   3000),
    ("nsteps",  None,          "int",   1,   20000),
    ("seed",    None,          "int",   0,   2**31 - 1),
    ("nthreads", None,         "int",   1,   NCPU),
    ("Lx",      "PLASMA_LX",   "float", 1e-9, 1.0),
    ("Ly",      "PLASMA_LY",   "float", 1e-9, 1.0),
    ("Lz",      "PLASMA_LZ",   "float", 1e-9, 1.0),
    ("dt",      "PLASMA_DT",   "float", 1e-18, 1.0),
    ("temp",    "PLASMA_TEMP", "float", 1.0,  1e12),
    ("Bx",      "PLASMA_BX",   "float", -1e6, 1e6),
    ("By",      "PLASMA_BY",   "float", -1e6, 1e6),
    ("Bz",      "PLASMA_BZ",   "float", -1e6, 1e6),
    ("Ex",      "PLASMA_EX",   "float", -1e12, 1e12),
    ("Ey",      "PLASMA_EY",   "float", -1e12, 1e12),
    ("Ez",      "PLASMA_EZ",   "float", -1e12, 1e12),
    ("soft",    "PLASMA_SOFT", "float", 1e-12, 1.0),
    ("stride",  "PLASMA_STRIDE", "int", 1,   100000),
]
DEFAULTS = {
    "N": 500, "nsteps": 2000, "seed": 12345, "nthreads": min(8, NCPU),
    "Lx": 1e-3, "Ly": 1e-3, "Lz": 1e-3, "dt": 1e-12, "temp": 1e4,
    "Bx": 0.0, "By": 0.0, "Bz": 1.0, "Ex": 1e3, "Ey": 0.0, "Ez": 0.0,
    "soft": 1e-6, "stride": 10,
}

# Guardrails on total work (the server is reachable on the network).
MAX_COST     = 5e10          # cap on N^2 * nsteps (pair-updates over the run)
MAX_FRAMES   = 1200          # cap on nsteps/stride (matplotlib render blows up)

# --- Job store -------------------------------------------------------------
JOBS = {}                    # job_id -> job dict
JOBS_LOCK = threading.Lock()
JOB_QUEUE = queue.Queue()


def clamp(value, lo, hi):
    if lo is not None and value < lo:
        value = lo
    if hi is not None and value > hi:
        value = hi
    return value


def validate_params(raw):
    """Coerce, clamp and sanity-check the submitted parameters.

    Returns (params, notes).  Raises ValueError with a human message on a
    fatal problem (non-positive size, too much work, too many frames)."""
    params = {}
    notes = []
    for key, _env, kind, lo, hi in PARAM_SPEC:
        val = raw.get(key, DEFAULTS[key])
        try:
            val = int(val) if kind == "int" else float(val)
        except (TypeError, ValueError):
            raise ValueError("%s: %r is not a valid number" % (key, val))
        if val != val or val in (float("inf"), float("-inf")):   # NaN / inf
            raise ValueError("%s: value must be finite" % key)
        clamped = clamp(val, lo, hi)
        if clamped != val:
            notes.append("%s clamped %s -> %s" % (key, val, clamped))
        params[key] = clamped

    # Extra sanity beyond the per-field ranges.
    nframes = params["nsteps"] // params["stride"] + 1
    if nframes > MAX_FRAMES:
        raise ValueError(
            "Too many animation frames (%d). Increase 'save stride' or lower "
            "'steps' so steps/stride <= %d." % (nframes, MAX_FRAMES))
    cost = float(params["N"]) ** 2 * params["nsteps"]
    if cost > MAX_COST:
        raise ValueError(
            "Requested run is too large (N^2 x steps = %.2g, limit %.2g). "
            "Lower N and/or steps." % (cost, MAX_COST))
    return params, notes


def build_env(params):
    """Environment for the C binary: base env + PLASMA_* overrides + outdir."""
    env = os.environ.copy()
    for key, envvar, kind, _lo, _hi in PARAM_SPEC:
        if envvar is None:
            continue
        env[envvar] = repr(params[key]) if kind == "float" else str(params[key])
    return env


def read_meta(path):
    meta = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if "=" in line:
                k, v = line.split("=", 1)
                meta[k.strip()] = v.strip()
    return meta


def prune_old_jobs():
    """Keep only the newest MAX_JOBS_KEPT per-job output directories."""
    try:
        dirs = [os.path.join(JOBS_DIR, d) for d in os.listdir(JOBS_DIR)]
    except FileNotFoundError:
        return
    dirs = [d for d in dirs if os.path.isdir(d)]
    dirs.sort(key=os.path.getmtime, reverse=True)
    for old in dirs[MAX_JOBS_KEPT:]:
        shutil.rmtree(old, ignore_errors=True)


def set_state(job, state, message=""):
    with JOBS_LOCK:
        job["state"] = state
        job["message"] = message


def append_log(job, text):
    with JOBS_LOCK:
        for line in text.rstrip("\n").split("\n"):
            job["log"].append(line)
        del job["log"][:-LOG_TAIL_LINES]      # keep only the tail


def ensure_binary(job):
    """Build ./plasma_sim_parallel if missing or older than its source."""
    fresh = (os.path.exists(BINARY) and os.path.exists(BINARY_SRC)
             and os.path.getmtime(BINARY) >= os.path.getmtime(BINARY_SRC))
    if fresh:
        return True
    set_state(job, "building", "Compiling the simulator...")
    proc = subprocess.run(["make", "parallel"], cwd=ROOT_DIR,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True)
    append_log(job, proc.stdout or "")
    if proc.returncode != 0 or not os.path.exists(BINARY):
        set_state(job, "error", "Build failed.")
        return False
    return True


def stream_subprocess(job, cmd, env, cwd):
    """Run cmd, streaming merged stdout/stderr into the job log. Return rc."""
    proc = subprocess.Popen(cmd, cwd=cwd, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    for line in proc.stdout:
        append_log(job, line)
    proc.stdout.close()
    return proc.wait()


def run_job(job):
    params = job["params"]
    job_dir = os.path.join(JOBS_DIR, job["id"])
    os.makedirs(job_dir, exist_ok=True)
    env = build_env(params)
    env["PLASMA_OUTDIR"] = job_dir

    if not ensure_binary(job):
        return

    # --- Simulate ----------------------------------------------------------
    set_state(job, "simulating",
              "Running the simulation (%d particles, %d steps)..."
              % (2 * params["N"], params["nsteps"]))
    cmd = [BINARY, str(params["N"]), str(params["nsteps"]),
           str(params["seed"]), str(params["nthreads"])]
    rc = stream_subprocess(job, cmd, env, ROOT_DIR)
    if rc != 0:
        set_state(job, "error", "Simulation exited with code %d." % rc)
        return

    metas = glob.glob(os.path.join(job_dir, "plasma_meta_*.txt"))
    if not metas:
        set_state(job, "error", "Simulation produced no metadata file.")
        return
    meta_path = metas[0]

    # --- Render ------------------------------------------------------------
    set_state(job, "rendering", "Rendering animation and diagnostic plots...")
    viz = os.path.join(ROOT_DIR, "visualize.py")
    rc = stream_subprocess(job, ["python3", viz, meta_path], env, ROOT_DIR)
    if rc != 0:
        set_state(job, "error", "Visualization exited with code %d." % rc)
        return

    gifs  = glob.glob(os.path.join(job_dir, "plasma_animation_*.gif"))
    plots = glob.glob(os.path.join(job_dir, "plasma_plots_*.png"))
    if not gifs:
        set_state(job, "error", "No animation (.gif) was produced.")
        return

    with JOBS_LOCK:
        job["gif"] = gifs[0]
        job["plots"] = plots[0] if plots else None
        job["meta"] = read_meta(meta_path)
    set_state(job, "done", "Done.")


def worker_loop():
    while True:
        job = JOB_QUEUE.get()
        try:
            prune_old_jobs()
            run_job(job)
        except Exception as exc:                              # never kill worker
            set_state(job, "error", "Server error: %s" % exc)
            append_log(job, "EXCEPTION: %s" % exc)
        finally:
            JOB_QUEUE.task_done()


# --- HTTP handler ----------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    server_version = "PlasmaWeb/1.0"

    def log_message(self, fmt, *args):        # quieter console
        pass

    # -- helpers --
    def _send_json(self, obj, code=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, data, ctype, code=200, download_name=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        if download_name:
            self.send_header("Content-Disposition",
                             'inline; filename="%s"' % download_name)
        self.end_headers()
        self.wfile.write(data)

    def _job_public(self, job):
        with JOBS_LOCK:
            out = {
                "job_id": job["id"],
                "state": job["state"],
                "message": job["message"],
                "log_tail": "\n".join(job["log"]),
                "notes": job["notes"],
                "meta": job.get("meta"),
            }
            if job["state"] == "done":
                out["gif_url"] = "/api/file/%s/gif" % job["id"]
                if job.get("plots"):
                    out["plots_url"] = "/api/file/%s/plots" % job["id"]
        return out

    # -- routing --
    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/" or path == "/index.html":
            return self._serve_index()
        if path == "/favicon.ico":
            return self._send_bytes(b"", "image/x-icon", code=204)
        if path == "/api/config":
            return self._send_json({"defaults": DEFAULTS, "ncpu": NCPU,
                                    "max_cost": MAX_COST, "max_frames": MAX_FRAMES})
        if path.startswith("/api/status/"):
            return self._handle_status(path.rsplit("/", 1)[-1])
        if path.startswith("/api/file/"):
            return self._handle_file(path)
        self._send_json({"error": "not found"}, code=404)

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path == "/api/run":
            return self._handle_run()
        self._send_json({"error": "not found"}, code=404)

    # -- handlers --
    def _serve_index(self):
        try:
            with open(INDEX_HTML, "rb") as fh:
                data = fh.read()
        except FileNotFoundError:
            return self._send_json({"error": "index.html missing"}, code=500)
        self._send_bytes(data, "text/html; charset=utf-8")

    def _handle_run(self):
        length = int(self.headers.get("Content-Length", 0) or 0)
        raw_body = self.rfile.read(length) if length else b"{}"
        try:
            raw = json.loads(raw_body.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            return self._send_json({"error": "invalid JSON body"}, code=400)
        try:
            params, notes = validate_params(raw)
        except ValueError as exc:
            return self._send_json({"error": str(exc)}, code=400)

        job_id = uuid.uuid4().hex
        job = {"id": job_id, "state": "queued",
               "message": "Queued (position %d)." % (JOB_QUEUE.qsize() + 1),
               "params": params, "notes": notes, "log": [],
               "gif": None, "plots": None, "meta": None}
        with JOBS_LOCK:
            JOBS[job_id] = job
        JOB_QUEUE.put(job)
        self._send_json({"job_id": job_id, "notes": notes})

    def _handle_status(self, job_id):
        with JOBS_LOCK:
            job = JOBS.get(job_id)
        if not job:
            return self._send_json({"error": "unknown job"}, code=404)
        self._send_json(self._job_public(job))

    def _handle_file(self, path):
        parts = path.strip("/").split("/")          # ['api','file',<id>,<kind>]
        if len(parts) != 4:
            return self._send_json({"error": "bad file path"}, code=400)
        job_id, kind = parts[2], parts[3]
        with JOBS_LOCK:
            job = JOBS.get(job_id)
            fpath = job.get(kind) if job else None   # kind must be 'gif'/'plots'
        if kind not in ("gif", "plots") or not fpath or not os.path.exists(fpath):
            return self._send_json({"error": "file not found"}, code=404)
        ctype = "image/gif" if kind == "gif" else "image/png"
        with open(fpath, "rb") as fh:
            data = fh.read()
        self._send_bytes(data, ctype, download_name=os.path.basename(fpath))


def main():
    ap = argparse.ArgumentParser(description="Plasma simulation web UI")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()

    os.makedirs(JOBS_DIR, exist_ok=True)
    threading.Thread(target=worker_loop, daemon=True).start()

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    url_host = "localhost" if args.host in ("0.0.0.0", "") else args.host
    print("Plasma simulation web UI")
    print("  serving on http://%s:%d  (bound to %s)"
          % (url_host, args.port, args.host))
    print("  project dir : %s" % ROOT_DIR)
    print("  job outputs : %s" % JOBS_DIR)
    print("  Ctrl-C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        httpd.shutdown()


if __name__ == "__main__":
    main()

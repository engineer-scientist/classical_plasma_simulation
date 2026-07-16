/* ============================================================================
 *  plasma_sim_parallel.c  --  Classical plasma simulation, OpenMP PARALLEL
 * ============================================================================
 *
 *  A box of dimensions Lx x Ly x Lz contains N electrons and N protons
 *  (NP = 2*N charged particles total).  The plasma is hot enough that the
 *  thermal kinetic energy dominates the Coulomb binding energy, so electrons
 *  and protons do not recombine into neutral atoms -- they stay a plasma.
 *
 *  Physics included (all classical / non-relativistic):
 *    1. Coulomb force between every pair of particles (attraction between
 *       unlike charges, repulsion between like charges).  A short-range
 *       "softening" length is used so the 1/r^2 force cannot blow up when two
 *       particles momentarily overlap (this is the numerical stand-in for the
 *       fact that at very small separations quantum effects / hard-core
 *       repulsion take over -- it also prevents unphysical recombination).
 *    2. The Lorentz force  F = q (E + v x B)  from uniform, static, external
 *       fields: an electric field E = (Ex, Ey, Ez) and a magnetic field
 *       B = (Bx, By, Bz).  The external E exerts q*E on each particle (pushing
 *       electrons and protons in opposite directions); combined with B it
 *       produces the classic charge-independent E x B drift.  Its potential
 *       energy  U_ext = -q (E . r)  is folded into the energy budget so the
 *       conservation check stays valid even though the field does work.
 *
 *  Particles specularly reflect off the six walls of the box (elastic bounce:
 *  the velocity component normal to the wall is reversed).
 *
 *  Time integration:
 *    We use the *Boris pusher*, the standard leap-frog scheme for charged
 *    particles in electromagnetic fields.  It splits each step into
 *        half electric kick  ->  magnetic rotation  ->  half electric kick
 *    where the "electric kick" here is the acceleration from the inter-particle
 *    Coulomb force (F/m), and the "rotation" applies the v x B turn exactly.
 *    This is far more stable/energy-conserving than plain explicit Euler, which
 *    would artificially pump energy into the magnetic gyration.  It still
 *    realises exactly the requested recipe: from the total force and the mass
 *    we get the acceleration, and from the acceleration + current position and
 *    velocity we advance to the end-of-step position and velocity.
 *
 *  Data layout (Structure-of-Arrays) keeps the O(N^2) force kernel -- the
 *  expensive part -- a simple, cache-friendly double loop that parallelises
 *  cleanly with OpenMP.
 *
 *  PARALLELISATION (this file): the force loop uses the "full-row" scheme.  Each
 *  thread owns a block of rows i and computes the force on i by summing over
 *  ALL j != i, writing only fx[i]/fy[i]/fz[i].  Because a thread never writes to
 *  another particle's force, there is no data race and no need for locks or
 *  private buffers -- it is embarrassingly parallel (#pragma omp parallel for).
 *  The trade-off is that every pair is now evaluated twice (once as (i,j) and
 *  once as (j,i)), so it does 2x the pair flops of the serial upper-triangle
 *  loop; the potential energy and virial sums are therefore halved at the end.
 *  This is approach (a) from the serial version's parallelisation note -- the
 *  simplest correct first step, and the one that scales most cleanly.
 *
 *  The per-particle Boris push and the kinetic-energy sum are O(N) and also
 *  parallelised (the latter with a reduction).  initialise() stays serial: it
 *  uses drand48() (not thread-safe) and running it serially means this build
 *  starts from the *identical* random state as plasma_sim_serial, so the two
 *  can be compared directly (results are physically equivalent, though not
 *  bit-identical, because the parallel sums add in a different order).
 *
 *  Output (all written to the current directory):
 *    plasma_meta.txt    - human-readable run metadata (used by visualize.py)
 *    plasma_traj.bin    - raw float32 positions, shape [nframes][NP][3]
 *    plasma_energy.csv  - step, time, KE, PE, total energy (for a stability check)
 *
 *  Build:  make parallel         (adds -fopenmp; see Makefile)
 *  Run:    ./plasma_sim_parallel [N] [nsteps] [seed] [nthreads]
 *          (all four are optional; nthreads defaults to OMP_NUM_THREADS or the
 *           number of cores.  All fall back to the defaults below.)
 *
 *  ------------------------------------------------------------------------- */

#define _GNU_SOURCE          /* expose drand48/srand48 and M_PI under -std=c11 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>        /* mkdir()  */
#include <errno.h>           /* errno, EEXIST */
#include <omp.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ----------------------------- physical constants (SI) ------------------- */
static const double E_CHARGE = 1.602176634e-19;   /* elementary charge   [C]  */
static const double MASS_E   = 9.1093837015e-31;  /* electron mass       [kg] */
static const double MASS_P   = 1.67262192369e-27; /* proton mass         [kg] */
static const double KB       = 1.380649e-23;      /* Boltzmann constant  [J/K]*/
static const double KE_COUL  = 8.9875517873681764e9; /* Coulomb constant 1/(4*pi*eps0) [N m^2/C^2] */

/* ----------------------------- default parameters ------------------------ *
 * These are "suitable values" chosen so the dynamics are rich but stable and
 * the run finishes in a few seconds.  Feel free to edit them or override N,
 * nsteps and seed on the command line.                                       */
static int    N           = 5000;    /* particles PER species (NP = 2*N)      */
static double Lx          = 1.0e-3;  /* box size x  [m]                       */
static double Ly          = 1.0e-3;  /* box size y  [m]                       */
static double Lz          = 1.0e-3;  /* box size z  [m]                       */
static double DT          = 1.0e-12; /* time step   [s]  (1 picosecond)       */
static long   NSTEPS      = 5000;    /* number of time steps T                */
static int    SAVE_STRIDE = 10;      /* save a trajectory frame every k steps */
static double TEMP        = 1.0e4;   /* plasma temperature [K] (~8.6 eV is hot  *
                                      * enough to keep it ionised)            */
static double BX          = 0.0;     /* external magnetic field [T] ...       */
static double BY          = 0.0;
static double BZ          = 1.0;     /* ... dominant along z -> gyration in xy */
static double EX          = 1.0e3;   /* external electric field [V/m] ...      *
                                      * default points along x, PERPENDICULAR   *
                                      * to B, so it drives a clean, charge- and *
                                      * mass-independent E x B drift of |E|/|B| *
                                      * ~= 500 m/s (in -y) rather than runaway   *
                                      * acceleration.  Set EX=EY=EZ=0.0 to       *
                                      * recover a pure magnetic-field run.       */
static double EY          = 0.0;
static double EZ          = 0.0;     /* a component along B (here z) would       *
                                      * freely accelerate the two species in     *
                                      * opposite directions (charge separation). */
static double SOFT        = 1.0e-6;  /* Coulomb softening length [m]          */
static unsigned int SEED  = 12345u;  /* RNG seed (reproducible runs)          */

/* ----------------------------- simple RNG helpers ------------------------ */
/* drand48() gives uniform [0,1); Box-Muller turns two uniforms into one
 * standard-normal sample, used to draw Maxwell-Boltzmann velocity components. */
static double gaussian(void)
{
    double u1, u2;
    do { u1 = drand48(); } while (u1 <= 1e-300); /* avoid log(0) */
    u2 = drand48();
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

/* ============================================================================
 *  Global particle state (Structure-of-Arrays).
 *  Indices 0 .. N-1     are electrons.
 *  Indices N .. 2N-1    are protons.
 * ========================================================================== */
static double *rx, *ry, *rz;   /* positions   [m]     */
static double *vx, *vy, *vz;   /* velocities  [m/s]   */
static double *fx, *fy, *fz;   /* forces      [N]     */
static double *q,  *m;         /* charge [C], mass [kg] per particle */
static double  PE;             /* total Coulomb potential energy [J] (filled by
                                * compute_forces at the current positions)    */
static double  VIRIAL;         /* pair virial  Sum_{i<j} r_ij . F_ij  [J]     *
                                * (also filled by compute_forces; used for the *
                                * virial pressure -- see the pressure note in  *
                                * main()).                                     */

static void allocate(int NP)
{
    rx = malloc(NP*sizeof(double)); ry = malloc(NP*sizeof(double)); rz = malloc(NP*sizeof(double));
    vx = malloc(NP*sizeof(double)); vy = malloc(NP*sizeof(double)); vz = malloc(NP*sizeof(double));
    fx = malloc(NP*sizeof(double)); fy = malloc(NP*sizeof(double)); fz = malloc(NP*sizeof(double));
    q  = malloc(NP*sizeof(double)); m  = malloc(NP*sizeof(double));
    if (!rx||!ry||!rz||!vx||!vy||!vz||!fx||!fy||!fz||!q||!m) {
        fprintf(stderr, "ERROR: out of memory allocating %d particles\n", NP);
        exit(1);
    }
}

static void free_all(void)
{
    free(rx); free(ry); free(rz);
    free(vx); free(vy); free(vz);
    free(fx); free(fy); free(fz);
    free(q);  free(m);
}

/* ----------------------------------------------------------------------------
 *  Initialise particles: random uniform positions inside the box and
 *  Maxwell-Boltzmann (Gaussian) velocities set by the temperature.  Each
 *  species' bulk (mean) velocity is subtracted off so the plasma as a whole
 *  does not drift -- this keeps the cloud centred in the box for a cleaner
 *  visualisation without changing the physics of the internal dynamics.
 * ------------------------------------------------------------------------- */
static void initialise(int NP)
{
    for (int i = 0; i < NP; ++i) {
        int is_electron = (i < N);
        q[i] = is_electron ? -E_CHARGE : +E_CHARGE;
        m[i] = is_electron ?  MASS_E   :  MASS_P;

        rx[i] = drand48() * Lx;
        ry[i] = drand48() * Ly;
        rz[i] = drand48() * Lz;

        double sigma = sqrt(KB * TEMP / m[i]);   /* thermal speed per axis */
        vx[i] = sigma * gaussian();
        vy[i] = sigma * gaussian();
        vz[i] = sigma * gaussian();
    }

    /* Remove net drift of each species separately. */
    for (int s = 0; s < 2; ++s) {
        int lo = s * N, hi = lo + N;
        double mx = 0, my = 0, mz = 0;
        for (int i = lo; i < hi; ++i) { mx += vx[i]; my += vy[i]; mz += vz[i]; }
        mx /= N; my /= N; mz /= N;
        for (int i = lo; i < hi; ++i) { vx[i] -= mx; vy[i] -= my; vz[i] -= mz; }
    }
}

/* ----------------------------------------------------------------------------
 *  compute_forces:  fill fx/fy/fz with the total Coulomb force on every
 *  particle and accumulate the total potential energy in the global PE.
 *
 *  Softened Coulomb:  with d = r_i - r_j and  r2 = |d|^2 + SOFT^2,
 *      F_on_i = KE_COUL * q_i * q_j * d / r2^{3/2}
 *  which is repulsive (points along +d, away from j) when q_i*q_j > 0 and
 *  attractive when q_i*q_j < 0, exactly as required.
 *
 *  PARALLEL "full-row" form:  the outer loop over i is split across threads with
 *  #pragma omp parallel for.  Each thread computes the force on its rows i by
 *  summing over ALL j != i and writes only fx[i]/fy[i]/fz[i] -- it never touches
 *  another particle's force, so there is no data race (this is why we drop the
 *  serial j>i + Newton's-third-law trick here).  Every unordered pair is thus
 *  visited twice, so the potential energy and virial -- accumulated with an
 *  OpenMP reduction -- are halved at the end to recover their true totals.
 * ------------------------------------------------------------------------- */
static void compute_forces(int NP)
{
    double pe = 0.0, vir = 0.0;
    const double soft2 = SOFT * SOFT;

    #pragma omp parallel for schedule(static) reduction(+:pe,vir)
    for (int i = 0; i < NP; ++i) {
        const double xi = rx[i], yi = ry[i], zi = rz[i], qi = q[i];
        double fxi = 0.0, fyi = 0.0, fzi = 0.0;
        for (int j = 0; j < NP; ++j) {
            if (j == i) continue;
            double dx = xi - rx[j];
            double dy = yi - ry[j];
            double dz = zi - rz[j];
            double r2 = dx*dx + dy*dy + dz*dz + soft2;
            double inv_r  = 1.0 / sqrt(r2);
            double inv_r3 = inv_r * inv_r * inv_r;
            double qq = qi * q[j];
            double fmag = KE_COUL * qq * inv_r3;   /* so F_vec = fmag * d      */

            double Fx = fmag * dx, Fy = fmag * dy, Fz = fmag * dz;
            fxi += Fx;  fyi += Fy;  fzi += Fz;     /* force on i (from all j)  */

            pe  += KE_COUL * qq * inv_r;           /* softened potential energy */
            vir += Fx*dx + Fy*dy + Fz*dz;          /* r_ij . F_ij  (pair virial)*/
        }
        fx[i] = fxi;  fy[i] = fyi;  fz[i] = fzi;   /* thread is sole owner of i */
    }
    /* Every pair was counted twice (row i and row j), so halve the scalar sums.*/
    PE = 0.5 * pe;
    VIRIAL = 0.5 * vir;
}

/* ----------------------------------------------------------------------------
 *  Reflect a particle off the six walls (specular / elastic bounce).
 * ------------------------------------------------------------------------- */
static inline void reflect(int i)
{
    if (rx[i] < 0.0) { rx[i] = -rx[i];        vx[i] = -vx[i]; }
    if (rx[i] > Lx ) { rx[i] = 2.0*Lx - rx[i]; vx[i] = -vx[i]; }
    if (ry[i] < 0.0) { ry[i] = -ry[i];        vy[i] = -vy[i]; }
    if (ry[i] > Ly ) { ry[i] = 2.0*Ly - ry[i]; vy[i] = -vy[i]; }
    if (rz[i] < 0.0) { rz[i] = -rz[i];        vz[i] = -vz[i]; }
    if (rz[i] > Lz ) { rz[i] = 2.0*Lz - rz[i]; vz[i] = -vz[i]; }
}

/* ----------------------------------------------------------------------------
 *  One Boris pusher step for particle i.
 *
 *    a          = (F_coulomb + q*E) / m         (electric acceleration)
 *    v_minus    = v + a*(dt/2)                  (half kick)
 *    t          = (q/m)*(dt/2)*B                (rotation vector)
 *    v_prime    = v_minus + v_minus x t
 *    s          = 2 t / (1 + |t|^2)
 *    v_plus     = v_minus + v_prime x s         (exact rotation by the field)
 *    v_new      = v_plus  + a*(dt/2)            (second half kick)
 *    r_new      = r + v_new*dt
 *
 *  The magnetic rotation conserves |v| exactly, so it adds no spurious energy.
 * ------------------------------------------------------------------------- */
static inline void push_particle(int i, double dt)
{
    double inv_m = 1.0 / m[i];
    /* acceleration from the inter-particle Coulomb force PLUS the uniform
     * external electric field (F = q E).  The v x B part is not included here;
     * it is applied as the exact magnetic rotation below. */
    double ax = (fx[i] + q[i]*EX) * inv_m;
    double ay = (fy[i] + q[i]*EY) * inv_m;
    double az = (fz[i] + q[i]*EZ) * inv_m;

    /* first half electric kick */
    double vmx = vx[i] + ax * 0.5 * dt;
    double vmy = vy[i] + ay * 0.5 * dt;
    double vmz = vz[i] + az * 0.5 * dt;

    /* rotation vectors t and s */
    double tconst = q[i] * inv_m * 0.5 * dt;
    double tx = tconst * BX, ty = tconst * BY, tz = tconst * BZ;
    double t2 = tx*tx + ty*ty + tz*tz;
    double sfac = 2.0 / (1.0 + t2);
    double sx = tx * sfac, sy = ty * sfac, sz = tz * sfac;

    /* v' = v_minus + v_minus x t */
    double vpx = vmx + (vmy*tz - vmz*ty);
    double vpy = vmy + (vmz*tx - vmx*tz);
    double vpz = vmz + (vmx*ty - vmy*tx);

    /* v+ = v_minus + v' x s */
    double vplx = vmx + (vpy*sz - vpz*sy);
    double vply = vmy + (vpz*sx - vpx*sz);
    double vplz = vmz + (vpx*sy - vpy*sx);

    /* second half electric kick -> new velocity */
    vx[i] = vplx + ax * 0.5 * dt;
    vy[i] = vply + ay * 0.5 * dt;
    vz[i] = vplz + az * 0.5 * dt;

    /* position update using the end-of-step velocity */
    rx[i] += vx[i] * dt;
    ry[i] += vy[i] * dt;
    rz[i] += vz[i] * dt;

    reflect(i);
}

/* ----------------------------------------------------------------------------
 *  Total kinetic energy (0.5 m v^2 summed over all particles).
 * ------------------------------------------------------------------------- */
static double kinetic_energy(int NP)
{
    double ke = 0.0;
    #pragma omp parallel for schedule(static) reduction(+:ke)
    for (int i = 0; i < NP; ++i)
        ke += 0.5 * m[i] * (vx[i]*vx[i] + vy[i]*vy[i] + vz[i]*vz[i]);
    return ke;
}

/* ----------------------------------------------------------------------------
 *  Potential energy of the particles in the uniform external electric field:
 *      U_ext = Sum_i  -q_i (E . r_i)
 *  Its gradient is  -dU/dr_i = q_i E, exactly the force applied in
 *  push_particle().  This is the "energy-accounting fix": once the external
 *  field does work on the particles, KE + PE_coulomb alone is NOT conserved,
 *  but KE + PE_coulomb + U_ext is -- so the drift diagnostic uses that sum.
 *  The O(N) sum uses an OpenMP reduction, like kinetic_energy() above.
 * ------------------------------------------------------------------------- */
static double external_pe(int NP)
{
    double u = 0.0;
    #pragma omp parallel for schedule(static) reduction(+:u)
    for (int i = 0; i < NP; ++i)
        u -= q[i] * (EX*rx[i] + EY*ry[i] + EZ*rz[i]);
    return u;
}

/* ----------------------------------------------------------------------------
 *  Append the current frame's positions to the open binary file as float32.
 * ------------------------------------------------------------------------- */
static void write_frame(FILE *fp, int NP, float *buf)
{
    for (int i = 0; i < NP; ++i) {
        buf[3*i+0] = (float)rx[i];
        buf[3*i+1] = (float)ry[i];
        buf[3*i+2] = (float)rz[i];
    }
    fwrite(buf, sizeof(float), (size_t)NP*3, fp);
}

/* ----------------------------------------------------------------------------
 *  Format a double in a compact scientific style for use in file names:
 *  one digit after the decimal point, and an exponent with no leading zeros
 *  and no '+' sign, e.g.  1.0e-3, 1.5e-3, 1.0e5, 1.0e-12.  (C's default "%e"
 *  would give 1.0e-03 / 1.0e+05, which is uglier in a file name.)
 * ------------------------------------------------------------------------- */
static void fmt_exp(char *out, size_t n, double val)
{
    char tmp[24];                              /* "%.1e" is <= ~10 chars       */
    snprintf(tmp, sizeof tmp, "%.1e", val);    /* e.g. "1.0e-03" or "1.0e+05" */
    if (n == 0) return;

    /* Copy the string over verbatim, but as we pass the 'e' drop a '+' sign and
     * any leading zeros in the exponent (turning "1.0e+05" -> "1.0e5" and
     * "1.0e-03" -> "1.0e-3").  Plain bounded char copies -- no snprintf "%s%s"
     * of the scratch buffer, which the compiler cannot bound-check. */
    size_t o = 0;
    char *p = tmp;
    for (; *p && *p != 'e' && o + 1 < n; ++p) out[o++] = *p;   /* mantissa */
    if (*p == 'e') {
        if (o + 1 < n) out[o++] = 'e';
        char *ex = p + 1;
        if      (*ex == '-') { if (o + 1 < n) out[o++] = '-'; ex++; }
        else if (*ex == '+') { ex++; }
        while (ex[0] == '0' && ex[1] != '\0') ex++;            /* strip zeros */
        while (*ex && o + 1 < n) out[o++] = *ex++;
    }
    out[o] = '\0';
}

/* ----------------------------------------------------------------------------
 *  Build the parameter suffix that is appended to every output file name (just
 *  before the extension) so that runs with different parameters do not clobber
 *  one another's output.  Example, for N=500, Lx=1mm, Ly=2mm, Lz=1.5mm,
 *  DT=1e-12, 2000 steps, stride 10, T=1e5, B=(0.5,2,1):
 *      _N500_Lx1.0e-3_Ly2.0e-3_Lz1.5e-3_DT1.0e-12_stp2000_strd10_T1.0e5_Bx0.5_By2.0_Bz1.0_Ex0.0_Ey0.0_Ez0.0
 * ------------------------------------------------------------------------- */
static void build_suffix(char *out, size_t n)
{
    char sLx[32], sLy[32], sLz[32], sDT[32], sT[32];
    fmt_exp(sLx, sizeof sLx, Lx);
    fmt_exp(sLy, sizeof sLy, Ly);
    fmt_exp(sLz, sizeof sLz, Lz);
    fmt_exp(sDT, sizeof sDT, DT);
    fmt_exp(sT,  sizeof sT,  TEMP);
    snprintf(out, n,
             "_N%d_Lx%s_Ly%s_Lz%s_DT%s_stp%ld_strd%d_T%s_Bx%.1f_By%.1f_Bz%.1f"
             "_Ex%.1f_Ey%.1f_Ez%.1f",
             N, sLx, sLy, sLz, sDT, NSTEPS, SAVE_STRIDE, sT, BX, BY, BZ,
             EX, EY, EZ);
}

/* ----------------------------------------------------------------------------
 *  Three-way verdict string: "below" if x < lo, "above" if x >= hi, else "mid".
 * ------------------------------------------------------------------------- */
static const char *band(double x, double lo, double hi,
                        const char *below, const char *mid, const char *above)
{
    return x < lo ? below : (x < hi ? mid : above);
}

/* ----------------------------------------------------------------------------
 *  Print the dimensionless plasma parameters so any run shows, at a glance,
 *  *which physical regime it is actually in* -- crucially, whether the box is
 *  large enough (L >> lambda_D) and populated enough (N_Debye >> 1) for genuine
 *  collective plasma behaviour, or whether it is really just a few charged
 *  particles gyrating in a box.  All quantities are electron-based (the
 *  electrons set the Debye length and plasma frequency); T_e = T_i = TEMP here.
 *
 *      lambda_D = sqrt(eps0 kB T / (n_e e^2))          Debye (screening) length
 *      omega_pe = sqrt(n_e e^2 / (eps0 m_e))           electron plasma frequency
 *      Gamma    = (e^2 / 4 pi eps0 a) / (kB T)         Coulomb coupling (a = WS radius)
 *      N_Debye  = (4/3) pi n_e lambda_D^3              electrons per Debye sphere
 *
 *  Rules of thumb for a "real" (collective, weakly-coupled) plasma:
 *      L / lambda_D >> 1,   N_Debye >> 1,   Gamma << 1,   omega_pe*dt < ~0.2.
 * ------------------------------------------------------------------------- */
static void plasma_regime_report(void)
{
    const double eps0 = 1.0 / (4.0 * M_PI * KE_COUL);   /* vacuum permittivity  */
    const double V    = Lx * Ly * Lz;
    const double ne   = (double)N / V;                  /* electron density m^-3 */
    const double kT   = KB * TEMP;
    const double e2   = E_CHARGE * E_CHARGE;

    double lambda_D = sqrt(eps0 * kT / (ne * e2));
    double omega_pe = sqrt(ne * e2 / (eps0 * MASS_E));
    double a_ws     = cbrt(3.0 / (4.0 * M_PI * ne));    /* Wigner-Seitz radius   */
    double Gamma    = (e2 * KE_COUL / a_ws) / kT;
    double N_Debye  = (4.0 / 3.0) * M_PI * ne * lambda_D * lambda_D * lambda_D;
    double L_min    = Lx < Ly ? (Lx < Lz ? Lx : Lz) : (Ly < Lz ? Ly : Lz);
    double L_over_D = L_min / lambda_D;
    double wpe_dt   = omega_pe * DT;

    fprintf(stderr,
        "  --- plasma regime diagnostics (electron-based) ---\n"
        "  n_e         : %.3g m^-3\n"
        "  lambda_D    : %.3g m   (Debye screening length)\n"
        "  L/lambda_D  : %.3g   (%s)\n"
        "  N_Debye     : %.3g   (electrons per Debye sphere; %s)\n"
        "  Gamma       : %.3g   (Coulomb coupling; %s)\n"
        "  omega_pe*dt : %.3g   (%s)\n\n",
        ne, lambda_D,
        L_over_D, band(L_over_D, 1.0, 10.0,
                       "sub-Debye: no shielding / not a collective plasma",
                       "marginal: only a few Debye lengths across",
                       "collective: L >> lambda_D"),
        N_Debye, band(N_Debye, 1.0, 1.0e3,
                      "too few: discrete & collisional, not fluid-like",
                      "marginal",
                      "many per sphere -> weakly coupled, low discreteness noise"),
        Gamma, band(Gamma, 0.1, 1.0,
                    "weakly coupled (ideal-gas-like)",
                    "moderately coupled",
                    "strongly coupled (liquid-like)"),
        wpe_dt, wpe_dt < 0.2 ? "plasma oscillations time-resolved"
                             : "under-resolved: dt too large for omega_pe");

    /* External-field kinematics.  A perpendicular E gives the charge- and
     * mass-independent E x B drift (the whole plasma slides bodily); any
     * component of E along B instead accelerates electrons and protons in
     * opposite directions (free acceleration -> charge separation / current). */
    double Emag = sqrt(EX*EX + EY*EY + EZ*EZ);
    double Bmag = sqrt(BX*BX + BY*BY + BZ*BZ);
    if (Emag > 0.0) {
        if (Bmag > 0.0) {
            double B2   = Bmag * Bmag;
            double vdx  = (EY*BZ - EZ*BY) / B2;
            double vdy  = (EZ*BX - EX*BZ) / B2;
            double vdz  = (EX*BY - EY*BX) / B2;
            double vd   = sqrt(vdx*vdx + vdy*vdy + vdz*vdz);
            double Epar = fabs(EX*BX + EY*BY + EZ*BZ) / Bmag;   /* |E along B| */
            fprintf(stderr,
                "  --- external E field ---\n"
                "  |E|         : %.3g V/m\n"
                "  ExB drift   : %.3g m/s   (charge-independent, along E x B)\n"
                "  E_parallel  : %.3g V/m   (0 -> no runaway; >0 accelerates the\n"
                "                two species oppositely along B)\n\n",
                Emag, vd, Epar);
        } else {
            fprintf(stderr,
                "  --- external E field ---\n"
                "  |E|         : %.3g V/m   (no B: each species is uniformly\n"
                "                accelerated in opposite directions)\n\n",
                Emag);
        }
    }
}

/* ----------------------------------------------------------------------------
 *  Monotonic wall-clock time in seconds (immune to wall-clock adjustments).
 *  Used to measure execution time; CLOCK_MONOTONIC is also the right choice for
 *  the OpenMP build, so both programs report directly comparable numbers.
 * ------------------------------------------------------------------------- */
static double wall_seconds(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* ----------------------------------------------------------------------------
 *  Output directory.  Everything the simulation writes (trajectory, energy,
 *  metadata and the shared timing log) goes here; it defaults to "output" and
 *  can be overridden with the PLASMA_OUTDIR environment variable.  ensure_dir()
 *  creates it on demand so a fresh checkout / deleted folder just works.
 * ------------------------------------------------------------------------- */
static const char *output_dir(void)
{
    const char *d = getenv("PLASMA_OUTDIR");
    return (d && d[0]) ? d : "output";
}

static void ensure_dir(const char *d)
{
    if (mkdir(d, 0755) != 0 && errno != EEXIST)
        fprintf(stderr, "WARN: could not create output directory '%s': %s\n",
                d, strerror(errno));
}

/* ----------------------------------------------------------------------------
 *  Append one timing record to the shared, append-only "plasma_scaling.csv"
 *  (header written only when the file is first created).  This is the raw data
 *  later consumed by plot_scaling.py to build the strong- and weak-scaling
 *  curves, so every run -- serial or parallel, at any core count -- adds a row.
 *
 *      version           "serial" or "parallel"
 *      threads           cores used (1 for the serial build)
 *      wall_total_s      wall-clock time of the whole time-stepping loop
 *      force_s           time spent inside compute_forces() (the parallel kernel)
 *      per_step_ms       wall_total_s / nsteps, in milliseconds
 *      pair_updates_per_s  size-independent throughput of the force kernel
 *      energy_drift_pct  100*(E1-E0)/|E0|  (a correctness sanity check)
 * ------------------------------------------------------------------------- */
static void append_timing_row(const char *version, int threads, int NP,
                              double wall_total_s, double force_s,
                              double drift_pct)
{
    char fname[512];
    snprintf(fname, sizeof fname, "%s/plasma_scaling.csv", output_dir());
    int need_header = 1;
    FILE *chk = fopen(fname, "r");
    if (chk) {
        fseek(chk, 0, SEEK_END);
        if (ftell(chk) > 0) need_header = 0;
        fclose(chk);
    }
    FILE *f = fopen(fname, "a");
    if (!f) {
        fprintf(stderr, "WARN: could not open %s to record timing\n", fname);
        return;
    }
    if (need_header)
        fprintf(f, "version,threads,N,NP,nsteps,save_stride,Lx,Ly,Lz,dt,"
                   "temperature,Bz,wall_total_s,force_s,per_step_ms,"
                   "pair_updates_per_s,energy_drift_pct,unix_time\n");

    double per_step_ms = NSTEPS > 0 ? (wall_total_s / (double)NSTEPS) * 1e3 : 0.0;
    double npairs      = (double)NP * (double)(NP - 1) / 2.0;
    double pair_ups    = force_s > 0.0 ? npairs * (double)NSTEPS / force_s : 0.0;

    fprintf(f, "%s,%d,%d,%d,%ld,%d,%.9e,%.9e,%.9e,%.9e,%.9e,%.3f,"
               "%.9e,%.9e,%.6f,%.6e,%.6e,%ld\n",
            version, threads, N, NP, NSTEPS, SAVE_STRIDE,
            Lx, Ly, Lz, DT, TEMP, BZ,
            wall_total_s, force_s, per_step_ms, pair_ups, drift_pct,
            (long)time(NULL));
    fclose(f);
}

/* ========================================================================== */
int main(int argc, char **argv)
{
    /* optional positional overrides: N, nsteps, seed, nthreads */
    if (argc > 1) N      = atoi(argv[1]);
    if (argc > 2) NSTEPS = atol(argv[2]);
    if (argc > 3) SEED   = (unsigned int)strtoul(argv[3], NULL, 10);
    if (N < 1) { fprintf(stderr, "N must be >= 1\n"); return 1; }

    /* Thread count: explicit 4th arg wins; otherwise OpenMP's default (which
     * honours the OMP_NUM_THREADS environment variable, else the core count). */
    if (argc > 4) {
        int req = atoi(argv[4]);
        if (req > 0) omp_set_num_threads(req);
    }
    const int nthreads = omp_get_max_threads();

    /* Optional physics-parameter overrides from the environment (used by the web
     * UI so runs need no recompile).  Each is applied only when the variable is
     * set and non-empty, so an unset environment reproduces the defaults above.
     * Read here -- before initialise()/build_suffix() -- so the initial state and
     * the parameter-tagged filenames reflect the overridden values. */
    const char *s;
    if ((s = getenv("PLASMA_LX"))     && *s) Lx  = atof(s);
    if ((s = getenv("PLASMA_LY"))     && *s) Ly  = atof(s);
    if ((s = getenv("PLASMA_LZ"))     && *s) Lz  = atof(s);
    if ((s = getenv("PLASMA_DT"))     && *s) DT  = atof(s);
    if ((s = getenv("PLASMA_TEMP"))   && *s) TEMP = atof(s);
    if ((s = getenv("PLASMA_BX"))     && *s) BX  = atof(s);
    if ((s = getenv("PLASMA_BY"))     && *s) BY  = atof(s);
    if ((s = getenv("PLASMA_BZ"))     && *s) BZ  = atof(s);
    if ((s = getenv("PLASMA_EX"))     && *s) EX  = atof(s);
    if ((s = getenv("PLASMA_EY"))     && *s) EY  = atof(s);
    if ((s = getenv("PLASMA_EZ"))     && *s) EZ  = atof(s);
    if ((s = getenv("PLASMA_SOFT"))   && *s) SOFT = atof(s);
    if ((s = getenv("PLASMA_STRIDE")) && *s) SAVE_STRIDE = atoi(s);

    /* Timing-only mode (env PLASMA_TIMING_ONLY=1) skips every disk output except
     * the append-only plasma_scaling.csv timing row, so scaling sweeps stay
     * pure-compute and do not repeatedly rewrite the large trajectory files. */
    const char *to_env = getenv("PLASMA_TIMING_ONLY");
    const int timing_only = (to_env && to_env[0] && strcmp(to_env, "0") != 0);

    const int NP = 2 * N;
    srand48(SEED);
    allocate(NP);
    initialise(NP);

    /* Parameter-tagged output file names (built after any command-line
     * overrides to N / nsteps have been applied above). */
    const char *outdir = output_dir();
    ensure_dir(outdir);

    char suffix[256];
    build_suffix(suffix, sizeof suffix);
    char traj_name[512], en_name[512], meta_name[512];
    snprintf(traj_name, sizeof traj_name, "%s/plasma_traj%s.bin",   outdir, suffix);
    snprintf(en_name,   sizeof en_name,   "%s/plasma_energy%s.csv", outdir, suffix);
    snprintf(meta_name, sizeof meta_name, "%s/plasma_meta%s.txt",   outdir, suffix);

    FILE *ftraj = NULL, *fen = NULL;
    float *framebuf = NULL;
    if (!timing_only) {
        ftraj = fopen(traj_name, "wb");
        fen   = fopen(en_name, "w");
        if (!ftraj || !fen) { fprintf(stderr, "ERROR: cannot open output files\n"); return 1; }
        fprintf(fen, "step,time_s,KE_J,PE_J,Etot_J,T_kin_K,P_Pa,Uext_J\n");
        framebuf = malloc((size_t)NP * 3 * sizeof(float));
    }

    /* Box volume and running means of the two derived diagnostics (kinetic
     * temperature and pressure), averaged over the saved frames below. */
    const double VOLUME = Lx * Ly * Lz;
    double t_sum = 0.0, p_sum = 0.0;

    /* Initial forces + energy for the first frame / first push. */
    compute_forces(NP);
    double E0 = kinetic_energy(NP) + PE + external_pe(NP);

    long nframes = 0;
    fprintf(stderr,
            "Plasma simulation (OpenMP parallel, %d threads)\n"
            "  particles : %d electrons + %d protons = %d\n"
            "  box       : %.3g x %.3g x %.3g m\n"
            "  steps     : %ld,  dt = %.3g s,  total = %.3g s\n"
            "  B field   : (%.3g, %.3g, %.3g) T\n"
            "  E field   : (%.3g, %.3g, %.3g) V/m\n"
            "  T_plasma  : %.3g K,  save every %d steps\n\n",
            nthreads, N, N, NP, Lx, Ly, Lz, NSTEPS, DT, NSTEPS*DT, BX, BY, BZ,
            EX, EY, EZ, TEMP, SAVE_STRIDE);

    plasma_regime_report();

    /* Execution-time measurement: wall_start .. end of loop is the total time;
     * force_s accumulates the compute_forces() calls inside the loop (a clean
     * subset of the total, and the part the parallel build speeds up). */
    double force_s = 0.0;
    double wall_start = wall_seconds();

    for (long step = 0; step <= NSTEPS; ++step) {

        /* --- save a frame (positions + energy diagnostics) --- */
        if (!timing_only && step % SAVE_STRIDE == 0) {
            write_frame(ftraj, NP, framebuf);
            double ke   = kinetic_energy(NP);
            double uext = external_pe(NP);   /* PE in the external E field */

            /* Instantaneous kinetic temperature from equipartition
             *     (3/2) NP kB T_kin = KE                                     *
             * and the virial (mechanical) pressure                          *
             *     P = [ 2 KE + Sum_{i<j} r_ij.F_ij ] / (3 V).               *
             * The 2*KE term is the ideal-gas part (P = n kB T_kin); the     *
             * virial term is the Coulomb correction (repulsion raises P,    *
             * attraction lowers it).  The v x B force does no work and is   *
             * purely rotational, so it does not enter the scalar pressure.  *
             * The uniform external E is a body force (not a wall/pair       *
             * interaction), so it likewise is not part of this virial       *
             * pressure.                                                     */
            double tkin = 2.0 * ke / (3.0 * NP * KB);
            double pres = (2.0 * ke + VIRIAL) / (3.0 * VOLUME);
            t_sum += tkin;  p_sum += pres;

            /* Etot includes U_ext so it is the conserved quantity even when  *
             * the external field does work; U_ext is also logged on its own. */
            fprintf(fen, "%ld,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e\n",
                    step, step*DT, ke, PE, ke + PE + uext, tkin, pres, uext);
            nframes++;
        }

        if (step == NSTEPS) break;   /* last iteration only records the frame */

        /* --- advance all particles by one time step (each i independent) --- *
         * Forces are already current for this step (computed at the end of
         * the previous iteration, or before the loop for step 0).            */
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < NP; ++i) push_particle(i, DT);

        /* --- recompute forces at the new positions for the next step --- */
        double tf0 = wall_seconds();
        compute_forces(NP);
        force_s += wall_seconds() - tf0;

        if (step % 200 == 0)
            fprintf(stderr, "  step %6ld / %ld\n", step, NSTEPS);
    }

    double wall_total_s = wall_seconds() - wall_start;
    double E1 = kinetic_energy(NP) + PE + external_pe(NP);
    double drift_pct = 100.0 * (E1 - E0) / fabs(E0);
    double t_mean = nframes ? t_sum / nframes : 0.0;   /* mean kinetic temp [K] */
    double p_mean = nframes ? p_sum / nframes : 0.0;   /* mean pressure     [Pa]*/

    fprintf(stderr,
            "\nDone (OpenMP parallel, %d threads).\n"
            "  timing  : wall %.4f s total, %.4f s in force kernel (%.1f%%)\n"
            "            %.4f ms/step, %.3g pair-updates/s\n"
            "  energy  : E0 = %.6e J, E1 = %.6e J, drift = %.3g%%\n",
            nthreads, wall_total_s, force_s,
            wall_total_s > 0.0 ? 100.0 * force_s / wall_total_s : 0.0,
            NSTEPS > 0 ? wall_total_s / (double)NSTEPS * 1e3 : 0.0,
            force_s > 0.0 ? ((double)NP*(NP-1)/2.0)*(double)NSTEPS/force_s : 0.0,
            E0, E1, drift_pct);

    /* Always record the timing row (for later strong/weak scaling plots). */
    append_timing_row("parallel", nthreads, NP, wall_total_s, force_s, drift_pct);

    if (!timing_only) {
        fprintf(stderr,
            "  T_kin   : %.4g K (mean over frames; set point %.4g K)\n"
            "  pressure: %.4g Pa (mean virial pressure over frames)\n"
            "  frames  : %ld written\n"
            "  output  : %s\n            %s\n            %s\n",
            t_mean, TEMP, p_mean, nframes, traj_name, en_name, meta_name);

        /* --- write metadata last, now that nframes is known --- */
        FILE *fmeta = fopen(meta_name, "w");
        fprintf(fmeta,
                "N=%d\nNP=%d\nnframes=%ld\nsave_stride=%d\nnsteps=%ld\n"
                "dt=%.9e\nLx=%.9e\nLy=%.9e\nLz=%.9e\n"
                "Bx=%.9e\nBy=%.9e\nBz=%.9e\nEx=%.9e\nEy=%.9e\nEz=%.9e\n"
                "temperature=%.9e\nseed=%u\n"
                "volume=%.9e\nT_kinetic=%.9e\npressure=%.9e\n",
                N, NP, nframes, SAVE_STRIDE, NSTEPS,
                DT, Lx, Ly, Lz, BX, BY, BZ, EX, EY, EZ, TEMP, SEED,
                VOLUME, t_mean, p_mean);
        fclose(fmeta);

        fclose(ftraj);
        fclose(fen);
        free(framebuf);
    } else {
        fprintf(stderr,
            "  (timing-only mode: trajectory / energy / meta output skipped)\n");
    }

    free_all();
    return 0;
}

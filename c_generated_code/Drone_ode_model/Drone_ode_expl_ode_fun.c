/* This file was automatically generated by CasADi.
   The CasADi copyright holders make no ownership claim of its contents. */
#ifdef __cplusplus
extern "C" {
#endif

/* How to prefix internal symbols */
#ifdef CASADI_CODEGEN_PREFIX
  #define CASADI_NAMESPACE_CONCAT(NS, ID) _CASADI_NAMESPACE_CONCAT(NS, ID)
  #define _CASADI_NAMESPACE_CONCAT(NS, ID) NS ## ID
  #define CASADI_PREFIX(ID) CASADI_NAMESPACE_CONCAT(CODEGEN_PREFIX, ID)
#else
  #define CASADI_PREFIX(ID) Drone_ode_expl_ode_fun_ ## ID
#endif

#include <math.h>

#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int int
#endif

/* Add prefix to internal symbols */
#define casadi_c0 CASADI_PREFIX(c0)
#define casadi_clear CASADI_PREFIX(clear)
#define casadi_copy CASADI_PREFIX(copy)
#define casadi_dot CASADI_PREFIX(dot)
#define casadi_f0 CASADI_PREFIX(f0)
#define casadi_house CASADI_PREFIX(house)
#define casadi_if_else CASADI_PREFIX(if_else)
#define casadi_qr CASADI_PREFIX(qr)
#define casadi_qr_colcomb CASADI_PREFIX(qr_colcomb)
#define casadi_qr_mv CASADI_PREFIX(qr_mv)
#define casadi_qr_singular CASADI_PREFIX(qr_singular)
#define casadi_qr_solve CASADI_PREFIX(qr_solve)
#define casadi_qr_trs CASADI_PREFIX(qr_trs)
#define casadi_s0 CASADI_PREFIX(s0)
#define casadi_s1 CASADI_PREFIX(s1)
#define casadi_s2 CASADI_PREFIX(s2)
#define casadi_s3 CASADI_PREFIX(s3)
#define casadi_s4 CASADI_PREFIX(s4)
#define casadi_s5 CASADI_PREFIX(s5)
#define casadi_s6 CASADI_PREFIX(s6)
#define casadi_scal CASADI_PREFIX(scal)

/* Symbol visibility in DLLs */
#ifndef CASADI_SYMBOL_EXPORT
  #if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
    #if defined(STATIC_LINKED)
      #define CASADI_SYMBOL_EXPORT
    #else
      #define CASADI_SYMBOL_EXPORT __declspec(dllexport)
    #endif
  #elif defined(__GNUC__) && defined(GCC_HASCLASSVISIBILITY)
    #define CASADI_SYMBOL_EXPORT __attribute__ ((visibility ("default")))
  #else
    #define CASADI_SYMBOL_EXPORT
  #endif
#endif

void casadi_clear(casadi_real* x, casadi_int n) {
  casadi_int i;
  if (x) {
    for (i=0; i<n; ++i) *x++ = 0;
  }
}

void casadi_copy(const casadi_real* x, casadi_int n, casadi_real* y) {
  casadi_int i;
  if (y) {
    if (x) {
      for (i=0; i<n; ++i) *y++ = *x++;
    } else {
      for (i=0; i<n; ++i) *y++ = 0.;
    }
  }
}

casadi_real casadi_if_else(casadi_real c, casadi_real x, casadi_real y) { return c!=0 ? x : y;}

void casadi_scal(casadi_int n, casadi_real alpha, casadi_real* x) {
  casadi_int i;
  if (!x) return;
  for (i=0; i<n; ++i) *x++ *= alpha;
}

casadi_real casadi_dot(casadi_int n, const casadi_real* x, const casadi_real* y) {
  casadi_int i;
  casadi_real r = 0;
  for (i=0; i<n; ++i) r += *x++ * *y++;
  return r;
}

casadi_real casadi_house(casadi_real* v, casadi_real* beta, casadi_int nv) {
  casadi_int i;
  casadi_real v0, sigma, s, sigma_is_zero, v0_nonpos;
  v0 = v[0];
  sigma=0;
  for (i=1; i<nv; ++i) sigma += v[i]*v[i];
  s = sqrt(v0*v0 + sigma);
  sigma_is_zero = sigma==0;
  v0_nonpos = v0<=0;
  v[0] = casadi_if_else(sigma_is_zero, 1,
                 casadi_if_else(v0_nonpos, v0-s, -sigma/(v0+s)));
  *beta = casadi_if_else(sigma_is_zero, 2*v0_nonpos, -1/(s*v[0]));
  return s;
}
void casadi_qr(const casadi_int* sp_a, const casadi_real* nz_a, casadi_real* x,
               const casadi_int* sp_v, casadi_real* nz_v, const casadi_int* sp_r, casadi_real* nz_r, casadi_real* beta,
               const casadi_int* prinv, const casadi_int* pc) {
   casadi_int ncol, nrow, r, c, k, k1;
   casadi_real alpha;
   const casadi_int *a_colind, *a_row, *v_colind, *v_row, *r_colind, *r_row;
   ncol = sp_a[1];
   a_colind=sp_a+2; a_row=sp_a+2+ncol+1;
   nrow = sp_v[0];
   v_colind=sp_v+2; v_row=sp_v+2+ncol+1;
   r_colind=sp_r+2; r_row=sp_r+2+ncol+1;
   for (r=0; r<nrow; ++r) x[r] = 0;
   for (c=0; c<ncol; ++c) {
     for (k=a_colind[pc[c]]; k<a_colind[pc[c]+1]; ++k) x[prinv[a_row[k]]] = nz_a[k];
     for (k=r_colind[c]; k<r_colind[c+1] && (r=r_row[k])<c; ++k) {
       alpha = 0;
       for (k1=v_colind[r]; k1<v_colind[r+1]; ++k1) alpha += nz_v[k1]*x[v_row[k1]];
       alpha *= beta[r];
       for (k1=v_colind[r]; k1<v_colind[r+1]; ++k1) x[v_row[k1]] -= alpha*nz_v[k1];
       *nz_r++ = x[r];
       x[r] = 0;
     }
     for (k=v_colind[c]; k<v_colind[c+1]; ++k) {
       nz_v[k] = x[v_row[k]];
       x[v_row[k]] = 0;
     }
     *nz_r++ = casadi_house(nz_v + v_colind[c], beta + c, v_colind[c+1] - v_colind[c]);
   }
 }
void casadi_qr_mv(const casadi_int* sp_v, const casadi_real* v, const casadi_real* beta, casadi_real* x,
                  casadi_int tr) {
  casadi_int ncol, c, c1, k;
  casadi_real alpha;
  const casadi_int *colind, *row;
  ncol=sp_v[1];
  colind=sp_v+2; row=sp_v+2+ncol+1;
  for (c1=0; c1<ncol; ++c1) {
    c = tr ? c1 : ncol-1-c1;
    alpha=0;
    for (k=colind[c]; k<colind[c+1]; ++k) alpha += v[k]*x[row[k]];
    alpha *= beta[c];
    for (k=colind[c]; k<colind[c+1]; ++k) x[row[k]] -= alpha*v[k];
  }
}
void casadi_qr_trs(const casadi_int* sp_r, const casadi_real* nz_r, casadi_real* x, casadi_int tr) {
  casadi_int ncol, r, c, k;
  const casadi_int *colind, *row;
  ncol=sp_r[1];
  colind=sp_r+2; row=sp_r+2+ncol+1;
  if (tr) {
    for (c=0; c<ncol; ++c) {
      for (k=colind[c]; k<colind[c+1]; ++k) {
        r = row[k];
        if (r==c) {
          x[c] /= nz_r[k];
        } else {
          x[c] -= nz_r[k]*x[r];
        }
      }
    }
  } else {
    for (c=ncol-1; c>=0; --c) {
      for (k=colind[c+1]-1; k>=colind[c]; --k) {
        r=row[k];
        if (r==c) {
          x[r] /= nz_r[k];
        } else {
          x[r] -= nz_r[k]*x[c];
        }
      }
    }
  }
}
void casadi_qr_solve(casadi_real* x, casadi_int nrhs, casadi_int tr,
                     const casadi_int* sp_v, const casadi_real* v, const casadi_int* sp_r, const casadi_real* r,
                     const casadi_real* beta, const casadi_int* prinv, const casadi_int* pc, casadi_real* w) {
  casadi_int k, c, nrow_ext, ncol;
  nrow_ext = sp_v[0]; ncol = sp_v[1];
  for (k=0; k<nrhs; ++k) {
    if (tr) {
      for (c=0; c<ncol; ++c) w[c] = x[pc[c]];
      casadi_qr_trs(sp_r, r, w, 1);
      casadi_qr_mv(sp_v, v, beta, w, 0);
      for (c=0; c<ncol; ++c) x[c] = w[prinv[c]];
    } else {
      for (c=0; c<nrow_ext; ++c) w[c] = 0;
      for (c=0; c<ncol; ++c) w[prinv[c]] = x[c];
      casadi_qr_mv(sp_v, v, beta, w, 1);
      casadi_qr_trs(sp_r, r, w, 0);
      for (c=0; c<ncol; ++c) x[pc[c]] = w[c];
    }
    x += ncol;
  }
}
casadi_int casadi_qr_singular(casadi_real* rmin, casadi_int* irmin, const casadi_real* nz_r,
                             const casadi_int* sp_r, const casadi_int* pc, casadi_real eps) {
  casadi_real rd, rd_min;
  casadi_int ncol, c, nullity;
  const casadi_int* r_colind;
  nullity = 0;
  ncol = sp_r[1];
  r_colind = sp_r + 2;
  for (c=0; c<ncol; ++c) {
    rd = fabs(nz_r[r_colind[c+1]-1]);
    if (rd<eps) nullity++;
    if (c==0 || rd < rd_min) {
      rd_min = rd;
      if (rmin) *rmin = rd;
      if (irmin) *irmin = pc[c];
    }
  }
  return nullity;
}
void casadi_qr_colcomb(casadi_real* v, const casadi_real* nz_r, const casadi_int* sp_r,
                       const casadi_int* pc, casadi_real eps, casadi_int ind) {
  casadi_int ncol, r, c, k;
  const casadi_int *r_colind, *r_row;
  ncol = sp_r[1];
  r_colind = sp_r + 2;
  r_row = r_colind + ncol + 1;
  for (c=0; c<ncol; ++c) {
    if (fabs(nz_r[r_colind[c+1]-1])<eps && 0==ind--) {
      ind = c;
      break;
    }
  }
  casadi_clear(v, ncol);
  v[pc[ind]] = 1.;
  for (k=r_colind[ind]; k<r_colind[ind+1]-1; ++k) {
    v[pc[r_row[k]]] = -nz_r[k];
  }
  for (c=ind-1; c>=0; --c) {
    for (k=r_colind[c+1]-1; k>=r_colind[c]; --k) {
      r=r_row[k];
      if (r==c) {
        if (fabs(nz_r[k])<eps) {
          v[pc[r]] = 0;
        } else {
          v[pc[r]] /= nz_r[k];
        }
      } else {
        v[pc[r]] -= nz_r[k]*v[pc[c]];
      }
    }
  }
  casadi_scal(ncol, 1./sqrt(casadi_dot(ncol, v, v)), v);
}

static const casadi_int casadi_s0[4] = {0, 1, 2, 3};
static const casadi_int casadi_s1[23] = {4, 4, 0, 4, 8, 12, 16, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
static const casadi_int casadi_s2[17] = {4, 4, 0, 4, 7, 9, 10, 0, 1, 2, 3, 1, 2, 3, 2, 3, 3};
static const casadi_int casadi_s3[17] = {4, 4, 0, 1, 3, 6, 10, 0, 0, 1, 0, 1, 2, 0, 1, 2, 3};
static const casadi_int casadi_s4[12] = {8, 1, 0, 8, 0, 1, 2, 3, 4, 5, 6, 7};
static const casadi_int casadi_s5[8] = {4, 1, 0, 4, 0, 1, 2, 3};
static const casadi_int casadi_s6[16] = {12, 1, 0, 12, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

static const casadi_real casadi_c0[16] = {1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.};

/* Drone_ode_expl_ode_fun:(i0[8],i1[4],i2[12])->(o0[8]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_int i, j, k;
  casadi_real *rr, *ss, *tt;
  const casadi_real *cs;
  casadi_real *w0=w+8, *w1=w+16, *w2=w+32, w3, w4, *w5=w+50, *w6=w+82, *w7=w+114, *w8=w+130, *w9=w+146, w10, *w11=w+163, *w12=w+195, *w13=w+259, w14, w15, w16, w17, w18, *w19=w+328, *w20=w+336;
  /* #0: @0 = zeros(8x1) */
  casadi_clear(w0, 8);
  /* #1: @1 = zeros(4x4) */
  casadi_clear(w1, 16);
  /* #2: @2 = zeros(4x4) */
  casadi_clear(w2, 16);
  /* #3: @3 = input[0][3] */
  w3 = arg[0] ? arg[0][3] : 0;
  /* #4: @4 = cos(@3) */
  w4 = cos( w3 );
  /* #5: (@2[0] = @4) */
  for (rr=w2+0, ss=(&w4); rr!=w2+1; rr+=1) *rr = *ss++;
  /* #6: @4 = sin(@3) */
  w4 = sin( w3 );
  /* #7: @4 = (-@4) */
  w4 = (- w4 );
  /* #8: (@2[4] = @4) */
  for (rr=w2+4, ss=(&w4); rr!=w2+5; rr+=1) *rr = *ss++;
  /* #9: @4 = 0 */
  w4 = 0.;
  /* #10: (@2[8] = @4) */
  for (rr=w2+8, ss=(&w4); rr!=w2+9; rr+=1) *rr = *ss++;
  /* #11: @4 = 0 */
  w4 = 0.;
  /* #12: (@2[12] = @4) */
  for (rr=w2+12, ss=(&w4); rr!=w2+13; rr+=1) *rr = *ss++;
  /* #13: @4 = sin(@3) */
  w4 = sin( w3 );
  /* #14: (@2[1] = @4) */
  for (rr=w2+1, ss=(&w4); rr!=w2+2; rr+=1) *rr = *ss++;
  /* #15: @4 = cos(@3) */
  w4 = cos( w3 );
  /* #16: (@2[5] = @4) */
  for (rr=w2+5, ss=(&w4); rr!=w2+6; rr+=1) *rr = *ss++;
  /* #17: @4 = 0 */
  w4 = 0.;
  /* #18: (@2[9] = @4) */
  for (rr=w2+9, ss=(&w4); rr!=w2+10; rr+=1) *rr = *ss++;
  /* #19: @4 = 0 */
  w4 = 0.;
  /* #20: (@2[13] = @4) */
  for (rr=w2+13, ss=(&w4); rr!=w2+14; rr+=1) *rr = *ss++;
  /* #21: @4 = 0 */
  w4 = 0.;
  /* #22: (@2[2] = @4) */
  for (rr=w2+2, ss=(&w4); rr!=w2+3; rr+=1) *rr = *ss++;
  /* #23: @4 = 0 */
  w4 = 0.;
  /* #24: (@2[6] = @4) */
  for (rr=w2+6, ss=(&w4); rr!=w2+7; rr+=1) *rr = *ss++;
  /* #25: @4 = 1 */
  w4 = 1.;
  /* #26: (@2[10] = @4) */
  for (rr=w2+10, ss=(&w4); rr!=w2+11; rr+=1) *rr = *ss++;
  /* #27: @4 = 0 */
  w4 = 0.;
  /* #28: (@2[14] = @4) */
  for (rr=w2+14, ss=(&w4); rr!=w2+15; rr+=1) *rr = *ss++;
  /* #29: @4 = 0 */
  w4 = 0.;
  /* #30: (@2[3] = @4) */
  for (rr=w2+3, ss=(&w4); rr!=w2+4; rr+=1) *rr = *ss++;
  /* #31: @4 = 0 */
  w4 = 0.;
  /* #32: (@2[7] = @4) */
  for (rr=w2+7, ss=(&w4); rr!=w2+8; rr+=1) *rr = *ss++;
  /* #33: @4 = 0 */
  w4 = 0.;
  /* #34: (@2[11] = @4) */
  for (rr=w2+11, ss=(&w4); rr!=w2+12; rr+=1) *rr = *ss++;
  /* #35: @4 = 1 */
  w4 = 1.;
  /* #36: (@2[15] = @4) */
  for (rr=w2+15, ss=(&w4); rr!=w2+16; rr+=1) *rr = *ss++;
  /* #37: @5 = horzcat(@1, @2) */
  rr=w5;
  for (i=0, cs=w1; i<16; ++i) *rr++ = *cs++;
  for (i=0, cs=w2; i<16; ++i) *rr++ = *cs++;
  /* #38: @6 = @5' */
  for (i=0, rr=w6, cs=w5; i<8; ++i) for (j=0; j<4; ++j) rr[i+j*8] = *cs++;
  /* #39: @1 = zeros(4x4) */
  casadi_clear(w1, 16);
  /* #40: @2 = zeros(4x4) */
  casadi_clear(w2, 16);
  /* #41: @7 = 
  [[1, 0, 0, 0], 
   [0, 1, 0, 0], 
   [0, 0, 1, 0], 
   [0, 0, 0, 1]] */
  casadi_copy(casadi_c0, 16, w7);
  /* #42: @8 = zeros(4x4) */
  casadi_clear(w8, 16);
  /* #43: @4 = 0.6756 */
  w4 = 6.7559999999999998e-01;
  /* #44: (@8[0] = @4) */
  for (rr=w8+0, ss=(&w4); rr!=w8+1; rr+=1) *rr = *ss++;
  /* #45: @4 = 0 */
  w4 = 0.;
  /* #46: (@8[4] = @4) */
  for (rr=w8+4, ss=(&w4); rr!=w8+5; rr+=1) *rr = *ss++;
  /* #47: @4 = 0 */
  w4 = 0.;
  /* #48: (@8[8] = @4) */
  for (rr=w8+8, ss=(&w4); rr!=w8+9; rr+=1) *rr = *ss++;
  /* #49: @4 = 0 */
  w4 = 0.;
  /* #50: (@8[12] = @4) */
  for (rr=w8+12, ss=(&w4); rr!=w8+13; rr+=1) *rr = *ss++;
  /* #51: @4 = 0 */
  w4 = 0.;
  /* #52: (@8[1] = @4) */
  for (rr=w8+1, ss=(&w4); rr!=w8+2; rr+=1) *rr = *ss++;
  /* #53: @4 = 0.6344 */
  w4 = 6.3439999999999996e-01;
  /* #54: (@8[5] = @4) */
  for (rr=w8+5, ss=(&w4); rr!=w8+6; rr+=1) *rr = *ss++;
  /* #55: @4 = 0 */
  w4 = 0.;
  /* #56: (@8[9] = @4) */
  for (rr=w8+9, ss=(&w4); rr!=w8+10; rr+=1) *rr = *ss++;
  /* #57: @4 = 0 */
  w4 = 0.;
  /* #58: (@8[13] = @4) */
  for (rr=w8+13, ss=(&w4); rr!=w8+14; rr+=1) *rr = *ss++;
  /* #59: @4 = 0 */
  w4 = 0.;
  /* #60: (@8[2] = @4) */
  for (rr=w8+2, ss=(&w4); rr!=w8+3; rr+=1) *rr = *ss++;
  /* #61: @4 = 0 */
  w4 = 0.;
  /* #62: (@8[6] = @4) */
  for (rr=w8+6, ss=(&w4); rr!=w8+7; rr+=1) *rr = *ss++;
  /* #63: @4 = 0.408 */
  w4 = 4.0799999999999997e-01;
  /* #64: (@8[10] = @4) */
  for (rr=w8+10, ss=(&w4); rr!=w8+11; rr+=1) *rr = *ss++;
  /* #65: @4 = 0 */
  w4 = 0.;
  /* #66: (@8[14] = @4) */
  for (rr=w8+14, ss=(&w4); rr!=w8+15; rr+=1) *rr = *ss++;
  /* #67: @4 = 0 */
  w4 = 0.;
  /* #68: (@8[3] = @4) */
  for (rr=w8+3, ss=(&w4); rr!=w8+4; rr+=1) *rr = *ss++;
  /* #69: @4 = 0 */
  w4 = 0.;
  /* #70: (@8[7] = @4) */
  for (rr=w8+7, ss=(&w4); rr!=w8+8; rr+=1) *rr = *ss++;
  /* #71: @4 = 0 */
  w4 = 0.;
  /* #72: (@8[11] = @4) */
  for (rr=w8+11, ss=(&w4); rr!=w8+12; rr+=1) *rr = *ss++;
  /* #73: @4 = 0.2953 */
  w4 = 2.9530000000000001e-01;
  /* #74: (@8[15] = @4) */
  for (rr=w8+15, ss=(&w4); rr!=w8+16; rr+=1) *rr = *ss++;
  /* #75: @7 = (@8\@7) */
  rr = w7;
  ss = w8;
  {
    /* FIXME(@jaeandersson): Memory allocation can be avoided */
    casadi_real v[10], r[10], beta[4], w[8];
    casadi_qr(casadi_s1, ss, w, casadi_s2, v, casadi_s3, r, beta, casadi_s0, casadi_s0);
    casadi_qr_solve(rr, 4, 0, casadi_s2, v, casadi_s3, r, beta, casadi_s0, casadi_s0, w);
  }
  /* #76: @9 = zeros(4x4) */
  casadi_clear(w9, 16);
  /* #77: @4 = 0.5941 */
  w4 = 5.9409999999999996e-01;
  /* #78: (@9[0] = @4) */
  for (rr=w9+0, ss=(&w4); rr!=w9+1; rr+=1) *rr = *ss++;
  /* #79: @4 = -0.8109 */
  w4 = -8.1089999999999995e-01;
  /* #80: @10 = input[0][7] */
  w10 = arg[0] ? arg[0][7] : 0;
  /* #81: @4 = (@4*@10) */
  w4 *= w10;
  /* #82: (@9[4] = @4) */
  for (rr=w9+4, ss=(&w4); rr!=w9+5; rr+=1) *rr = *ss++;
  /* #83: @4 = 0 */
  w4 = 0.;
  /* #84: (@9[8] = @4) */
  for (rr=w9+8, ss=(&w4); rr!=w9+9; rr+=1) *rr = *ss++;
  /* #85: @4 = 0 */
  w4 = 0.;
  /* #86: (@9[12] = @4) */
  for (rr=w9+12, ss=(&w4); rr!=w9+13; rr+=1) *rr = *ss++;
  /* #87: @4 = 0.3984 */
  w4 = 3.9839999999999998e-01;
  /* #88: @4 = (@4*@10) */
  w4 *= w10;
  /* #89: (@9[1] = @4) */
  for (rr=w9+1, ss=(&w4); rr!=w9+2; rr+=1) *rr = *ss++;
  /* #90: @4 = 0.704 */
  w4 = 7.0399999999999996e-01;
  /* #91: (@9[5] = @4) */
  for (rr=w9+5, ss=(&w4); rr!=w9+6; rr+=1) *rr = *ss++;
  /* #92: @4 = 0 */
  w4 = 0.;
  /* #93: (@9[9] = @4) */
  for (rr=w9+9, ss=(&w4); rr!=w9+10; rr+=1) *rr = *ss++;
  /* #94: @4 = 0 */
  w4 = 0.;
  /* #95: (@9[13] = @4) */
  for (rr=w9+13, ss=(&w4); rr!=w9+14; rr+=1) *rr = *ss++;
  /* #96: @4 = 0 */
  w4 = 0.;
  /* #97: (@9[2] = @4) */
  for (rr=w9+2, ss=(&w4); rr!=w9+3; rr+=1) *rr = *ss++;
  /* #98: @4 = 0 */
  w4 = 0.;
  /* #99: (@9[6] = @4) */
  for (rr=w9+6, ss=(&w4); rr!=w9+7; rr+=1) *rr = *ss++;
  /* #100: @4 = 0.9365 */
  w4 = 9.3650000000000000e-01;
  /* #101: (@9[10] = @4) */
  for (rr=w9+10, ss=(&w4); rr!=w9+11; rr+=1) *rr = *ss++;
  /* #102: @4 = 0 */
  w4 = 0.;
  /* #103: (@9[14] = @4) */
  for (rr=w9+14, ss=(&w4); rr!=w9+15; rr+=1) *rr = *ss++;
  /* #104: @4 = 0 */
  w4 = 0.;
  /* #105: (@9[3] = @4) */
  for (rr=w9+3, ss=(&w4); rr!=w9+4; rr+=1) *rr = *ss++;
  /* #106: @4 = 0 */
  w4 = 0.;
  /* #107: (@9[7] = @4) */
  for (rr=w9+7, ss=(&w4); rr!=w9+8; rr+=1) *rr = *ss++;
  /* #108: @4 = 0 */
  w4 = 0.;
  /* #109: (@9[11] = @4) */
  for (rr=w9+11, ss=(&w4); rr!=w9+12; rr+=1) *rr = *ss++;
  /* #110: @4 = 0.9752 */
  w4 = 9.7519999999999996e-01;
  /* #111: (@9[15] = @4) */
  for (rr=w9+15, ss=(&w4); rr!=w9+16; rr+=1) *rr = *ss++;
  /* #112: @2 = mac(@7,@9,@2) */
  for (i=0, rr=w2; i<4; ++i) for (j=0; j<4; ++j, ++rr) for (k=0, ss=w7+j, tt=w9+i*4; k<4; ++k) *rr += ss[k*4]**tt++;
  /* #113: @2 = (-@2) */
  for (i=0, rr=w2, cs=w2; i<16; ++i) *rr++ = (- *cs++ );
  /* #114: @5 = horzcat(@1, @2) */
  rr=w5;
  for (i=0, cs=w1; i<16; ++i) *rr++ = *cs++;
  for (i=0, cs=w2; i<16; ++i) *rr++ = *cs++;
  /* #115: @11 = @5' */
  for (i=0, rr=w11, cs=w5; i<8; ++i) for (j=0; j<4; ++j) rr[i+j*8] = *cs++;
  /* #116: @12 = horzcat(@6, @11) */
  rr=w12;
  for (i=0, cs=w6; i<32; ++i) *rr++ = *cs++;
  for (i=0, cs=w11; i<32; ++i) *rr++ = *cs++;
  /* #117: @13 = @12' */
  for (i=0, rr=w13, cs=w12; i<8; ++i) for (j=0; j<8; ++j) rr[i+j*8] = *cs++;
  /* #118: @4 = input[0][0] */
  w4 = arg[0] ? arg[0][0] : 0;
  /* #119: @14 = input[0][1] */
  w14 = arg[0] ? arg[0][1] : 0;
  /* #120: @15 = input[0][2] */
  w15 = arg[0] ? arg[0][2] : 0;
  /* #121: @16 = input[0][4] */
  w16 = arg[0] ? arg[0][4] : 0;
  /* #122: @17 = input[0][5] */
  w17 = arg[0] ? arg[0][5] : 0;
  /* #123: @18 = input[0][6] */
  w18 = arg[0] ? arg[0][6] : 0;
  /* #124: @19 = vertcat(@4, @14, @15, @3, @16, @17, @18, @10) */
  rr=w19;
  *rr++ = w4;
  *rr++ = w14;
  *rr++ = w15;
  *rr++ = w3;
  *rr++ = w16;
  *rr++ = w17;
  *rr++ = w18;
  *rr++ = w10;
  /* #125: @0 = mac(@13,@19,@0) */
  for (i=0, rr=w0; i<1; ++i) for (j=0; j<8; ++j, ++rr) for (k=0, ss=w13+j, tt=w19+i*8; k<8; ++k) *rr += ss[k*8]**tt++;
  /* #126: @19 = zeros(8x1) */
  casadi_clear(w19, 8);
  /* #127: @1 = zeros(4x4) */
  casadi_clear(w1, 16);
  /* #128: @2 = 
  [[1, 0, 0, 0], 
   [0, 1, 0, 0], 
   [0, 0, 1, 0], 
   [0, 0, 0, 1]] */
  casadi_copy(casadi_c0, 16, w2);
  /* #129: @2 = (@8\@2) */
  rr = w2;
  ss = w8;
  {
    /* FIXME(@jaeandersson): Memory allocation can be avoided */
    casadi_real v[10], r[10], beta[4], w[8];
    casadi_qr(casadi_s1, ss, w, casadi_s2, v, casadi_s3, r, beta, casadi_s0, casadi_s0);
    casadi_qr_solve(rr, 4, 0, casadi_s2, v, casadi_s3, r, beta, casadi_s0, casadi_s0, w);
  }
  /* #130: @8 = @2' */
  for (i=0, rr=w8, cs=w2; i<4; ++i) for (j=0; j<4; ++j) rr[i+j*4] = *cs++;
  /* #131: @6 = horzcat(@1, @8) */
  rr=w6;
  for (i=0, cs=w1; i<16; ++i) *rr++ = *cs++;
  for (i=0, cs=w8; i<16; ++i) *rr++ = *cs++;
  /* #132: @11 = @6' */
  for (i=0, rr=w11, cs=w6; i<8; ++i) for (j=0; j<4; ++j) rr[i+j*8] = *cs++;
  /* #133: @4 = input[1][0] */
  w4 = arg[1] ? arg[1][0] : 0;
  /* #134: @14 = input[1][1] */
  w14 = arg[1] ? arg[1][1] : 0;
  /* #135: @15 = input[1][2] */
  w15 = arg[1] ? arg[1][2] : 0;
  /* #136: @3 = input[1][3] */
  w3 = arg[1] ? arg[1][3] : 0;
  /* #137: @20 = vertcat(@4, @14, @15, @3) */
  rr=w20;
  *rr++ = w4;
  *rr++ = w14;
  *rr++ = w15;
  *rr++ = w3;
  /* #138: @19 = mac(@11,@20,@19) */
  for (i=0, rr=w19; i<1; ++i) for (j=0; j<8; ++j, ++rr) for (k=0, ss=w11+j, tt=w20+i*4; k<4; ++k) *rr += ss[k*8]**tt++;
  /* #139: @0 = (@0+@19) */
  for (i=0, rr=w0, cs=w19; i<8; ++i) (*rr++) += (*cs++);
  /* #140: output[0][0] = @0 */
  casadi_copy(w0, 8, res[0]);
  return 0;
}

CASADI_SYMBOL_EXPORT int Drone_ode_expl_ode_fun(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int Drone_ode_expl_ode_fun_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int Drone_ode_expl_ode_fun_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void Drone_ode_expl_ode_fun_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int Drone_ode_expl_ode_fun_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void Drone_ode_expl_ode_fun_release(int mem) {
}

CASADI_SYMBOL_EXPORT void Drone_ode_expl_ode_fun_incref(void) {
}

CASADI_SYMBOL_EXPORT void Drone_ode_expl_ode_fun_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int Drone_ode_expl_ode_fun_n_in(void) { return 3;}

CASADI_SYMBOL_EXPORT casadi_int Drone_ode_expl_ode_fun_n_out(void) { return 1;}

CASADI_SYMBOL_EXPORT casadi_real Drone_ode_expl_ode_fun_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* Drone_ode_expl_ode_fun_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* Drone_ode_expl_ode_fun_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* Drone_ode_expl_ode_fun_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s4;
    case 1: return casadi_s5;
    case 2: return casadi_s6;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* Drone_ode_expl_ode_fun_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s4;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int Drone_ode_expl_ode_fun_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 11;
  if (sz_res) *sz_res = 2;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 340;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif

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
  #define CASADI_PREFIX(ID) Drone_ode_expl_vde_adj_ ## ID
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

/* Drone_ode_expl_vde_adj:(i0[8],i1[8],i2[4],i3[8])->(o0[12]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_int i, j, k;
  casadi_real *rr, *ss, *tt;
  const casadi_real *cs;
  casadi_real *w0=w+8, *w1=w+16, *w2=w+32, w3, w4, *w5=w+50, *w6=w+82, *w7=w+114, *w8=w+130, *w9=w+146, w10, w11, w12, *w13=w+165, *w14=w+197, *w15=w+261, w16, w17, w18, w19, w20, w21, w22, w23, w24, w25, w26, *w27=w+280, *w28=w+344;
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
  /* #81: @11 = (@4*@10) */
  w11  = (w4*w10);
  /* #82: (@9[4] = @11) */
  for (rr=w9+4, ss=(&w11); rr!=w9+5; rr+=1) *rr = *ss++;
  /* #83: @11 = 0 */
  w11 = 0.;
  /* #84: (@9[8] = @11) */
  for (rr=w9+8, ss=(&w11); rr!=w9+9; rr+=1) *rr = *ss++;
  /* #85: @11 = 0 */
  w11 = 0.;
  /* #86: (@9[12] = @11) */
  for (rr=w9+12, ss=(&w11); rr!=w9+13; rr+=1) *rr = *ss++;
  /* #87: @11 = 0.3984 */
  w11 = 3.9839999999999998e-01;
  /* #88: @12 = (@11*@10) */
  w12  = (w11*w10);
  /* #89: (@9[1] = @12) */
  for (rr=w9+1, ss=(&w12); rr!=w9+2; rr+=1) *rr = *ss++;
  /* #90: @12 = 0.704 */
  w12 = 7.0399999999999996e-01;
  /* #91: (@9[5] = @12) */
  for (rr=w9+5, ss=(&w12); rr!=w9+6; rr+=1) *rr = *ss++;
  /* #92: @12 = 0 */
  w12 = 0.;
  /* #93: (@9[9] = @12) */
  for (rr=w9+9, ss=(&w12); rr!=w9+10; rr+=1) *rr = *ss++;
  /* #94: @12 = 0 */
  w12 = 0.;
  /* #95: (@9[13] = @12) */
  for (rr=w9+13, ss=(&w12); rr!=w9+14; rr+=1) *rr = *ss++;
  /* #96: @12 = 0 */
  w12 = 0.;
  /* #97: (@9[2] = @12) */
  for (rr=w9+2, ss=(&w12); rr!=w9+3; rr+=1) *rr = *ss++;
  /* #98: @12 = 0 */
  w12 = 0.;
  /* #99: (@9[6] = @12) */
  for (rr=w9+6, ss=(&w12); rr!=w9+7; rr+=1) *rr = *ss++;
  /* #100: @12 = 0.9365 */
  w12 = 9.3650000000000000e-01;
  /* #101: (@9[10] = @12) */
  for (rr=w9+10, ss=(&w12); rr!=w9+11; rr+=1) *rr = *ss++;
  /* #102: @12 = 0 */
  w12 = 0.;
  /* #103: (@9[14] = @12) */
  for (rr=w9+14, ss=(&w12); rr!=w9+15; rr+=1) *rr = *ss++;
  /* #104: @12 = 0 */
  w12 = 0.;
  /* #105: (@9[3] = @12) */
  for (rr=w9+3, ss=(&w12); rr!=w9+4; rr+=1) *rr = *ss++;
  /* #106: @12 = 0 */
  w12 = 0.;
  /* #107: (@9[7] = @12) */
  for (rr=w9+7, ss=(&w12); rr!=w9+8; rr+=1) *rr = *ss++;
  /* #108: @12 = 0 */
  w12 = 0.;
  /* #109: (@9[11] = @12) */
  for (rr=w9+11, ss=(&w12); rr!=w9+12; rr+=1) *rr = *ss++;
  /* #110: @12 = 0.9752 */
  w12 = 9.7519999999999996e-01;
  /* #111: (@9[15] = @12) */
  for (rr=w9+15, ss=(&w12); rr!=w9+16; rr+=1) *rr = *ss++;
  /* #112: @2 = mac(@7,@9,@2) */
  for (i=0, rr=w2; i<4; ++i) for (j=0; j<4; ++j, ++rr) for (k=0, ss=w7+j, tt=w9+i*4; k<4; ++k) *rr += ss[k*4]**tt++;
  /* #113: @2 = (-@2) */
  for (i=0, rr=w2, cs=w2; i<16; ++i) *rr++ = (- *cs++ );
  /* #114: @5 = horzcat(@1, @2) */
  rr=w5;
  for (i=0, cs=w1; i<16; ++i) *rr++ = *cs++;
  for (i=0, cs=w2; i<16; ++i) *rr++ = *cs++;
  /* #115: @13 = @5' */
  for (i=0, rr=w13, cs=w5; i<8; ++i) for (j=0; j<4; ++j) rr[i+j*8] = *cs++;
  /* #116: @14 = horzcat(@6, @13) */
  rr=w14;
  for (i=0, cs=w6; i<32; ++i) *rr++ = *cs++;
  for (i=0, cs=w13; i<32; ++i) *rr++ = *cs++;
  /* #117: @15 = input[1][0] */
  casadi_copy(arg[1], 8, w15);
  /* #118: @0 = mac(@14,@15,@0) */
  for (i=0, rr=w0; i<1; ++i) for (j=0; j<8; ++j, ++rr) for (k=0, ss=w14+j, tt=w15+i*8; k<8; ++k) *rr += ss[k*8]**tt++;
  /* #119: {@12, @16, @17, @18, @19, @20, @21, @22} = vertsplit(@0) */
  w12 = w0[0];
  w16 = w0[1];
  w17 = w0[2];
  w18 = w0[3];
  w19 = w0[4];
  w20 = w0[5];
  w21 = w0[6];
  w22 = w0[7];
  /* #120: output[0][0] = @12 */
  if (res[0]) res[0][0] = w12;
  /* #121: output[0][1] = @16 */
  if (res[0]) res[0][1] = w16;
  /* #122: output[0][2] = @17 */
  if (res[0]) res[0][2] = w17;
  /* #123: @17 = sin(@3) */
  w17 = sin( w3 );
  /* #124: @14 = zeros(8x8) */
  casadi_clear(w14, 64);
  /* #125: @16 = input[0][0] */
  w16 = arg[0] ? arg[0][0] : 0;
  /* #126: @12 = input[0][1] */
  w12 = arg[0] ? arg[0][1] : 0;
  /* #127: @23 = input[0][2] */
  w23 = arg[0] ? arg[0][2] : 0;
  /* #128: @24 = input[0][4] */
  w24 = arg[0] ? arg[0][4] : 0;
  /* #129: @25 = input[0][5] */
  w25 = arg[0] ? arg[0][5] : 0;
  /* #130: @26 = input[0][6] */
  w26 = arg[0] ? arg[0][6] : 0;
  /* #131: @0 = vertcat(@16, @12, @23, @3, @24, @25, @26, @10) */
  rr=w0;
  *rr++ = w16;
  *rr++ = w12;
  *rr++ = w23;
  *rr++ = w3;
  *rr++ = w24;
  *rr++ = w25;
  *rr++ = w26;
  *rr++ = w10;
  /* #132: @0 = @0' */
  /* #133: @14 = mac(@15,@0,@14) */
  for (i=0, rr=w14; i<8; ++i) for (j=0; j<8; ++j, ++rr) for (k=0, ss=w15+j, tt=w0+i*1; k<1; ++k) *rr += ss[k*8]**tt++;
  /* #134: @27 = @14' */
  for (i=0, rr=w27, cs=w14; i<8; ++i) for (j=0; j<8; ++j) rr[i+j*8] = *cs++;
  /* #135: {@6, @13} = horzsplit(@27) */
  casadi_copy(w27, 32, w6);
  casadi_copy(w27+32, 32, w13);
  /* #136: @5 = @6' */
  for (i=0, rr=w5, cs=w6; i<4; ++i) for (j=0; j<8; ++j) rr[i+j*4] = *cs++;
  /* #137: {NULL, @1} = horzsplit(@5) */
  casadi_copy(w5+16, 16, w1);
  /* #138: @16 = 0 */
  w16 = 0.;
  /* #139: (@1[15] = @16) */
  for (rr=w1+15, ss=(&w16); rr!=w1+16; rr+=1) *rr = *ss++;
  /* #140: @16 = 0 */
  w16 = 0.;
  /* #141: (@1[11] = @16) */
  for (rr=w1+11, ss=(&w16); rr!=w1+12; rr+=1) *rr = *ss++;
  /* #142: @16 = 0 */
  w16 = 0.;
  /* #143: (@1[7] = @16) */
  for (rr=w1+7, ss=(&w16); rr!=w1+8; rr+=1) *rr = *ss++;
  /* #144: @16 = 0 */
  w16 = 0.;
  /* #145: (@1[3] = @16) */
  for (rr=w1+3, ss=(&w16); rr!=w1+4; rr+=1) *rr = *ss++;
  /* #146: @16 = 0 */
  w16 = 0.;
  /* #147: (@1[14] = @16) */
  for (rr=w1+14, ss=(&w16); rr!=w1+15; rr+=1) *rr = *ss++;
  /* #148: @16 = 0 */
  w16 = 0.;
  /* #149: (@1[10] = @16) */
  for (rr=w1+10, ss=(&w16); rr!=w1+11; rr+=1) *rr = *ss++;
  /* #150: @16 = 0 */
  w16 = 0.;
  /* #151: (@1[6] = @16) */
  for (rr=w1+6, ss=(&w16); rr!=w1+7; rr+=1) *rr = *ss++;
  /* #152: @16 = 0 */
  w16 = 0.;
  /* #153: (@1[2] = @16) */
  for (rr=w1+2, ss=(&w16); rr!=w1+3; rr+=1) *rr = *ss++;
  /* #154: @16 = 0 */
  w16 = 0.;
  /* #155: (@1[13] = @16) */
  for (rr=w1+13, ss=(&w16); rr!=w1+14; rr+=1) *rr = *ss++;
  /* #156: @16 = 0 */
  w16 = 0.;
  /* #157: (@1[9] = @16) */
  for (rr=w1+9, ss=(&w16); rr!=w1+10; rr+=1) *rr = *ss++;
  /* #158: @16 = @1[5] */
  for (rr=(&w16), ss=w1+5; ss!=w1+6; ss+=1) *rr++ = *ss;
  /* #159: @17 = (@17*@16) */
  w17 *= w16;
  /* #160: @18 = (@18-@17) */
  w18 -= w17;
  /* #161: @17 = cos(@3) */
  w17 = cos( w3 );
  /* #162: @16 = 0 */
  w16 = 0.;
  /* #163: (@1[5] = @16) */
  for (rr=w1+5, ss=(&w16); rr!=w1+6; rr+=1) *rr = *ss++;
  /* #164: @16 = @1[1] */
  for (rr=(&w16), ss=w1+1; ss!=w1+2; ss+=1) *rr++ = *ss;
  /* #165: @17 = (@17*@16) */
  w17 *= w16;
  /* #166: @18 = (@18+@17) */
  w18 += w17;
  /* #167: @17 = cos(@3) */
  w17 = cos( w3 );
  /* #168: @16 = 0 */
  w16 = 0.;
  /* #169: (@1[1] = @16) */
  for (rr=w1+1, ss=(&w16); rr!=w1+2; rr+=1) *rr = *ss++;
  /* #170: @16 = 0 */
  w16 = 0.;
  /* #171: (@1[12] = @16) */
  for (rr=w1+12, ss=(&w16); rr!=w1+13; rr+=1) *rr = *ss++;
  /* #172: @16 = 0 */
  w16 = 0.;
  /* #173: (@1[8] = @16) */
  for (rr=w1+8, ss=(&w16); rr!=w1+9; rr+=1) *rr = *ss++;
  /* #174: @16 = @1[4] */
  for (rr=(&w16), ss=w1+4; ss!=w1+5; ss+=1) *rr++ = *ss;
  /* #175: @17 = (@17*@16) */
  w17 *= w16;
  /* #176: @18 = (@18-@17) */
  w18 -= w17;
  /* #177: @3 = sin(@3) */
  w3 = sin( w3 );
  /* #178: @17 = 0 */
  w17 = 0.;
  /* #179: (@1[4] = @17) */
  for (rr=w1+4, ss=(&w17); rr!=w1+5; rr+=1) *rr = *ss++;
  /* #180: @17 = @1[0] */
  for (rr=(&w17), ss=w1+0; ss!=w1+1; ss+=1) *rr++ = *ss;
  /* #181: @3 = (@3*@17) */
  w3 *= w17;
  /* #182: @18 = (@18-@3) */
  w18 -= w3;
  /* #183: output[0][3] = @18 */
  if (res[0]) res[0][3] = w18;
  /* #184: output[0][4] = @19 */
  if (res[0]) res[0][4] = w19;
  /* #185: output[0][5] = @20 */
  if (res[0]) res[0][5] = w20;
  /* #186: output[0][6] = @21 */
  if (res[0]) res[0][6] = w21;
  /* #187: @1 = zeros(4x4) */
  casadi_clear(w1, 16);
  /* #188: @2 = @7' */
  for (i=0, rr=w2, cs=w7; i<4; ++i) for (j=0; j<4; ++j) rr[i+j*4] = *cs++;
  /* #189: @5 = @13' */
  for (i=0, rr=w5, cs=w13; i<4; ++i) for (j=0; j<8; ++j) rr[i+j*4] = *cs++;
  /* #190: {NULL, @7} = horzsplit(@5) */
  casadi_copy(w5+16, 16, w7);
  /* #191: @7 = (-@7) */
  for (i=0, rr=w7, cs=w7; i<16; ++i) *rr++ = (- *cs++ );
  /* #192: @1 = mac(@2,@7,@1) */
  for (i=0, rr=w1; i<4; ++i) for (j=0; j<4; ++j, ++rr) for (k=0, ss=w2+j, tt=w7+i*4; k<4; ++k) *rr += ss[k*4]**tt++;
  /* #193: @21 = 0 */
  w21 = 0.;
  /* #194: (@1[15] = @21) */
  for (rr=w1+15, ss=(&w21); rr!=w1+16; rr+=1) *rr = *ss++;
  /* #195: @21 = 0 */
  w21 = 0.;
  /* #196: (@1[11] = @21) */
  for (rr=w1+11, ss=(&w21); rr!=w1+12; rr+=1) *rr = *ss++;
  /* #197: @21 = 0 */
  w21 = 0.;
  /* #198: (@1[7] = @21) */
  for (rr=w1+7, ss=(&w21); rr!=w1+8; rr+=1) *rr = *ss++;
  /* #199: @21 = 0 */
  w21 = 0.;
  /* #200: (@1[3] = @21) */
  for (rr=w1+3, ss=(&w21); rr!=w1+4; rr+=1) *rr = *ss++;
  /* #201: @21 = 0 */
  w21 = 0.;
  /* #202: (@1[14] = @21) */
  for (rr=w1+14, ss=(&w21); rr!=w1+15; rr+=1) *rr = *ss++;
  /* #203: @21 = 0 */
  w21 = 0.;
  /* #204: (@1[10] = @21) */
  for (rr=w1+10, ss=(&w21); rr!=w1+11; rr+=1) *rr = *ss++;
  /* #205: @21 = 0 */
  w21 = 0.;
  /* #206: (@1[6] = @21) */
  for (rr=w1+6, ss=(&w21); rr!=w1+7; rr+=1) *rr = *ss++;
  /* #207: @21 = 0 */
  w21 = 0.;
  /* #208: (@1[2] = @21) */
  for (rr=w1+2, ss=(&w21); rr!=w1+3; rr+=1) *rr = *ss++;
  /* #209: @21 = 0 */
  w21 = 0.;
  /* #210: (@1[13] = @21) */
  for (rr=w1+13, ss=(&w21); rr!=w1+14; rr+=1) *rr = *ss++;
  /* #211: @21 = 0 */
  w21 = 0.;
  /* #212: (@1[9] = @21) */
  for (rr=w1+9, ss=(&w21); rr!=w1+10; rr+=1) *rr = *ss++;
  /* #213: @21 = 0 */
  w21 = 0.;
  /* #214: (@1[5] = @21) */
  for (rr=w1+5, ss=(&w21); rr!=w1+6; rr+=1) *rr = *ss++;
  /* #215: @21 = @1[1] */
  for (rr=(&w21), ss=w1+1; ss!=w1+2; ss+=1) *rr++ = *ss;
  /* #216: @11 = (@11*@21) */
  w11 *= w21;
  /* #217: @22 = (@22+@11) */
  w22 += w11;
  /* #218: @11 = 0 */
  w11 = 0.;
  /* #219: (@1[1] = @11) */
  for (rr=w1+1, ss=(&w11); rr!=w1+2; rr+=1) *rr = *ss++;
  /* #220: @11 = 0 */
  w11 = 0.;
  /* #221: (@1[12] = @11) */
  for (rr=w1+12, ss=(&w11); rr!=w1+13; rr+=1) *rr = *ss++;
  /* #222: @11 = 0 */
  w11 = 0.;
  /* #223: (@1[8] = @11) */
  for (rr=w1+8, ss=(&w11); rr!=w1+9; rr+=1) *rr = *ss++;
  /* #224: @11 = @1[4] */
  for (rr=(&w11), ss=w1+4; ss!=w1+5; ss+=1) *rr++ = *ss;
  /* #225: @4 = (@4*@11) */
  w4 *= w11;
  /* #226: @22 = (@22+@4) */
  w22 += w4;
  /* #227: output[0][7] = @22 */
  if (res[0]) res[0][7] = w22;
  /* #228: @28 = zeros(4x1) */
  casadi_clear(w28, 4);
  /* #229: @1 = zeros(4x4) */
  casadi_clear(w1, 16);
  /* #230: @2 = 
  [[1, 0, 0, 0], 
   [0, 1, 0, 0], 
   [0, 0, 1, 0], 
   [0, 0, 0, 1]] */
  casadi_copy(casadi_c0, 16, w2);
  /* #231: @2 = (@8\@2) */
  rr = w2;
  ss = w8;
  {
    /* FIXME(@jaeandersson): Memory allocation can be avoided */
    casadi_real v[10], r[10], beta[4], w[8];
    casadi_qr(casadi_s1, ss, w, casadi_s2, v, casadi_s3, r, beta, casadi_s0, casadi_s0);
    casadi_qr_solve(rr, 4, 0, casadi_s2, v, casadi_s3, r, beta, casadi_s0, casadi_s0, w);
  }
  /* #232: @8 = @2' */
  for (i=0, rr=w8, cs=w2; i<4; ++i) for (j=0; j<4; ++j) rr[i+j*4] = *cs++;
  /* #233: @5 = horzcat(@1, @8) */
  rr=w5;
  for (i=0, cs=w1; i<16; ++i) *rr++ = *cs++;
  for (i=0, cs=w8; i<16; ++i) *rr++ = *cs++;
  /* #234: @28 = mac(@5,@15,@28) */
  for (i=0, rr=w28; i<1; ++i) for (j=0; j<4; ++j, ++rr) for (k=0, ss=w5+j, tt=w15+i*8; k<8; ++k) *rr += ss[k*4]**tt++;
  /* #235: {@22, @4, @11, @21} = vertsplit(@28) */
  w22 = w28[0];
  w4 = w28[1];
  w11 = w28[2];
  w21 = w28[3];
  /* #236: output[0][8] = @22 */
  if (res[0]) res[0][8] = w22;
  /* #237: output[0][9] = @4 */
  if (res[0]) res[0][9] = w4;
  /* #238: output[0][10] = @11 */
  if (res[0]) res[0][10] = w11;
  /* #239: output[0][11] = @21 */
  if (res[0]) res[0][11] = w21;
  return 0;
}

CASADI_SYMBOL_EXPORT int Drone_ode_expl_vde_adj(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int Drone_ode_expl_vde_adj_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int Drone_ode_expl_vde_adj_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void Drone_ode_expl_vde_adj_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int Drone_ode_expl_vde_adj_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void Drone_ode_expl_vde_adj_release(int mem) {
}

CASADI_SYMBOL_EXPORT void Drone_ode_expl_vde_adj_incref(void) {
}

CASADI_SYMBOL_EXPORT void Drone_ode_expl_vde_adj_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int Drone_ode_expl_vde_adj_n_in(void) { return 4;}

CASADI_SYMBOL_EXPORT casadi_int Drone_ode_expl_vde_adj_n_out(void) { return 1;}

CASADI_SYMBOL_EXPORT casadi_real Drone_ode_expl_vde_adj_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* Drone_ode_expl_vde_adj_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* Drone_ode_expl_vde_adj_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* Drone_ode_expl_vde_adj_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s4;
    case 1: return casadi_s4;
    case 2: return casadi_s5;
    case 3: return casadi_s4;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* Drone_ode_expl_vde_adj_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s6;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int Drone_ode_expl_vde_adj_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 12;
  if (sz_res) *sz_res = 9;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 348;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif

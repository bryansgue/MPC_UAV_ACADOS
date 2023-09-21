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
  #define CASADI_PREFIX(ID) Drone_ode_cost_ext_cost_fun_jac_hess_ ## ID
#endif

#include <math.h>

#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int int
#endif

/* Add prefix to internal symbols */
#define casadi_clear CASADI_PREFIX(clear)
#define casadi_copy CASADI_PREFIX(copy)
#define casadi_f0 CASADI_PREFIX(f0)
#define casadi_fill CASADI_PREFIX(fill)
#define casadi_mtimes CASADI_PREFIX(mtimes)
#define casadi_s0 CASADI_PREFIX(s0)
#define casadi_s1 CASADI_PREFIX(s1)
#define casadi_s10 CASADI_PREFIX(s10)
#define casadi_s11 CASADI_PREFIX(s11)
#define casadi_s12 CASADI_PREFIX(s12)
#define casadi_s2 CASADI_PREFIX(s2)
#define casadi_s3 CASADI_PREFIX(s3)
#define casadi_s4 CASADI_PREFIX(s4)
#define casadi_s5 CASADI_PREFIX(s5)
#define casadi_s6 CASADI_PREFIX(s6)
#define casadi_s7 CASADI_PREFIX(s7)
#define casadi_s8 CASADI_PREFIX(s8)
#define casadi_s9 CASADI_PREFIX(s9)
#define casadi_trans CASADI_PREFIX(trans)

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

void casadi_fill(casadi_real* x, casadi_int n, casadi_real alpha) {
  casadi_int i;
  if (x) {
    for (i=0; i<n; ++i) *x++ = alpha;
  }
}

void casadi_mtimes(const casadi_real* x, const casadi_int* sp_x, const casadi_real* y, const casadi_int* sp_y, casadi_real* z, const casadi_int* sp_z, casadi_real* w, casadi_int tr) {
  casadi_int ncol_x, ncol_y, ncol_z, cc;
  const casadi_int *colind_x, *row_x, *colind_y, *row_y, *colind_z, *row_z;
  ncol_x = sp_x[1];
  colind_x = sp_x+2; row_x = sp_x + 2 + ncol_x+1;
  ncol_y = sp_y[1];
  colind_y = sp_y+2; row_y = sp_y + 2 + ncol_y+1;
  ncol_z = sp_z[1];
  colind_z = sp_z+2; row_z = sp_z + 2 + ncol_z+1;
  if (tr) {
    for (cc=0; cc<ncol_z; ++cc) {
      casadi_int kk;
      for (kk=colind_y[cc]; kk<colind_y[cc+1]; ++kk) {
        w[row_y[kk]] = y[kk];
      }
      for (kk=colind_z[cc]; kk<colind_z[cc+1]; ++kk) {
        casadi_int kk1;
        casadi_int rr = row_z[kk];
        for (kk1=colind_x[rr]; kk1<colind_x[rr+1]; ++kk1) {
          z[kk] += x[kk1] * w[row_x[kk1]];
        }
      }
    }
  } else {
    for (cc=0; cc<ncol_y; ++cc) {
      casadi_int kk;
      for (kk=colind_z[cc]; kk<colind_z[cc+1]; ++kk) {
        w[row_z[kk]] = z[kk];
      }
      for (kk=colind_y[cc]; kk<colind_y[cc+1]; ++kk) {
        casadi_int kk1;
        casadi_int rr = row_y[kk];
        for (kk1=colind_x[rr]; kk1<colind_x[rr+1]; ++kk1) {
          w[row_x[kk1]] += x[kk1]*y[kk];
        }
      }
      for (kk=colind_z[cc]; kk<colind_z[cc+1]; ++kk) {
        z[kk] = w[row_z[kk]];
      }
    }
  }
}

void casadi_trans(const casadi_real* x, const casadi_int* sp_x, casadi_real* y,
    const casadi_int* sp_y, casadi_int* tmp) {
  casadi_int ncol_x, nnz_x, ncol_y, k;
  const casadi_int* row_x, *colind_y;
  ncol_x = sp_x[1];
  nnz_x = sp_x[2 + ncol_x];
  row_x = sp_x + 2 + ncol_x+1;
  ncol_y = sp_y[1];
  colind_y = sp_y+2;
  for (k=0; k<ncol_y; ++k) tmp[k] = colind_y[k];
  for (k=0; k<nnz_x; ++k) {
    y[tmp[row_x[k]]++] = x[k];
  }
}

static const casadi_int casadi_s0[11] = {1, 4, 0, 1, 2, 3, 4, 0, 0, 0, 0};
static const casadi_int casadi_s1[23] = {4, 4, 0, 4, 8, 12, 16, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
static const casadi_int casadi_s2[8] = {1, 4, 0, 1, 1, 1, 1, 0};
static const casadi_int casadi_s3[8] = {1, 4, 0, 0, 1, 1, 1, 0};
static const casadi_int casadi_s4[8] = {1, 4, 0, 0, 0, 1, 1, 0};
static const casadi_int casadi_s5[8] = {1, 4, 0, 0, 0, 0, 1, 0};
static const casadi_int casadi_s6[47] = {12, 12, 0, 4, 8, 12, 16, 20, 24, 28, 32, 32, 32, 32, 32, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7};
static const casadi_int casadi_s7[12] = {8, 1, 0, 8, 0, 1, 2, 3, 4, 5, 6, 7};
static const casadi_int casadi_s8[8] = {4, 1, 0, 4, 0, 1, 2, 3};
static const casadi_int casadi_s9[3] = {0, 0, 0};
static const casadi_int casadi_s10[5] = {1, 1, 0, 1, 0};
static const casadi_int casadi_s11[16] = {12, 1, 0, 12, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
static const casadi_int casadi_s12[15] = {0, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

/* Drone_ode_cost_ext_cost_fun_jac_hess:(i0[8],i1[4],i2[],i3[8])->(o0,o1[12],o2[12x12,32nz],o3[],o4[0x12]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_int i, j, k;
  casadi_real *rr, *ss, *tt;
  const casadi_real *cs;
  casadi_real w0, *w1=w+2, w2, w3, w4, w5, w6, w7, w8, w9, *w10=w+14, *w11=w+22, *w12=w+26, *w13=w+30, *w14=w+46, *w15=w+50, *w16=w+54, *w17=w+70, *w18=w+86, *w19=w+102, *w20=w+134, w24, w25, w26, *w27=w+143, *w28=w+155, *w29=w+157;
  /* #0: @0 = 0 */
  w0 = 0.;
  /* #1: @1 = zeros(1x4) */
  casadi_clear(w1, 4);
  /* #2: @2 = input[3][0] */
  w2 = arg[3] ? arg[3][0] : 0;
  /* #3: @3 = input[3][1] */
  w3 = arg[3] ? arg[3][1] : 0;
  /* #4: @4 = input[3][2] */
  w4 = arg[3] ? arg[3][2] : 0;
  /* #5: @5 = input[3][3] */
  w5 = arg[3] ? arg[3][3] : 0;
  /* #6: @6 = input[3][4] */
  w6 = arg[3] ? arg[3][4] : 0;
  /* #7: @7 = input[3][5] */
  w7 = arg[3] ? arg[3][5] : 0;
  /* #8: @8 = input[3][6] */
  w8 = arg[3] ? arg[3][6] : 0;
  /* #9: @9 = input[3][7] */
  w9 = arg[3] ? arg[3][7] : 0;
  /* #10: @10 = vertcat(@2, @3, @4, @5, @6, @7, @8, @9) */
  rr=w10;
  *rr++ = w2;
  *rr++ = w3;
  *rr++ = w4;
  *rr++ = w5;
  *rr++ = w6;
  *rr++ = w7;
  *rr++ = w8;
  *rr++ = w9;
  /* #11: @11 = @10[:4] */
  for (rr=w11, ss=w10+0; ss!=w10+4; ss+=1) *rr++ = *ss;
  /* #12: @2 = input[0][0] */
  w2 = arg[0] ? arg[0][0] : 0;
  /* #13: @3 = input[0][1] */
  w3 = arg[0] ? arg[0][1] : 0;
  /* #14: @4 = input[0][2] */
  w4 = arg[0] ? arg[0][2] : 0;
  /* #15: @5 = input[0][3] */
  w5 = arg[0] ? arg[0][3] : 0;
  /* #16: @6 = input[0][4] */
  w6 = arg[0] ? arg[0][4] : 0;
  /* #17: @7 = input[0][5] */
  w7 = arg[0] ? arg[0][5] : 0;
  /* #18: @8 = input[0][6] */
  w8 = arg[0] ? arg[0][6] : 0;
  /* #19: @9 = input[0][7] */
  w9 = arg[0] ? arg[0][7] : 0;
  /* #20: @10 = vertcat(@2, @3, @4, @5, @6, @7, @8, @9) */
  rr=w10;
  *rr++ = w2;
  *rr++ = w3;
  *rr++ = w4;
  *rr++ = w5;
  *rr++ = w6;
  *rr++ = w7;
  *rr++ = w8;
  *rr++ = w9;
  /* #21: @12 = @10[:4] */
  for (rr=w12, ss=w10+0; ss!=w10+4; ss+=1) *rr++ = *ss;
  /* #22: @11 = (@11-@12) */
  for (i=0, rr=w11, cs=w12; i<4; ++i) (*rr++) -= (*cs++);
  /* #23: @12 = @11' */
  casadi_copy(w11, 4, w12);
  /* #24: @13 = zeros(4x4) */
  casadi_clear(w13, 16);
  /* #25: @2 = 1.1 */
  w2 = 1.1000000000000001e+00;
  /* #26: (@13[0] = @2) */
  for (rr=w13+0, ss=(&w2); rr!=w13+1; rr+=1) *rr = *ss++;
  /* #27: @2 = 1.1 */
  w2 = 1.1000000000000001e+00;
  /* #28: (@13[5] = @2) */
  for (rr=w13+5, ss=(&w2); rr!=w13+6; rr+=1) *rr = *ss++;
  /* #29: @2 = 1.1 */
  w2 = 1.1000000000000001e+00;
  /* #30: (@13[10] = @2) */
  for (rr=w13+10, ss=(&w2); rr!=w13+11; rr+=1) *rr = *ss++;
  /* #31: @2 = 1 */
  w2 = 1.;
  /* #32: (@13[15] = @2) */
  for (rr=w13+15, ss=(&w2); rr!=w13+16; rr+=1) *rr = *ss++;
  /* #33: @1 = mac(@12,@13,@1) */
  for (i=0, rr=w1; i<4; ++i) for (j=0; j<1; ++j, ++rr) for (k=0, ss=w12+j, tt=w13+i*4; k<4; ++k) *rr += ss[k*1]**tt++;
  /* #34: @0 = mac(@1,@11,@0) */
  for (i=0, rr=(&w0); i<1; ++i) for (j=0; j<1; ++j, ++rr) for (k=0, ss=w1+j, tt=w11+i*4; k<4; ++k) *rr += ss[k*1]**tt++;
  /* #35: @2 = 0 */
  w2 = 0.;
  /* #36: @12 = zeros(1x4) */
  casadi_clear(w12, 4);
  /* #37: @3 = input[1][0] */
  w3 = arg[1] ? arg[1][0] : 0;
  /* #38: @4 = input[1][1] */
  w4 = arg[1] ? arg[1][1] : 0;
  /* #39: @5 = input[1][2] */
  w5 = arg[1] ? arg[1][2] : 0;
  /* #40: @6 = input[1][3] */
  w6 = arg[1] ? arg[1][3] : 0;
  /* #41: @14 = vertcat(@3, @4, @5, @6) */
  rr=w14;
  *rr++ = w3;
  *rr++ = w4;
  *rr++ = w5;
  *rr++ = w6;
  /* #42: @15 = @14' */
  casadi_copy(w14, 4, w15);
  /* #43: @16 = zeros(4x4) */
  casadi_clear(w16, 16);
  /* #44: @3 = 0.65 */
  w3 = 6.5000000000000002e-01;
  /* #45: (@16[0] = @3) */
  for (rr=w16+0, ss=(&w3); rr!=w16+1; rr+=1) *rr = *ss++;
  /* #46: @3 = 0.65 */
  w3 = 6.5000000000000002e-01;
  /* #47: (@16[5] = @3) */
  for (rr=w16+5, ss=(&w3); rr!=w16+6; rr+=1) *rr = *ss++;
  /* #48: @3 = 0.65 */
  w3 = 6.5000000000000002e-01;
  /* #49: (@16[10] = @3) */
  for (rr=w16+10, ss=(&w3); rr!=w16+11; rr+=1) *rr = *ss++;
  /* #50: @3 = 0.65 */
  w3 = 6.5000000000000002e-01;
  /* #51: (@16[15] = @3) */
  for (rr=w16+15, ss=(&w3); rr!=w16+16; rr+=1) *rr = *ss++;
  /* #52: @12 = mac(@15,@16,@12) */
  for (i=0, rr=w12; i<4; ++i) for (j=0; j<1; ++j, ++rr) for (k=0, ss=w15+j, tt=w16+i*4; k<4; ++k) *rr += ss[k*1]**tt++;
  /* #53: @2 = mac(@12,@14,@2) */
  for (i=0, rr=(&w2); i<1; ++i) for (j=0; j<1; ++j, ++rr) for (k=0, ss=w12+j, tt=w14+i*4; k<4; ++k) *rr += ss[k*1]**tt++;
  /* #54: @0 = (@0+@2) */
  w0 += w2;
  /* #55: output[0][0] = @0 */
  if (res[0]) res[0][0] = w0;
  /* #56: @12 = @12' */
  /* #57: @15 = zeros(1x4) */
  casadi_clear(w15, 4);
  /* #58: @14 = @14' */
  /* #59: @17 = @16' */
  for (i=0, rr=w17, cs=w16; i<4; ++i) for (j=0; j<4; ++j) rr[i+j*4] = *cs++;
  /* #60: @15 = mac(@14,@17,@15) */
  for (i=0, rr=w15; i<4; ++i) for (j=0; j<1; ++j, ++rr) for (k=0, ss=w14+j, tt=w17+i*4; k<4; ++k) *rr += ss[k*1]**tt++;
  /* #61: @15 = @15' */
  /* #62: @12 = (@12+@15) */
  for (i=0, rr=w12, cs=w15; i<4; ++i) (*rr++) += (*cs++);
  /* #63: {@0, @2, @3, @4} = vertsplit(@12) */
  w0 = w12[0];
  w2 = w12[1];
  w3 = w12[2];
  w4 = w12[3];
  /* #64: output[1][0] = @0 */
  if (res[1]) res[1][0] = w0;
  /* #65: output[1][1] = @2 */
  if (res[1]) res[1][1] = w2;
  /* #66: output[1][2] = @3 */
  if (res[1]) res[1][2] = w3;
  /* #67: output[1][3] = @4 */
  if (res[1]) res[1][3] = w4;
  /* #68: @10 = zeros(8x1) */
  casadi_clear(w10, 8);
  /* #69: @1 = @1' */
  /* #70: @12 = zeros(1x4) */
  casadi_clear(w12, 4);
  /* #71: @11 = @11' */
  /* #72: @18 = @13' */
  for (i=0, rr=w18, cs=w13; i<4; ++i) for (j=0; j<4; ++j) rr[i+j*4] = *cs++;
  /* #73: @12 = mac(@11,@18,@12) */
  for (i=0, rr=w12; i<4; ++i) for (j=0; j<1; ++j, ++rr) for (k=0, ss=w11+j, tt=w18+i*4; k<4; ++k) *rr += ss[k*1]**tt++;
  /* #74: @12 = @12' */
  /* #75: @1 = (@1+@12) */
  for (i=0, rr=w1, cs=w12; i<4; ++i) (*rr++) += (*cs++);
  /* #76: @1 = (-@1) */
  for (i=0, rr=w1, cs=w1; i<4; ++i) *rr++ = (- *cs++ );
  /* #77: (@10[:4] += @1) */
  for (rr=w10+0, ss=w1; rr!=w10+4; rr+=1) *rr += *ss++;
  /* #78: {@4, @3, @2, @0, @5, @6, @7, @8} = vertsplit(@10) */
  w4 = w10[0];
  w3 = w10[1];
  w2 = w10[2];
  w0 = w10[3];
  w5 = w10[4];
  w6 = w10[5];
  w7 = w10[6];
  w8 = w10[7];
  /* #79: output[1][4] = @4 */
  if (res[1]) res[1][4] = w4;
  /* #80: output[1][5] = @3 */
  if (res[1]) res[1][5] = w3;
  /* #81: output[1][6] = @2 */
  if (res[1]) res[1][6] = w2;
  /* #82: output[1][7] = @0 */
  if (res[1]) res[1][7] = w0;
  /* #83: output[1][8] = @5 */
  if (res[1]) res[1][8] = w5;
  /* #84: output[1][9] = @6 */
  if (res[1]) res[1][9] = w6;
  /* #85: output[1][10] = @7 */
  if (res[1]) res[1][10] = w7;
  /* #86: output[1][11] = @8 */
  if (res[1]) res[1][11] = w8;
  /* #87: @19 = zeros(12x12,32nz) */
  casadi_clear(w19, 32);
  /* #88: @1 = zeros(1x4) */
  casadi_clear(w1, 4);
  /* #89: @20 = ones(12x1,6nz) */
  casadi_fill(w20, 6, 1.);
  /* #90: {@8, NULL, NULL, NULL, @7, NULL, NULL, NULL, NULL, NULL, NULL, NULL} = vertsplit(@20) */
  w8 = w20[0];
  w7 = w20[1];
  /* #91: @21 = 00 */
  /* #92: @22 = 00 */
  /* #93: @23 = 00 */
  /* #94: @6 = vertcat(@8, @21, @22, @23) */
  rr=(&w6);
  *rr++ = w8;
  /* #95: @8 = @6' */
  casadi_copy((&w6), 1, (&w8));
  /* #96: @1 = mac(@8,@16,@1) */
  casadi_mtimes((&w8), casadi_s2, w16, casadi_s1, w1, casadi_s0, w, 0);
  /* #97: @1 = @1' */
  /* #98: @12 = zeros(1x4) */
  casadi_clear(w12, 4);
  /* #99: @6 = @6' */
  /* #100: @12 = mac(@6,@17,@12) */
  casadi_mtimes((&w6), casadi_s2, w17, casadi_s1, w12, casadi_s0, w, 0);
  /* #101: @12 = @12' */
  /* #102: @1 = (@1+@12) */
  for (i=0, rr=w1, cs=w12; i<4; ++i) (*rr++) += (*cs++);
  /* #103: {@6, @8, @5, @0} = vertsplit(@1) */
  w6 = w1[0];
  w8 = w1[1];
  w5 = w1[2];
  w0 = w1[3];
  /* #104: @10 = zeros(8x1) */
  casadi_clear(w10, 8);
  /* #105: @1 = zeros(1x4) */
  casadi_clear(w1, 4);
  /* #106: @2 = @7[0] */
  for (rr=(&w2), ss=(&w7)+0; ss!=(&w7)+1; ss+=1) *rr++ = *ss;
  /* #107: @2 = (-@2) */
  w2 = (- w2 );
  /* #108: @7 = @2' */
  casadi_copy((&w2), 1, (&w7));
  /* #109: @1 = mac(@7,@13,@1) */
  casadi_mtimes((&w7), casadi_s2, w13, casadi_s1, w1, casadi_s0, w, 0);
  /* #110: @1 = @1' */
  /* #111: @12 = zeros(1x4) */
  casadi_clear(w12, 4);
  /* #112: @2 = @2' */
  /* #113: @12 = mac(@2,@18,@12) */
  casadi_mtimes((&w2), casadi_s2, w18, casadi_s1, w12, casadi_s0, w, 0);
  /* #114: @12 = @12' */
  /* #115: @1 = (@1+@12) */
  for (i=0, rr=w1, cs=w12; i<4; ++i) (*rr++) += (*cs++);
  /* #116: @1 = (-@1) */
  for (i=0, rr=w1, cs=w1; i<4; ++i) *rr++ = (- *cs++ );
  /* #117: (@10[:4] += @1) */
  for (rr=w10+0, ss=w1; rr!=w10+4; rr+=1) *rr += *ss++;
  /* #118: {@2, @7, @3, @4, @9, @24, @25, @26} = vertsplit(@10) */
  w2 = w10[0];
  w7 = w10[1];
  w3 = w10[2];
  w4 = w10[3];
  w9 = w10[4];
  w24 = w10[5];
  w25 = w10[6];
  w26 = w10[7];
  /* #119: @27 = vertcat(@6, @8, @5, @0, @2, @7, @3, @4, @9, @24, @25, @26) */
  rr=w27;
  *rr++ = w6;
  *rr++ = w8;
  *rr++ = w5;
  *rr++ = w0;
  *rr++ = w2;
  *rr++ = w7;
  *rr++ = w3;
  *rr++ = w4;
  *rr++ = w9;
  *rr++ = w24;
  *rr++ = w25;
  *rr++ = w26;
  /* #120: @10 = @27[:8] */
  for (rr=w10, ss=w27+0; ss!=w27+8; ss+=1) *rr++ = *ss;
  /* #121: (@19[:32:16;:4] = @10) */
  for (rr=w19+0, ss=w10; rr!=w19+32; rr+=16) for (tt=rr+0; tt!=rr+4; tt+=1) *tt = *ss++;
  /* #122: @10 = @27[:8] */
  for (rr=w10, ss=w27+0; ss!=w27+8; ss+=1) *rr++ = *ss;
  /* #123: (@19[:32:4] = @10) */
  for (rr=w19+0, ss=w10; rr!=w19+32; rr+=4) *rr = *ss++;
  /* #124: @1 = zeros(1x4) */
  casadi_clear(w1, 4);
  /* #125: @21 = 00 */
  /* #126: @28 = ones(12x1,2nz) */
  casadi_fill(w28, 2, 1.);
  /* #127: {NULL, @6, NULL, NULL, NULL, @8, NULL, NULL, NULL, NULL, NULL, NULL} = vertsplit(@28) */
  w6 = w28[0];
  w8 = w28[1];
  /* #128: @22 = 00 */
  /* #129: @23 = 00 */
  /* #130: @5 = vertcat(@21, @6, @22, @23) */
  rr=(&w5);
  *rr++ = w6;
  /* #131: @6 = @5' */
  casadi_copy((&w5), 1, (&w6));
  /* #132: @1 = mac(@6,@16,@1) */
  casadi_mtimes((&w6), casadi_s3, w16, casadi_s1, w1, casadi_s0, w, 0);
  /* #133: @1 = @1' */
  /* #134: @12 = zeros(1x4) */
  casadi_clear(w12, 4);
  /* #135: @5 = @5' */
  /* #136: @12 = mac(@5,@17,@12) */
  casadi_mtimes((&w5), casadi_s3, w17, casadi_s1, w12, casadi_s0, w, 0);
  /* #137: @12 = @12' */
  /* #138: @1 = (@1+@12) */
  for (i=0, rr=w1, cs=w12; i<4; ++i) (*rr++) += (*cs++);
  /* #139: {@5, @6, @0, @2} = vertsplit(@1) */
  w5 = w1[0];
  w6 = w1[1];
  w0 = w1[2];
  w2 = w1[3];
  /* #140: @10 = zeros(8x1) */
  casadi_clear(w10, 8);
  /* #141: @1 = zeros(1x4) */
  casadi_clear(w1, 4);
  /* #142: @7 = @8[0] */
  for (rr=(&w7), ss=(&w8)+0; ss!=(&w8)+1; ss+=1) *rr++ = *ss;
  /* #143: @7 = (-@7) */
  w7 = (- w7 );
  /* #144: @8 = @7' */
  casadi_copy((&w7), 1, (&w8));
  /* #145: @1 = mac(@8,@13,@1) */
  casadi_mtimes((&w8), casadi_s3, w13, casadi_s1, w1, casadi_s0, w, 0);
  /* #146: @1 = @1' */
  /* #147: @12 = zeros(1x4) */
  casadi_clear(w12, 4);
  /* #148: @7 = @7' */
  /* #149: @12 = mac(@7,@18,@12) */
  casadi_mtimes((&w7), casadi_s3, w18, casadi_s1, w12, casadi_s0, w, 0);
  /* #150: @12 = @12' */
  /* #151: @1 = (@1+@12) */
  for (i=0, rr=w1, cs=w12; i<4; ++i) (*rr++) += (*cs++);
  /* #152: @1 = (-@1) */
  for (i=0, rr=w1, cs=w1; i<4; ++i) *rr++ = (- *cs++ );
  /* #153: (@10[:4] += @1) */
  for (rr=w10+0, ss=w1; rr!=w10+4; rr+=1) *rr += *ss++;
  /* #154: {@7, @8, @3, @4, @9, @24, @25, @26} = vertsplit(@10) */
  w7 = w10[0];
  w8 = w10[1];
  w3 = w10[2];
  w4 = w10[3];
  w9 = w10[4];
  w24 = w10[5];
  w25 = w10[6];
  w26 = w10[7];
  /* #155: @27 = vertcat(@5, @6, @0, @2, @7, @8, @3, @4, @9, @24, @25, @26) */
  rr=w27;
  *rr++ = w5;
  *rr++ = w6;
  *rr++ = w0;
  *rr++ = w2;
  *rr++ = w7;
  *rr++ = w8;
  *rr++ = w3;
  *rr++ = w4;
  *rr++ = w9;
  *rr++ = w24;
  *rr++ = w25;
  *rr++ = w26;
  /* #156: @10 = @27[:8] */
  for (rr=w10, ss=w27+0; ss!=w27+8; ss+=1) *rr++ = *ss;
  /* #157: (@19[:32:16;4:8] = @10) */
  for (rr=w19+0, ss=w10; rr!=w19+32; rr+=16) for (tt=rr+4; tt!=rr+8; tt+=1) *tt = *ss++;
  /* #158: @10 = @27[:8] */
  for (rr=w10, ss=w27+0; ss!=w27+8; ss+=1) *rr++ = *ss;
  /* #159: (@19[1:33:4] = @10) */
  for (rr=w19+1, ss=w10; rr!=w19+33; rr+=4) *rr = *ss++;
  /* #160: @1 = zeros(1x4) */
  casadi_clear(w1, 4);
  /* #161: @21 = 00 */
  /* #162: @22 = 00 */
  /* #163: @28 = ones(12x1,2nz) */
  casadi_fill(w28, 2, 1.);
  /* #164: {NULL, NULL, @5, NULL, NULL, NULL, @6, NULL, NULL, NULL, NULL, NULL} = vertsplit(@28) */
  w5 = w28[0];
  w6 = w28[1];
  /* #165: @23 = 00 */
  /* #166: @0 = vertcat(@21, @22, @5, @23) */
  rr=(&w0);
  *rr++ = w5;
  /* #167: @5 = @0' */
  casadi_copy((&w0), 1, (&w5));
  /* #168: @1 = mac(@5,@16,@1) */
  casadi_mtimes((&w5), casadi_s4, w16, casadi_s1, w1, casadi_s0, w, 0);
  /* #169: @1 = @1' */
  /* #170: @12 = zeros(1x4) */
  casadi_clear(w12, 4);
  /* #171: @0 = @0' */
  /* #172: @12 = mac(@0,@17,@12) */
  casadi_mtimes((&w0), casadi_s4, w17, casadi_s1, w12, casadi_s0, w, 0);
  /* #173: @12 = @12' */
  /* #174: @1 = (@1+@12) */
  for (i=0, rr=w1, cs=w12; i<4; ++i) (*rr++) += (*cs++);
  /* #175: {@0, @5, @2, @7} = vertsplit(@1) */
  w0 = w1[0];
  w5 = w1[1];
  w2 = w1[2];
  w7 = w1[3];
  /* #176: @10 = zeros(8x1) */
  casadi_clear(w10, 8);
  /* #177: @1 = zeros(1x4) */
  casadi_clear(w1, 4);
  /* #178: @8 = @6[0] */
  for (rr=(&w8), ss=(&w6)+0; ss!=(&w6)+1; ss+=1) *rr++ = *ss;
  /* #179: @8 = (-@8) */
  w8 = (- w8 );
  /* #180: @6 = @8' */
  casadi_copy((&w8), 1, (&w6));
  /* #181: @1 = mac(@6,@13,@1) */
  casadi_mtimes((&w6), casadi_s4, w13, casadi_s1, w1, casadi_s0, w, 0);
  /* #182: @1 = @1' */
  /* #183: @12 = zeros(1x4) */
  casadi_clear(w12, 4);
  /* #184: @8 = @8' */
  /* #185: @12 = mac(@8,@18,@12) */
  casadi_mtimes((&w8), casadi_s4, w18, casadi_s1, w12, casadi_s0, w, 0);
  /* #186: @12 = @12' */
  /* #187: @1 = (@1+@12) */
  for (i=0, rr=w1, cs=w12; i<4; ++i) (*rr++) += (*cs++);
  /* #188: @1 = (-@1) */
  for (i=0, rr=w1, cs=w1; i<4; ++i) *rr++ = (- *cs++ );
  /* #189: (@10[:4] += @1) */
  for (rr=w10+0, ss=w1; rr!=w10+4; rr+=1) *rr += *ss++;
  /* #190: {@8, @6, @3, @4, @9, @24, @25, @26} = vertsplit(@10) */
  w8 = w10[0];
  w6 = w10[1];
  w3 = w10[2];
  w4 = w10[3];
  w9 = w10[4];
  w24 = w10[5];
  w25 = w10[6];
  w26 = w10[7];
  /* #191: @27 = vertcat(@0, @5, @2, @7, @8, @6, @3, @4, @9, @24, @25, @26) */
  rr=w27;
  *rr++ = w0;
  *rr++ = w5;
  *rr++ = w2;
  *rr++ = w7;
  *rr++ = w8;
  *rr++ = w6;
  *rr++ = w3;
  *rr++ = w4;
  *rr++ = w9;
  *rr++ = w24;
  *rr++ = w25;
  *rr++ = w26;
  /* #192: @10 = @27[:8] */
  for (rr=w10, ss=w27+0; ss!=w27+8; ss+=1) *rr++ = *ss;
  /* #193: (@19[:32:16;8:12] = @10) */
  for (rr=w19+0, ss=w10; rr!=w19+32; rr+=16) for (tt=rr+8; tt!=rr+12; tt+=1) *tt = *ss++;
  /* #194: @10 = @27[:8] */
  for (rr=w10, ss=w27+0; ss!=w27+8; ss+=1) *rr++ = *ss;
  /* #195: (@19[2:34:4] = @10) */
  for (rr=w19+2, ss=w10; rr!=w19+34; rr+=4) *rr = *ss++;
  /* #196: @1 = zeros(1x4) */
  casadi_clear(w1, 4);
  /* #197: @21 = 00 */
  /* #198: @22 = 00 */
  /* #199: @23 = 00 */
  /* #200: @28 = ones(12x1,2nz) */
  casadi_fill(w28, 2, 1.);
  /* #201: {NULL, NULL, NULL, @0, NULL, NULL, NULL, @5, NULL, NULL, NULL, NULL} = vertsplit(@28) */
  w0 = w28[0];
  w5 = w28[1];
  /* #202: @2 = vertcat(@21, @22, @23, @0) */
  rr=(&w2);
  *rr++ = w0;
  /* #203: @0 = @2' */
  casadi_copy((&w2), 1, (&w0));
  /* #204: @1 = mac(@0,@16,@1) */
  casadi_mtimes((&w0), casadi_s5, w16, casadi_s1, w1, casadi_s0, w, 0);
  /* #205: @1 = @1' */
  /* #206: @12 = zeros(1x4) */
  casadi_clear(w12, 4);
  /* #207: @2 = @2' */
  /* #208: @12 = mac(@2,@17,@12) */
  casadi_mtimes((&w2), casadi_s5, w17, casadi_s1, w12, casadi_s0, w, 0);
  /* #209: @12 = @12' */
  /* #210: @1 = (@1+@12) */
  for (i=0, rr=w1, cs=w12; i<4; ++i) (*rr++) += (*cs++);
  /* #211: {@2, @0, @7, @8} = vertsplit(@1) */
  w2 = w1[0];
  w0 = w1[1];
  w7 = w1[2];
  w8 = w1[3];
  /* #212: @10 = zeros(8x1) */
  casadi_clear(w10, 8);
  /* #213: @1 = zeros(1x4) */
  casadi_clear(w1, 4);
  /* #214: @6 = @5[0] */
  for (rr=(&w6), ss=(&w5)+0; ss!=(&w5)+1; ss+=1) *rr++ = *ss;
  /* #215: @6 = (-@6) */
  w6 = (- w6 );
  /* #216: @5 = @6' */
  casadi_copy((&w6), 1, (&w5));
  /* #217: @1 = mac(@5,@13,@1) */
  casadi_mtimes((&w5), casadi_s5, w13, casadi_s1, w1, casadi_s0, w, 0);
  /* #218: @1 = @1' */
  /* #219: @12 = zeros(1x4) */
  casadi_clear(w12, 4);
  /* #220: @6 = @6' */
  /* #221: @12 = mac(@6,@18,@12) */
  casadi_mtimes((&w6), casadi_s5, w18, casadi_s1, w12, casadi_s0, w, 0);
  /* #222: @12 = @12' */
  /* #223: @1 = (@1+@12) */
  for (i=0, rr=w1, cs=w12; i<4; ++i) (*rr++) += (*cs++);
  /* #224: @1 = (-@1) */
  for (i=0, rr=w1, cs=w1; i<4; ++i) *rr++ = (- *cs++ );
  /* #225: (@10[:4] += @1) */
  for (rr=w10+0, ss=w1; rr!=w10+4; rr+=1) *rr += *ss++;
  /* #226: {@6, @5, @3, @4, @9, @24, @25, @26} = vertsplit(@10) */
  w6 = w10[0];
  w5 = w10[1];
  w3 = w10[2];
  w4 = w10[3];
  w9 = w10[4];
  w24 = w10[5];
  w25 = w10[6];
  w26 = w10[7];
  /* #227: @27 = vertcat(@2, @0, @7, @8, @6, @5, @3, @4, @9, @24, @25, @26) */
  rr=w27;
  *rr++ = w2;
  *rr++ = w0;
  *rr++ = w7;
  *rr++ = w8;
  *rr++ = w6;
  *rr++ = w5;
  *rr++ = w3;
  *rr++ = w4;
  *rr++ = w9;
  *rr++ = w24;
  *rr++ = w25;
  *rr++ = w26;
  /* #228: @10 = @27[:8] */
  for (rr=w10, ss=w27+0; ss!=w27+8; ss+=1) *rr++ = *ss;
  /* #229: (@19[:32:16;12:16] = @10) */
  for (rr=w19+0, ss=w10; rr!=w19+32; rr+=16) for (tt=rr+12; tt!=rr+16; tt+=1) *tt = *ss++;
  /* #230: @10 = @27[:8] */
  for (rr=w10, ss=w27+0; ss!=w27+8; ss+=1) *rr++ = *ss;
  /* #231: (@19[3:35:4] = @10) */
  for (rr=w19+3, ss=w10; rr!=w19+35; rr+=4) *rr = *ss++;
  /* #232: @29 = @19' */
  casadi_trans(w19,casadi_s6, w29, casadi_s6, iw);
  /* #233: output[2][0] = @29 */
  casadi_copy(w29, 32, res[2]);
  return 0;
}

CASADI_SYMBOL_EXPORT int Drone_ode_cost_ext_cost_fun_jac_hess(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int Drone_ode_cost_ext_cost_fun_jac_hess_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int Drone_ode_cost_ext_cost_fun_jac_hess_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void Drone_ode_cost_ext_cost_fun_jac_hess_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int Drone_ode_cost_ext_cost_fun_jac_hess_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void Drone_ode_cost_ext_cost_fun_jac_hess_release(int mem) {
}

CASADI_SYMBOL_EXPORT void Drone_ode_cost_ext_cost_fun_jac_hess_incref(void) {
}

CASADI_SYMBOL_EXPORT void Drone_ode_cost_ext_cost_fun_jac_hess_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int Drone_ode_cost_ext_cost_fun_jac_hess_n_in(void) { return 4;}

CASADI_SYMBOL_EXPORT casadi_int Drone_ode_cost_ext_cost_fun_jac_hess_n_out(void) { return 5;}

CASADI_SYMBOL_EXPORT casadi_real Drone_ode_cost_ext_cost_fun_jac_hess_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* Drone_ode_cost_ext_cost_fun_jac_hess_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* Drone_ode_cost_ext_cost_fun_jac_hess_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    case 1: return "o1";
    case 2: return "o2";
    case 3: return "o3";
    case 4: return "o4";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* Drone_ode_cost_ext_cost_fun_jac_hess_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s7;
    case 1: return casadi_s8;
    case 2: return casadi_s9;
    case 3: return casadi_s7;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* Drone_ode_cost_ext_cost_fun_jac_hess_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s10;
    case 1: return casadi_s11;
    case 2: return casadi_s6;
    case 3: return casadi_s9;
    case 4: return casadi_s12;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int Drone_ode_cost_ext_cost_fun_jac_hess_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 16;
  if (sz_res) *sz_res = 17;
  if (sz_iw) *sz_iw = 13;
  if (sz_w) *sz_w = 189;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif

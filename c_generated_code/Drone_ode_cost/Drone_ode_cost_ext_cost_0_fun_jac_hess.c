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
  #define CASADI_PREFIX(ID) Drone_ode_cost_ext_cost_0_fun_jac_hess_ ## ID
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
static const casadi_int casadi_s10[16] = {12, 1, 0, 12, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
static const casadi_int casadi_s11[5] = {1, 1, 0, 1, 0};
static const casadi_int casadi_s12[15] = {0, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

/* Drone_ode_cost_ext_cost_0_fun_jac_hess:(i0[8],i1[4],i2[],i3[12])->(o0,o1[12],o2[12x12,32nz],o3[],o4[0x12]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_int i, j, k;
  casadi_real *rr, *ss, *tt;
  const casadi_real *cs;
  casadi_real w0, *w1=w+2, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, *w14=w+18, *w15=w+30, *w16=w+34, *w17=w+42, *w18=w+46, *w19=w+62, *w20=w+66, *w21=w+70, *w22=w+86, *w23=w+102, *w24=w+118, *w25=w+150, *w29=w+156, *w30=w+158;
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
  /* #10: @10 = input[3][8] */
  w10 = arg[3] ? arg[3][8] : 0;
  /* #11: @11 = input[3][9] */
  w11 = arg[3] ? arg[3][9] : 0;
  /* #12: @12 = input[3][10] */
  w12 = arg[3] ? arg[3][10] : 0;
  /* #13: @13 = input[3][11] */
  w13 = arg[3] ? arg[3][11] : 0;
  /* #14: @14 = vertcat(@2, @3, @4, @5, @6, @7, @8, @9, @10, @11, @12, @13) */
  rr=w14;
  *rr++ = w2;
  *rr++ = w3;
  *rr++ = w4;
  *rr++ = w5;
  *rr++ = w6;
  *rr++ = w7;
  *rr++ = w8;
  *rr++ = w9;
  *rr++ = w10;
  *rr++ = w11;
  *rr++ = w12;
  *rr++ = w13;
  /* #15: @15 = @14[:4] */
  for (rr=w15, ss=w14+0; ss!=w14+4; ss+=1) *rr++ = *ss;
  /* #16: @2 = input[0][0] */
  w2 = arg[0] ? arg[0][0] : 0;
  /* #17: @3 = input[0][1] */
  w3 = arg[0] ? arg[0][1] : 0;
  /* #18: @4 = input[0][2] */
  w4 = arg[0] ? arg[0][2] : 0;
  /* #19: @5 = input[0][3] */
  w5 = arg[0] ? arg[0][3] : 0;
  /* #20: @6 = input[0][4] */
  w6 = arg[0] ? arg[0][4] : 0;
  /* #21: @7 = input[0][5] */
  w7 = arg[0] ? arg[0][5] : 0;
  /* #22: @8 = input[0][6] */
  w8 = arg[0] ? arg[0][6] : 0;
  /* #23: @9 = input[0][7] */
  w9 = arg[0] ? arg[0][7] : 0;
  /* #24: @16 = vertcat(@2, @3, @4, @5, @6, @7, @8, @9) */
  rr=w16;
  *rr++ = w2;
  *rr++ = w3;
  *rr++ = w4;
  *rr++ = w5;
  *rr++ = w6;
  *rr++ = w7;
  *rr++ = w8;
  *rr++ = w9;
  /* #25: @17 = @16[:4] */
  for (rr=w17, ss=w16+0; ss!=w16+4; ss+=1) *rr++ = *ss;
  /* #26: @15 = (@15-@17) */
  for (i=0, rr=w15, cs=w17; i<4; ++i) (*rr++) -= (*cs++);
  /* #27: @17 = @15' */
  casadi_copy(w15, 4, w17);
  /* #28: @18 = zeros(4x4) */
  casadi_clear(w18, 16);
  /* #29: @2 = 3 */
  w2 = 3.;
  /* #30: (@18[0] = @2) */
  for (rr=w18+0, ss=(&w2); rr!=w18+1; rr+=1) *rr = *ss++;
  /* #31: @2 = 3 */
  w2 = 3.;
  /* #32: (@18[5] = @2) */
  for (rr=w18+5, ss=(&w2); rr!=w18+6; rr+=1) *rr = *ss++;
  /* #33: @2 = 3 */
  w2 = 3.;
  /* #34: (@18[10] = @2) */
  for (rr=w18+10, ss=(&w2); rr!=w18+11; rr+=1) *rr = *ss++;
  /* #35: @2 = 0.5 */
  w2 = 5.0000000000000000e-01;
  /* #36: (@18[15] = @2) */
  for (rr=w18+15, ss=(&w2); rr!=w18+16; rr+=1) *rr = *ss++;
  /* #37: @1 = mac(@17,@18,@1) */
  for (i=0, rr=w1; i<4; ++i) for (j=0; j<1; ++j, ++rr) for (k=0, ss=w17+j, tt=w18+i*4; k<4; ++k) *rr += ss[k*1]**tt++;
  /* #38: @0 = mac(@1,@15,@0) */
  for (i=0, rr=(&w0); i<1; ++i) for (j=0; j<1; ++j, ++rr) for (k=0, ss=w1+j, tt=w15+i*4; k<4; ++k) *rr += ss[k*1]**tt++;
  /* #39: @2 = 0 */
  w2 = 0.;
  /* #40: @17 = zeros(1x4) */
  casadi_clear(w17, 4);
  /* #41: @3 = input[1][0] */
  w3 = arg[1] ? arg[1][0] : 0;
  /* #42: @4 = input[1][1] */
  w4 = arg[1] ? arg[1][1] : 0;
  /* #43: @5 = input[1][2] */
  w5 = arg[1] ? arg[1][2] : 0;
  /* #44: @6 = input[1][3] */
  w6 = arg[1] ? arg[1][3] : 0;
  /* #45: @19 = vertcat(@3, @4, @5, @6) */
  rr=w19;
  *rr++ = w3;
  *rr++ = w4;
  *rr++ = w5;
  *rr++ = w6;
  /* #46: @20 = @19' */
  casadi_copy(w19, 4, w20);
  /* #47: @21 = zeros(4x4) */
  casadi_clear(w21, 16);
  /* #48: @3 = 0.5 */
  w3 = 5.0000000000000000e-01;
  /* #49: (@21[0] = @3) */
  for (rr=w21+0, ss=(&w3); rr!=w21+1; rr+=1) *rr = *ss++;
  /* #50: @3 = 0.5 */
  w3 = 5.0000000000000000e-01;
  /* #51: (@21[5] = @3) */
  for (rr=w21+5, ss=(&w3); rr!=w21+6; rr+=1) *rr = *ss++;
  /* #52: @3 = 0.5 */
  w3 = 5.0000000000000000e-01;
  /* #53: (@21[10] = @3) */
  for (rr=w21+10, ss=(&w3); rr!=w21+11; rr+=1) *rr = *ss++;
  /* #54: @3 = 1.5 */
  w3 = 1.5000000000000000e+00;
  /* #55: (@21[15] = @3) */
  for (rr=w21+15, ss=(&w3); rr!=w21+16; rr+=1) *rr = *ss++;
  /* #56: @17 = mac(@20,@21,@17) */
  for (i=0, rr=w17; i<4; ++i) for (j=0; j<1; ++j, ++rr) for (k=0, ss=w20+j, tt=w21+i*4; k<4; ++k) *rr += ss[k*1]**tt++;
  /* #57: @2 = mac(@17,@19,@2) */
  for (i=0, rr=(&w2); i<1; ++i) for (j=0; j<1; ++j, ++rr) for (k=0, ss=w17+j, tt=w19+i*4; k<4; ++k) *rr += ss[k*1]**tt++;
  /* #58: @0 = (@0+@2) */
  w0 += w2;
  /* #59: output[0][0] = @0 */
  if (res[0]) res[0][0] = w0;
  /* #60: @17 = @17' */
  /* #61: @20 = zeros(1x4) */
  casadi_clear(w20, 4);
  /* #62: @19 = @19' */
  /* #63: @22 = @21' */
  for (i=0, rr=w22, cs=w21; i<4; ++i) for (j=0; j<4; ++j) rr[i+j*4] = *cs++;
  /* #64: @20 = mac(@19,@22,@20) */
  for (i=0, rr=w20; i<4; ++i) for (j=0; j<1; ++j, ++rr) for (k=0, ss=w19+j, tt=w22+i*4; k<4; ++k) *rr += ss[k*1]**tt++;
  /* #65: @20 = @20' */
  /* #66: @17 = (@17+@20) */
  for (i=0, rr=w17, cs=w20; i<4; ++i) (*rr++) += (*cs++);
  /* #67: {@0, @2, @3, @4} = vertsplit(@17) */
  w0 = w17[0];
  w2 = w17[1];
  w3 = w17[2];
  w4 = w17[3];
  /* #68: output[1][0] = @0 */
  if (res[1]) res[1][0] = w0;
  /* #69: output[1][1] = @2 */
  if (res[1]) res[1][1] = w2;
  /* #70: output[1][2] = @3 */
  if (res[1]) res[1][2] = w3;
  /* #71: output[1][3] = @4 */
  if (res[1]) res[1][3] = w4;
  /* #72: @16 = zeros(8x1) */
  casadi_clear(w16, 8);
  /* #73: @1 = @1' */
  /* #74: @17 = zeros(1x4) */
  casadi_clear(w17, 4);
  /* #75: @15 = @15' */
  /* #76: @23 = @18' */
  for (i=0, rr=w23, cs=w18; i<4; ++i) for (j=0; j<4; ++j) rr[i+j*4] = *cs++;
  /* #77: @17 = mac(@15,@23,@17) */
  for (i=0, rr=w17; i<4; ++i) for (j=0; j<1; ++j, ++rr) for (k=0, ss=w15+j, tt=w23+i*4; k<4; ++k) *rr += ss[k*1]**tt++;
  /* #78: @17 = @17' */
  /* #79: @1 = (@1+@17) */
  for (i=0, rr=w1, cs=w17; i<4; ++i) (*rr++) += (*cs++);
  /* #80: @1 = (-@1) */
  for (i=0, rr=w1, cs=w1; i<4; ++i) *rr++ = (- *cs++ );
  /* #81: (@16[:4] += @1) */
  for (rr=w16+0, ss=w1; rr!=w16+4; rr+=1) *rr += *ss++;
  /* #82: {@4, @3, @2, @0, @5, @6, @7, @8} = vertsplit(@16) */
  w4 = w16[0];
  w3 = w16[1];
  w2 = w16[2];
  w0 = w16[3];
  w5 = w16[4];
  w6 = w16[5];
  w7 = w16[6];
  w8 = w16[7];
  /* #83: output[1][4] = @4 */
  if (res[1]) res[1][4] = w4;
  /* #84: output[1][5] = @3 */
  if (res[1]) res[1][5] = w3;
  /* #85: output[1][6] = @2 */
  if (res[1]) res[1][6] = w2;
  /* #86: output[1][7] = @0 */
  if (res[1]) res[1][7] = w0;
  /* #87: output[1][8] = @5 */
  if (res[1]) res[1][8] = w5;
  /* #88: output[1][9] = @6 */
  if (res[1]) res[1][9] = w6;
  /* #89: output[1][10] = @7 */
  if (res[1]) res[1][10] = w7;
  /* #90: output[1][11] = @8 */
  if (res[1]) res[1][11] = w8;
  /* #91: @24 = zeros(12x12,32nz) */
  casadi_clear(w24, 32);
  /* #92: @1 = zeros(1x4) */
  casadi_clear(w1, 4);
  /* #93: @25 = ones(12x1,6nz) */
  casadi_fill(w25, 6, 1.);
  /* #94: {@8, NULL, NULL, NULL, @7, NULL, NULL, NULL, NULL, NULL, NULL, NULL} = vertsplit(@25) */
  w8 = w25[0];
  w7 = w25[1];
  /* #95: @26 = 00 */
  /* #96: @27 = 00 */
  /* #97: @28 = 00 */
  /* #98: @6 = vertcat(@8, @26, @27, @28) */
  rr=(&w6);
  *rr++ = w8;
  /* #99: @8 = @6' */
  casadi_copy((&w6), 1, (&w8));
  /* #100: @1 = mac(@8,@21,@1) */
  casadi_mtimes((&w8), casadi_s2, w21, casadi_s1, w1, casadi_s0, w, 0);
  /* #101: @1 = @1' */
  /* #102: @17 = zeros(1x4) */
  casadi_clear(w17, 4);
  /* #103: @6 = @6' */
  /* #104: @17 = mac(@6,@22,@17) */
  casadi_mtimes((&w6), casadi_s2, w22, casadi_s1, w17, casadi_s0, w, 0);
  /* #105: @17 = @17' */
  /* #106: @1 = (@1+@17) */
  for (i=0, rr=w1, cs=w17; i<4; ++i) (*rr++) += (*cs++);
  /* #107: {@6, @8, @5, @0} = vertsplit(@1) */
  w6 = w1[0];
  w8 = w1[1];
  w5 = w1[2];
  w0 = w1[3];
  /* #108: @16 = zeros(8x1) */
  casadi_clear(w16, 8);
  /* #109: @1 = zeros(1x4) */
  casadi_clear(w1, 4);
  /* #110: @2 = @7[0] */
  for (rr=(&w2), ss=(&w7)+0; ss!=(&w7)+1; ss+=1) *rr++ = *ss;
  /* #111: @2 = (-@2) */
  w2 = (- w2 );
  /* #112: @7 = @2' */
  casadi_copy((&w2), 1, (&w7));
  /* #113: @1 = mac(@7,@18,@1) */
  casadi_mtimes((&w7), casadi_s2, w18, casadi_s1, w1, casadi_s0, w, 0);
  /* #114: @1 = @1' */
  /* #115: @17 = zeros(1x4) */
  casadi_clear(w17, 4);
  /* #116: @2 = @2' */
  /* #117: @17 = mac(@2,@23,@17) */
  casadi_mtimes((&w2), casadi_s2, w23, casadi_s1, w17, casadi_s0, w, 0);
  /* #118: @17 = @17' */
  /* #119: @1 = (@1+@17) */
  for (i=0, rr=w1, cs=w17; i<4; ++i) (*rr++) += (*cs++);
  /* #120: @1 = (-@1) */
  for (i=0, rr=w1, cs=w1; i<4; ++i) *rr++ = (- *cs++ );
  /* #121: (@16[:4] += @1) */
  for (rr=w16+0, ss=w1; rr!=w16+4; rr+=1) *rr += *ss++;
  /* #122: {@2, @7, @3, @4, @9, @10, @11, @12} = vertsplit(@16) */
  w2 = w16[0];
  w7 = w16[1];
  w3 = w16[2];
  w4 = w16[3];
  w9 = w16[4];
  w10 = w16[5];
  w11 = w16[6];
  w12 = w16[7];
  /* #123: @14 = vertcat(@6, @8, @5, @0, @2, @7, @3, @4, @9, @10, @11, @12) */
  rr=w14;
  *rr++ = w6;
  *rr++ = w8;
  *rr++ = w5;
  *rr++ = w0;
  *rr++ = w2;
  *rr++ = w7;
  *rr++ = w3;
  *rr++ = w4;
  *rr++ = w9;
  *rr++ = w10;
  *rr++ = w11;
  *rr++ = w12;
  /* #124: @16 = @14[:8] */
  for (rr=w16, ss=w14+0; ss!=w14+8; ss+=1) *rr++ = *ss;
  /* #125: (@24[:32:16;:4] = @16) */
  for (rr=w24+0, ss=w16; rr!=w24+32; rr+=16) for (tt=rr+0; tt!=rr+4; tt+=1) *tt = *ss++;
  /* #126: @16 = @14[:8] */
  for (rr=w16, ss=w14+0; ss!=w14+8; ss+=1) *rr++ = *ss;
  /* #127: (@24[:32:4] = @16) */
  for (rr=w24+0, ss=w16; rr!=w24+32; rr+=4) *rr = *ss++;
  /* #128: @1 = zeros(1x4) */
  casadi_clear(w1, 4);
  /* #129: @26 = 00 */
  /* #130: @29 = ones(12x1,2nz) */
  casadi_fill(w29, 2, 1.);
  /* #131: {NULL, @6, NULL, NULL, NULL, @8, NULL, NULL, NULL, NULL, NULL, NULL} = vertsplit(@29) */
  w6 = w29[0];
  w8 = w29[1];
  /* #132: @27 = 00 */
  /* #133: @28 = 00 */
  /* #134: @5 = vertcat(@26, @6, @27, @28) */
  rr=(&w5);
  *rr++ = w6;
  /* #135: @6 = @5' */
  casadi_copy((&w5), 1, (&w6));
  /* #136: @1 = mac(@6,@21,@1) */
  casadi_mtimes((&w6), casadi_s3, w21, casadi_s1, w1, casadi_s0, w, 0);
  /* #137: @1 = @1' */
  /* #138: @17 = zeros(1x4) */
  casadi_clear(w17, 4);
  /* #139: @5 = @5' */
  /* #140: @17 = mac(@5,@22,@17) */
  casadi_mtimes((&w5), casadi_s3, w22, casadi_s1, w17, casadi_s0, w, 0);
  /* #141: @17 = @17' */
  /* #142: @1 = (@1+@17) */
  for (i=0, rr=w1, cs=w17; i<4; ++i) (*rr++) += (*cs++);
  /* #143: {@5, @6, @0, @2} = vertsplit(@1) */
  w5 = w1[0];
  w6 = w1[1];
  w0 = w1[2];
  w2 = w1[3];
  /* #144: @16 = zeros(8x1) */
  casadi_clear(w16, 8);
  /* #145: @1 = zeros(1x4) */
  casadi_clear(w1, 4);
  /* #146: @7 = @8[0] */
  for (rr=(&w7), ss=(&w8)+0; ss!=(&w8)+1; ss+=1) *rr++ = *ss;
  /* #147: @7 = (-@7) */
  w7 = (- w7 );
  /* #148: @8 = @7' */
  casadi_copy((&w7), 1, (&w8));
  /* #149: @1 = mac(@8,@18,@1) */
  casadi_mtimes((&w8), casadi_s3, w18, casadi_s1, w1, casadi_s0, w, 0);
  /* #150: @1 = @1' */
  /* #151: @17 = zeros(1x4) */
  casadi_clear(w17, 4);
  /* #152: @7 = @7' */
  /* #153: @17 = mac(@7,@23,@17) */
  casadi_mtimes((&w7), casadi_s3, w23, casadi_s1, w17, casadi_s0, w, 0);
  /* #154: @17 = @17' */
  /* #155: @1 = (@1+@17) */
  for (i=0, rr=w1, cs=w17; i<4; ++i) (*rr++) += (*cs++);
  /* #156: @1 = (-@1) */
  for (i=0, rr=w1, cs=w1; i<4; ++i) *rr++ = (- *cs++ );
  /* #157: (@16[:4] += @1) */
  for (rr=w16+0, ss=w1; rr!=w16+4; rr+=1) *rr += *ss++;
  /* #158: {@7, @8, @3, @4, @9, @10, @11, @12} = vertsplit(@16) */
  w7 = w16[0];
  w8 = w16[1];
  w3 = w16[2];
  w4 = w16[3];
  w9 = w16[4];
  w10 = w16[5];
  w11 = w16[6];
  w12 = w16[7];
  /* #159: @14 = vertcat(@5, @6, @0, @2, @7, @8, @3, @4, @9, @10, @11, @12) */
  rr=w14;
  *rr++ = w5;
  *rr++ = w6;
  *rr++ = w0;
  *rr++ = w2;
  *rr++ = w7;
  *rr++ = w8;
  *rr++ = w3;
  *rr++ = w4;
  *rr++ = w9;
  *rr++ = w10;
  *rr++ = w11;
  *rr++ = w12;
  /* #160: @16 = @14[:8] */
  for (rr=w16, ss=w14+0; ss!=w14+8; ss+=1) *rr++ = *ss;
  /* #161: (@24[:32:16;4:8] = @16) */
  for (rr=w24+0, ss=w16; rr!=w24+32; rr+=16) for (tt=rr+4; tt!=rr+8; tt+=1) *tt = *ss++;
  /* #162: @16 = @14[:8] */
  for (rr=w16, ss=w14+0; ss!=w14+8; ss+=1) *rr++ = *ss;
  /* #163: (@24[1:33:4] = @16) */
  for (rr=w24+1, ss=w16; rr!=w24+33; rr+=4) *rr = *ss++;
  /* #164: @1 = zeros(1x4) */
  casadi_clear(w1, 4);
  /* #165: @26 = 00 */
  /* #166: @27 = 00 */
  /* #167: @29 = ones(12x1,2nz) */
  casadi_fill(w29, 2, 1.);
  /* #168: {NULL, NULL, @5, NULL, NULL, NULL, @6, NULL, NULL, NULL, NULL, NULL} = vertsplit(@29) */
  w5 = w29[0];
  w6 = w29[1];
  /* #169: @28 = 00 */
  /* #170: @0 = vertcat(@26, @27, @5, @28) */
  rr=(&w0);
  *rr++ = w5;
  /* #171: @5 = @0' */
  casadi_copy((&w0), 1, (&w5));
  /* #172: @1 = mac(@5,@21,@1) */
  casadi_mtimes((&w5), casadi_s4, w21, casadi_s1, w1, casadi_s0, w, 0);
  /* #173: @1 = @1' */
  /* #174: @17 = zeros(1x4) */
  casadi_clear(w17, 4);
  /* #175: @0 = @0' */
  /* #176: @17 = mac(@0,@22,@17) */
  casadi_mtimes((&w0), casadi_s4, w22, casadi_s1, w17, casadi_s0, w, 0);
  /* #177: @17 = @17' */
  /* #178: @1 = (@1+@17) */
  for (i=0, rr=w1, cs=w17; i<4; ++i) (*rr++) += (*cs++);
  /* #179: {@0, @5, @2, @7} = vertsplit(@1) */
  w0 = w1[0];
  w5 = w1[1];
  w2 = w1[2];
  w7 = w1[3];
  /* #180: @16 = zeros(8x1) */
  casadi_clear(w16, 8);
  /* #181: @1 = zeros(1x4) */
  casadi_clear(w1, 4);
  /* #182: @8 = @6[0] */
  for (rr=(&w8), ss=(&w6)+0; ss!=(&w6)+1; ss+=1) *rr++ = *ss;
  /* #183: @8 = (-@8) */
  w8 = (- w8 );
  /* #184: @6 = @8' */
  casadi_copy((&w8), 1, (&w6));
  /* #185: @1 = mac(@6,@18,@1) */
  casadi_mtimes((&w6), casadi_s4, w18, casadi_s1, w1, casadi_s0, w, 0);
  /* #186: @1 = @1' */
  /* #187: @17 = zeros(1x4) */
  casadi_clear(w17, 4);
  /* #188: @8 = @8' */
  /* #189: @17 = mac(@8,@23,@17) */
  casadi_mtimes((&w8), casadi_s4, w23, casadi_s1, w17, casadi_s0, w, 0);
  /* #190: @17 = @17' */
  /* #191: @1 = (@1+@17) */
  for (i=0, rr=w1, cs=w17; i<4; ++i) (*rr++) += (*cs++);
  /* #192: @1 = (-@1) */
  for (i=0, rr=w1, cs=w1; i<4; ++i) *rr++ = (- *cs++ );
  /* #193: (@16[:4] += @1) */
  for (rr=w16+0, ss=w1; rr!=w16+4; rr+=1) *rr += *ss++;
  /* #194: {@8, @6, @3, @4, @9, @10, @11, @12} = vertsplit(@16) */
  w8 = w16[0];
  w6 = w16[1];
  w3 = w16[2];
  w4 = w16[3];
  w9 = w16[4];
  w10 = w16[5];
  w11 = w16[6];
  w12 = w16[7];
  /* #195: @14 = vertcat(@0, @5, @2, @7, @8, @6, @3, @4, @9, @10, @11, @12) */
  rr=w14;
  *rr++ = w0;
  *rr++ = w5;
  *rr++ = w2;
  *rr++ = w7;
  *rr++ = w8;
  *rr++ = w6;
  *rr++ = w3;
  *rr++ = w4;
  *rr++ = w9;
  *rr++ = w10;
  *rr++ = w11;
  *rr++ = w12;
  /* #196: @16 = @14[:8] */
  for (rr=w16, ss=w14+0; ss!=w14+8; ss+=1) *rr++ = *ss;
  /* #197: (@24[:32:16;8:12] = @16) */
  for (rr=w24+0, ss=w16; rr!=w24+32; rr+=16) for (tt=rr+8; tt!=rr+12; tt+=1) *tt = *ss++;
  /* #198: @16 = @14[:8] */
  for (rr=w16, ss=w14+0; ss!=w14+8; ss+=1) *rr++ = *ss;
  /* #199: (@24[2:34:4] = @16) */
  for (rr=w24+2, ss=w16; rr!=w24+34; rr+=4) *rr = *ss++;
  /* #200: @1 = zeros(1x4) */
  casadi_clear(w1, 4);
  /* #201: @26 = 00 */
  /* #202: @27 = 00 */
  /* #203: @28 = 00 */
  /* #204: @29 = ones(12x1,2nz) */
  casadi_fill(w29, 2, 1.);
  /* #205: {NULL, NULL, NULL, @0, NULL, NULL, NULL, @5, NULL, NULL, NULL, NULL} = vertsplit(@29) */
  w0 = w29[0];
  w5 = w29[1];
  /* #206: @2 = vertcat(@26, @27, @28, @0) */
  rr=(&w2);
  *rr++ = w0;
  /* #207: @0 = @2' */
  casadi_copy((&w2), 1, (&w0));
  /* #208: @1 = mac(@0,@21,@1) */
  casadi_mtimes((&w0), casadi_s5, w21, casadi_s1, w1, casadi_s0, w, 0);
  /* #209: @1 = @1' */
  /* #210: @17 = zeros(1x4) */
  casadi_clear(w17, 4);
  /* #211: @2 = @2' */
  /* #212: @17 = mac(@2,@22,@17) */
  casadi_mtimes((&w2), casadi_s5, w22, casadi_s1, w17, casadi_s0, w, 0);
  /* #213: @17 = @17' */
  /* #214: @1 = (@1+@17) */
  for (i=0, rr=w1, cs=w17; i<4; ++i) (*rr++) += (*cs++);
  /* #215: {@2, @0, @7, @8} = vertsplit(@1) */
  w2 = w1[0];
  w0 = w1[1];
  w7 = w1[2];
  w8 = w1[3];
  /* #216: @16 = zeros(8x1) */
  casadi_clear(w16, 8);
  /* #217: @1 = zeros(1x4) */
  casadi_clear(w1, 4);
  /* #218: @6 = @5[0] */
  for (rr=(&w6), ss=(&w5)+0; ss!=(&w5)+1; ss+=1) *rr++ = *ss;
  /* #219: @6 = (-@6) */
  w6 = (- w6 );
  /* #220: @5 = @6' */
  casadi_copy((&w6), 1, (&w5));
  /* #221: @1 = mac(@5,@18,@1) */
  casadi_mtimes((&w5), casadi_s5, w18, casadi_s1, w1, casadi_s0, w, 0);
  /* #222: @1 = @1' */
  /* #223: @17 = zeros(1x4) */
  casadi_clear(w17, 4);
  /* #224: @6 = @6' */
  /* #225: @17 = mac(@6,@23,@17) */
  casadi_mtimes((&w6), casadi_s5, w23, casadi_s1, w17, casadi_s0, w, 0);
  /* #226: @17 = @17' */
  /* #227: @1 = (@1+@17) */
  for (i=0, rr=w1, cs=w17; i<4; ++i) (*rr++) += (*cs++);
  /* #228: @1 = (-@1) */
  for (i=0, rr=w1, cs=w1; i<4; ++i) *rr++ = (- *cs++ );
  /* #229: (@16[:4] += @1) */
  for (rr=w16+0, ss=w1; rr!=w16+4; rr+=1) *rr += *ss++;
  /* #230: {@6, @5, @3, @4, @9, @10, @11, @12} = vertsplit(@16) */
  w6 = w16[0];
  w5 = w16[1];
  w3 = w16[2];
  w4 = w16[3];
  w9 = w16[4];
  w10 = w16[5];
  w11 = w16[6];
  w12 = w16[7];
  /* #231: @14 = vertcat(@2, @0, @7, @8, @6, @5, @3, @4, @9, @10, @11, @12) */
  rr=w14;
  *rr++ = w2;
  *rr++ = w0;
  *rr++ = w7;
  *rr++ = w8;
  *rr++ = w6;
  *rr++ = w5;
  *rr++ = w3;
  *rr++ = w4;
  *rr++ = w9;
  *rr++ = w10;
  *rr++ = w11;
  *rr++ = w12;
  /* #232: @16 = @14[:8] */
  for (rr=w16, ss=w14+0; ss!=w14+8; ss+=1) *rr++ = *ss;
  /* #233: (@24[:32:16;12:16] = @16) */
  for (rr=w24+0, ss=w16; rr!=w24+32; rr+=16) for (tt=rr+12; tt!=rr+16; tt+=1) *tt = *ss++;
  /* #234: @16 = @14[:8] */
  for (rr=w16, ss=w14+0; ss!=w14+8; ss+=1) *rr++ = *ss;
  /* #235: (@24[3:35:4] = @16) */
  for (rr=w24+3, ss=w16; rr!=w24+35; rr+=4) *rr = *ss++;
  /* #236: @30 = @24' */
  casadi_trans(w24,casadi_s6, w30, casadi_s6, iw);
  /* #237: output[2][0] = @30 */
  casadi_copy(w30, 32, res[2]);
  return 0;
}

CASADI_SYMBOL_EXPORT int Drone_ode_cost_ext_cost_0_fun_jac_hess(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int Drone_ode_cost_ext_cost_0_fun_jac_hess_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int Drone_ode_cost_ext_cost_0_fun_jac_hess_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void Drone_ode_cost_ext_cost_0_fun_jac_hess_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int Drone_ode_cost_ext_cost_0_fun_jac_hess_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void Drone_ode_cost_ext_cost_0_fun_jac_hess_release(int mem) {
}

CASADI_SYMBOL_EXPORT void Drone_ode_cost_ext_cost_0_fun_jac_hess_incref(void) {
}

CASADI_SYMBOL_EXPORT void Drone_ode_cost_ext_cost_0_fun_jac_hess_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int Drone_ode_cost_ext_cost_0_fun_jac_hess_n_in(void) { return 4;}

CASADI_SYMBOL_EXPORT casadi_int Drone_ode_cost_ext_cost_0_fun_jac_hess_n_out(void) { return 5;}

CASADI_SYMBOL_EXPORT casadi_real Drone_ode_cost_ext_cost_0_fun_jac_hess_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* Drone_ode_cost_ext_cost_0_fun_jac_hess_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* Drone_ode_cost_ext_cost_0_fun_jac_hess_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    case 1: return "o1";
    case 2: return "o2";
    case 3: return "o3";
    case 4: return "o4";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* Drone_ode_cost_ext_cost_0_fun_jac_hess_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s7;
    case 1: return casadi_s8;
    case 2: return casadi_s9;
    case 3: return casadi_s10;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* Drone_ode_cost_ext_cost_0_fun_jac_hess_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s11;
    case 1: return casadi_s10;
    case 2: return casadi_s6;
    case 3: return casadi_s9;
    case 4: return casadi_s12;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int Drone_ode_cost_ext_cost_0_fun_jac_hess_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 16;
  if (sz_res) *sz_res = 17;
  if (sz_iw) *sz_iw = 13;
  if (sz_w) *sz_w = 190;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif

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
#define casadi_s13 CASADI_PREFIX(s13)
#define casadi_s14 CASADI_PREFIX(s14)
#define casadi_s15 CASADI_PREFIX(s15)
#define casadi_s16 CASADI_PREFIX(s16)
#define casadi_s17 CASADI_PREFIX(s17)
#define casadi_s18 CASADI_PREFIX(s18)
#define casadi_s19 CASADI_PREFIX(s19)
#define casadi_s2 CASADI_PREFIX(s2)
#define casadi_s20 CASADI_PREFIX(s20)
#define casadi_s21 CASADI_PREFIX(s21)
#define casadi_s22 CASADI_PREFIX(s22)
#define casadi_s23 CASADI_PREFIX(s23)
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
static const casadi_int casadi_s3[9] = {1, 3, 0, 1, 2, 3, 0, 0, 0};
static const casadi_int casadi_s4[15] = {3, 3, 0, 3, 6, 9, 0, 1, 2, 0, 1, 2, 0, 1, 2};
static const casadi_int casadi_s5[7] = {1, 3, 0, 1, 1, 1, 0};
static const casadi_int casadi_s6[7] = {0, 1, 2, 3, 16, 17, 18};
static const casadi_int casadi_s7[7] = {0, 4, 8, 12, 16, 19, 22};
static const casadi_int casadi_s8[8] = {1, 4, 0, 0, 1, 1, 1, 0};
static const casadi_int casadi_s9[7] = {1, 3, 0, 0, 1, 1, 0};
static const casadi_int casadi_s10[7] = {4, 5, 6, 7, 19, 20, 21};
static const casadi_int casadi_s11[7] = {1, 5, 9, 13, 17, 20, 23};
static const casadi_int casadi_s12[8] = {1, 4, 0, 0, 0, 1, 1, 0};
static const casadi_int casadi_s13[7] = {1, 3, 0, 0, 0, 1, 0};
static const casadi_int casadi_s14[7] = {8, 9, 10, 11, 22, 23, 24};
static const casadi_int casadi_s15[7] = {2, 6, 10, 14, 18, 21, 24};
static const casadi_int casadi_s16[8] = {1, 4, 0, 0, 0, 0, 1, 0};
static const casadi_int casadi_s17[43] = {15, 15, 0, 4, 8, 12, 16, 19, 22, 25, 25, 25, 25, 25, 25, 25, 25, 25, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 4, 5, 6, 4, 5, 6};
static const casadi_int casadi_s18[15] = {11, 1, 0, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
static const casadi_int casadi_s19[8] = {4, 1, 0, 4, 0, 1, 2, 3};
static const casadi_int casadi_s20[3] = {0, 0, 0};
static const casadi_int casadi_s21[19] = {15, 1, 0, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14};
static const casadi_int casadi_s22[5] = {1, 1, 0, 1, 0};
static const casadi_int casadi_s23[18] = {0, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

/* Drone_ode_cost_ext_cost_0_fun_jac_hess:(i0[11],i1[4],i2[],i3[15])->(o0,o1[15],o2[15x15,25nz],o3[],o4[0x15]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_int i, j, k;
  casadi_real *rr, *ss, *tt;
  const casadi_int *cii;
  const casadi_real *cs;
  casadi_real w0, *w1=w+2, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15, w16, *w17=w+20, *w18=w+35, *w19=w+38, *w20=w+49, *w21=w+52, *w22=w+61, *w23=w+65, *w24=w+69, *w25=w+73, *w26=w+89, *w27=w+105, *w28=w+114, *w29=w+139, *w33=w+149, *w34=w+156, *w43=w+158;
  /* #0: @0 = 0 */
  w0 = 0.;
  /* #1: @1 = zeros(1x3) */
  casadi_clear(w1, 3);
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
  /* #14: @14 = input[3][12] */
  w14 = arg[3] ? arg[3][12] : 0;
  /* #15: @15 = input[3][13] */
  w15 = arg[3] ? arg[3][13] : 0;
  /* #16: @16 = input[3][14] */
  w16 = arg[3] ? arg[3][14] : 0;
  /* #17: @17 = vertcat(@2, @3, @4, @5, @6, @7, @8, @9, @10, @11, @12, @13, @14, @15, @16) */
  rr=w17;
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
  *rr++ = w14;
  *rr++ = w15;
  *rr++ = w16;
  /* #18: @18 = @17[:3] */
  for (rr=w18, ss=w17+0; ss!=w17+3; ss+=1) *rr++ = *ss;
  /* #19: @2 = input[0][0] */
  w2 = arg[0] ? arg[0][0] : 0;
  /* #20: @3 = input[0][1] */
  w3 = arg[0] ? arg[0][1] : 0;
  /* #21: @4 = input[0][2] */
  w4 = arg[0] ? arg[0][2] : 0;
  /* #22: @5 = input[0][3] */
  w5 = arg[0] ? arg[0][3] : 0;
  /* #23: @6 = input[0][4] */
  w6 = arg[0] ? arg[0][4] : 0;
  /* #24: @7 = input[0][5] */
  w7 = arg[0] ? arg[0][5] : 0;
  /* #25: @8 = input[0][6] */
  w8 = arg[0] ? arg[0][6] : 0;
  /* #26: @9 = input[0][7] */
  w9 = arg[0] ? arg[0][7] : 0;
  /* #27: @10 = input[0][8] */
  w10 = arg[0] ? arg[0][8] : 0;
  /* #28: @11 = input[0][9] */
  w11 = arg[0] ? arg[0][9] : 0;
  /* #29: @12 = input[0][10] */
  w12 = arg[0] ? arg[0][10] : 0;
  /* #30: @19 = vertcat(@2, @3, @4, @5, @6, @7, @8, @9, @10, @11, @12) */
  rr=w19;
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
  /* #31: @20 = @19[:3] */
  for (rr=w20, ss=w19+0; ss!=w19+3; ss+=1) *rr++ = *ss;
  /* #32: @18 = (@18-@20) */
  for (i=0, rr=w18, cs=w20; i<3; ++i) (*rr++) -= (*cs++);
  /* #33: @20 = @18' */
  casadi_copy(w18, 3, w20);
  /* #34: @21 = zeros(3x3) */
  casadi_clear(w21, 9);
  /* #35: @2 = 1.1 */
  w2 = 1.1000000000000001e+00;
  /* #36: (@21[0] = @2) */
  for (rr=w21+0, ss=(&w2); rr!=w21+1; rr+=1) *rr = *ss++;
  /* #37: @2 = 1.1 */
  w2 = 1.1000000000000001e+00;
  /* #38: (@21[4] = @2) */
  for (rr=w21+4, ss=(&w2); rr!=w21+5; rr+=1) *rr = *ss++;
  /* #39: @2 = 1.1 */
  w2 = 1.1000000000000001e+00;
  /* #40: (@21[8] = @2) */
  for (rr=w21+8, ss=(&w2); rr!=w21+9; rr+=1) *rr = *ss++;
  /* #41: @1 = mac(@20,@21,@1) */
  for (i=0, rr=w1; i<3; ++i) for (j=0; j<1; ++j, ++rr) for (k=0, ss=w20+j, tt=w21+i*3; k<3; ++k) *rr += ss[k*1]**tt++;
  /* #42: @0 = mac(@1,@18,@0) */
  for (i=0, rr=(&w0); i<1; ++i) for (j=0; j<1; ++j, ++rr) for (k=0, ss=w1+j, tt=w18+i*3; k<3; ++k) *rr += ss[k*1]**tt++;
  /* #43: @2 = 0 */
  w2 = 0.;
  /* #44: @22 = zeros(1x4) */
  casadi_clear(w22, 4);
  /* #45: @3 = input[1][0] */
  w3 = arg[1] ? arg[1][0] : 0;
  /* #46: @4 = input[1][1] */
  w4 = arg[1] ? arg[1][1] : 0;
  /* #47: @5 = input[1][2] */
  w5 = arg[1] ? arg[1][2] : 0;
  /* #48: @6 = input[1][3] */
  w6 = arg[1] ? arg[1][3] : 0;
  /* #49: @23 = vertcat(@3, @4, @5, @6) */
  rr=w23;
  *rr++ = w3;
  *rr++ = w4;
  *rr++ = w5;
  *rr++ = w6;
  /* #50: @24 = @23' */
  casadi_copy(w23, 4, w24);
  /* #51: @25 = zeros(4x4) */
  casadi_clear(w25, 16);
  /* #52: @3 = 1 */
  w3 = 1.;
  /* #53: (@25[0] = @3) */
  for (rr=w25+0, ss=(&w3); rr!=w25+1; rr+=1) *rr = *ss++;
  /* #54: @3 = 1 */
  w3 = 1.;
  /* #55: (@25[5] = @3) */
  for (rr=w25+5, ss=(&w3); rr!=w25+6; rr+=1) *rr = *ss++;
  /* #56: @3 = 1 */
  w3 = 1.;
  /* #57: (@25[10] = @3) */
  for (rr=w25+10, ss=(&w3); rr!=w25+11; rr+=1) *rr = *ss++;
  /* #58: @3 = 1 */
  w3 = 1.;
  /* #59: (@25[15] = @3) */
  for (rr=w25+15, ss=(&w3); rr!=w25+16; rr+=1) *rr = *ss++;
  /* #60: @22 = mac(@24,@25,@22) */
  for (i=0, rr=w22; i<4; ++i) for (j=0; j<1; ++j, ++rr) for (k=0, ss=w24+j, tt=w25+i*4; k<4; ++k) *rr += ss[k*1]**tt++;
  /* #61: @2 = mac(@22,@23,@2) */
  for (i=0, rr=(&w2); i<1; ++i) for (j=0; j<1; ++j, ++rr) for (k=0, ss=w22+j, tt=w23+i*4; k<4; ++k) *rr += ss[k*1]**tt++;
  /* #62: @0 = (@0+@2) */
  w0 += w2;
  /* #63: output[0][0] = @0 */
  if (res[0]) res[0][0] = w0;
  /* #64: @22 = @22' */
  /* #65: @24 = zeros(1x4) */
  casadi_clear(w24, 4);
  /* #66: @23 = @23' */
  /* #67: @26 = @25' */
  for (i=0, rr=w26, cs=w25; i<4; ++i) for (j=0; j<4; ++j) rr[i+j*4] = *cs++;
  /* #68: @24 = mac(@23,@26,@24) */
  for (i=0, rr=w24; i<4; ++i) for (j=0; j<1; ++j, ++rr) for (k=0, ss=w23+j, tt=w26+i*4; k<4; ++k) *rr += ss[k*1]**tt++;
  /* #69: @24 = @24' */
  /* #70: @22 = (@22+@24) */
  for (i=0, rr=w22, cs=w24; i<4; ++i) (*rr++) += (*cs++);
  /* #71: {@0, @2, @3, @4} = vertsplit(@22) */
  w0 = w22[0];
  w2 = w22[1];
  w3 = w22[2];
  w4 = w22[3];
  /* #72: output[1][0] = @0 */
  if (res[1]) res[1][0] = w0;
  /* #73: output[1][1] = @2 */
  if (res[1]) res[1][1] = w2;
  /* #74: output[1][2] = @3 */
  if (res[1]) res[1][2] = w3;
  /* #75: output[1][3] = @4 */
  if (res[1]) res[1][3] = w4;
  /* #76: @19 = zeros(11x1) */
  casadi_clear(w19, 11);
  /* #77: @1 = @1' */
  /* #78: @20 = zeros(1x3) */
  casadi_clear(w20, 3);
  /* #79: @18 = @18' */
  /* #80: @27 = @21' */
  for (i=0, rr=w27, cs=w21; i<3; ++i) for (j=0; j<3; ++j) rr[i+j*3] = *cs++;
  /* #81: @20 = mac(@18,@27,@20) */
  for (i=0, rr=w20; i<3; ++i) for (j=0; j<1; ++j, ++rr) for (k=0, ss=w18+j, tt=w27+i*3; k<3; ++k) *rr += ss[k*1]**tt++;
  /* #82: @20 = @20' */
  /* #83: @1 = (@1+@20) */
  for (i=0, rr=w1, cs=w20; i<3; ++i) (*rr++) += (*cs++);
  /* #84: @1 = (-@1) */
  for (i=0, rr=w1, cs=w1; i<3; ++i) *rr++ = (- *cs++ );
  /* #85: (@19[:3] += @1) */
  for (rr=w19+0, ss=w1; rr!=w19+3; rr+=1) *rr += *ss++;
  /* #86: {@4, @3, @2, @0, @5, @6, @7, @8, @9, @10, @11} = vertsplit(@19) */
  w4 = w19[0];
  w3 = w19[1];
  w2 = w19[2];
  w0 = w19[3];
  w5 = w19[4];
  w6 = w19[5];
  w7 = w19[6];
  w8 = w19[7];
  w9 = w19[8];
  w10 = w19[9];
  w11 = w19[10];
  /* #87: output[1][4] = @4 */
  if (res[1]) res[1][4] = w4;
  /* #88: output[1][5] = @3 */
  if (res[1]) res[1][5] = w3;
  /* #89: output[1][6] = @2 */
  if (res[1]) res[1][6] = w2;
  /* #90: output[1][7] = @0 */
  if (res[1]) res[1][7] = w0;
  /* #91: output[1][8] = @5 */
  if (res[1]) res[1][8] = w5;
  /* #92: output[1][9] = @6 */
  if (res[1]) res[1][9] = w6;
  /* #93: output[1][10] = @7 */
  if (res[1]) res[1][10] = w7;
  /* #94: output[1][11] = @8 */
  if (res[1]) res[1][11] = w8;
  /* #95: output[1][12] = @9 */
  if (res[1]) res[1][12] = w9;
  /* #96: output[1][13] = @10 */
  if (res[1]) res[1][13] = w10;
  /* #97: output[1][14] = @11 */
  if (res[1]) res[1][14] = w11;
  /* #98: @28 = zeros(15x15,25nz) */
  casadi_clear(w28, 25);
  /* #99: @22 = zeros(1x4) */
  casadi_clear(w22, 4);
  /* #100: @29 = ones(15x1,10nz) */
  casadi_fill(w29, 10, 1.);
  /* #101: {@11, NULL, NULL, NULL, @10, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL} = vertsplit(@29) */
  w11 = w29[0];
  w10 = w29[1];
  /* #102: @30 = 00 */
  /* #103: @31 = 00 */
  /* #104: @32 = 00 */
  /* #105: @9 = vertcat(@11, @30, @31, @32) */
  rr=(&w9);
  *rr++ = w11;
  /* #106: @11 = @9' */
  casadi_copy((&w9), 1, (&w11));
  /* #107: @22 = mac(@11,@25,@22) */
  casadi_mtimes((&w11), casadi_s2, w25, casadi_s1, w22, casadi_s0, w, 0);
  /* #108: @22 = @22' */
  /* #109: @24 = zeros(1x4) */
  casadi_clear(w24, 4);
  /* #110: @9 = @9' */
  /* #111: @24 = mac(@9,@26,@24) */
  casadi_mtimes((&w9), casadi_s2, w26, casadi_s1, w24, casadi_s0, w, 0);
  /* #112: @24 = @24' */
  /* #113: @22 = (@22+@24) */
  for (i=0, rr=w22, cs=w24; i<4; ++i) (*rr++) += (*cs++);
  /* #114: {@9, @11, @8, @7} = vertsplit(@22) */
  w9 = w22[0];
  w11 = w22[1];
  w8 = w22[2];
  w7 = w22[3];
  /* #115: @19 = zeros(11x1) */
  casadi_clear(w19, 11);
  /* #116: @1 = zeros(1x3) */
  casadi_clear(w1, 3);
  /* #117: @6 = @10[0] */
  for (rr=(&w6), ss=(&w10)+0; ss!=(&w10)+1; ss+=1) *rr++ = *ss;
  /* #118: @6 = (-@6) */
  w6 = (- w6 );
  /* #119: @10 = @6' */
  casadi_copy((&w6), 1, (&w10));
  /* #120: @1 = mac(@10,@21,@1) */
  casadi_mtimes((&w10), casadi_s5, w21, casadi_s4, w1, casadi_s3, w, 0);
  /* #121: @1 = @1' */
  /* #122: @20 = zeros(1x3) */
  casadi_clear(w20, 3);
  /* #123: @6 = @6' */
  /* #124: @20 = mac(@6,@27,@20) */
  casadi_mtimes((&w6), casadi_s5, w27, casadi_s4, w20, casadi_s3, w, 0);
  /* #125: @20 = @20' */
  /* #126: @1 = (@1+@20) */
  for (i=0, rr=w1, cs=w20; i<3; ++i) (*rr++) += (*cs++);
  /* #127: @1 = (-@1) */
  for (i=0, rr=w1, cs=w1; i<3; ++i) *rr++ = (- *cs++ );
  /* #128: (@19[:3] += @1) */
  for (rr=w19+0, ss=w1; rr!=w19+3; rr+=1) *rr += *ss++;
  /* #129: {@6, @10, @5, @0, @2, @3, @4, @12, @13, @14, @15} = vertsplit(@19) */
  w6 = w19[0];
  w10 = w19[1];
  w5 = w19[2];
  w0 = w19[3];
  w2 = w19[4];
  w3 = w19[5];
  w4 = w19[6];
  w12 = w19[7];
  w13 = w19[8];
  w14 = w19[9];
  w15 = w19[10];
  /* #130: @17 = vertcat(@9, @11, @8, @7, @6, @10, @5, @0, @2, @3, @4, @12, @13, @14, @15) */
  rr=w17;
  *rr++ = w9;
  *rr++ = w11;
  *rr++ = w8;
  *rr++ = w7;
  *rr++ = w6;
  *rr++ = w10;
  *rr++ = w5;
  *rr++ = w0;
  *rr++ = w2;
  *rr++ = w3;
  *rr++ = w4;
  *rr++ = w12;
  *rr++ = w13;
  *rr++ = w14;
  *rr++ = w15;
  /* #131: @33 = @17[:7] */
  for (rr=w33, ss=w17+0; ss!=w17+7; ss+=1) *rr++ = *ss;
  /* #132: (@28[0, 1, 2, 3, 16, 17, 18] = @33) */
  for (cii=casadi_s6, rr=w28, ss=w33; cii!=casadi_s6+7; ++cii, ++ss) if (*cii>=0) rr[*cii] = *ss;
  /* #133: @33 = @17[:7] */
  for (rr=w33, ss=w17+0; ss!=w17+7; ss+=1) *rr++ = *ss;
  /* #134: (@28[0, 4, 8, 12, 16, 19, 22] = @33) */
  for (cii=casadi_s7, rr=w28, ss=w33; cii!=casadi_s7+7; ++cii, ++ss) if (*cii>=0) rr[*cii] = *ss;
  /* #135: @22 = zeros(1x4) */
  casadi_clear(w22, 4);
  /* #136: @30 = 00 */
  /* #137: @34 = ones(15x1,2nz) */
  casadi_fill(w34, 2, 1.);
  /* #138: {NULL, @9, NULL, NULL, NULL, @11, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL} = vertsplit(@34) */
  w9 = w34[0];
  w11 = w34[1];
  /* #139: @31 = 00 */
  /* #140: @32 = 00 */
  /* #141: @8 = vertcat(@30, @9, @31, @32) */
  rr=(&w8);
  *rr++ = w9;
  /* #142: @9 = @8' */
  casadi_copy((&w8), 1, (&w9));
  /* #143: @22 = mac(@9,@25,@22) */
  casadi_mtimes((&w9), casadi_s8, w25, casadi_s1, w22, casadi_s0, w, 0);
  /* #144: @22 = @22' */
  /* #145: @24 = zeros(1x4) */
  casadi_clear(w24, 4);
  /* #146: @8 = @8' */
  /* #147: @24 = mac(@8,@26,@24) */
  casadi_mtimes((&w8), casadi_s8, w26, casadi_s1, w24, casadi_s0, w, 0);
  /* #148: @24 = @24' */
  /* #149: @22 = (@22+@24) */
  for (i=0, rr=w22, cs=w24; i<4; ++i) (*rr++) += (*cs++);
  /* #150: {@8, @9, @7, @6} = vertsplit(@22) */
  w8 = w22[0];
  w9 = w22[1];
  w7 = w22[2];
  w6 = w22[3];
  /* #151: @19 = zeros(11x1) */
  casadi_clear(w19, 11);
  /* #152: @1 = zeros(1x3) */
  casadi_clear(w1, 3);
  /* #153: @10 = @11[0] */
  for (rr=(&w10), ss=(&w11)+0; ss!=(&w11)+1; ss+=1) *rr++ = *ss;
  /* #154: @10 = (-@10) */
  w10 = (- w10 );
  /* #155: @11 = @10' */
  casadi_copy((&w10), 1, (&w11));
  /* #156: @1 = mac(@11,@21,@1) */
  casadi_mtimes((&w11), casadi_s9, w21, casadi_s4, w1, casadi_s3, w, 0);
  /* #157: @1 = @1' */
  /* #158: @20 = zeros(1x3) */
  casadi_clear(w20, 3);
  /* #159: @10 = @10' */
  /* #160: @20 = mac(@10,@27,@20) */
  casadi_mtimes((&w10), casadi_s9, w27, casadi_s4, w20, casadi_s3, w, 0);
  /* #161: @20 = @20' */
  /* #162: @1 = (@1+@20) */
  for (i=0, rr=w1, cs=w20; i<3; ++i) (*rr++) += (*cs++);
  /* #163: @1 = (-@1) */
  for (i=0, rr=w1, cs=w1; i<3; ++i) *rr++ = (- *cs++ );
  /* #164: (@19[:3] += @1) */
  for (rr=w19+0, ss=w1; rr!=w19+3; rr+=1) *rr += *ss++;
  /* #165: {@10, @11, @5, @0, @2, @3, @4, @12, @13, @14, @15} = vertsplit(@19) */
  w10 = w19[0];
  w11 = w19[1];
  w5 = w19[2];
  w0 = w19[3];
  w2 = w19[4];
  w3 = w19[5];
  w4 = w19[6];
  w12 = w19[7];
  w13 = w19[8];
  w14 = w19[9];
  w15 = w19[10];
  /* #166: @17 = vertcat(@8, @9, @7, @6, @10, @11, @5, @0, @2, @3, @4, @12, @13, @14, @15) */
  rr=w17;
  *rr++ = w8;
  *rr++ = w9;
  *rr++ = w7;
  *rr++ = w6;
  *rr++ = w10;
  *rr++ = w11;
  *rr++ = w5;
  *rr++ = w0;
  *rr++ = w2;
  *rr++ = w3;
  *rr++ = w4;
  *rr++ = w12;
  *rr++ = w13;
  *rr++ = w14;
  *rr++ = w15;
  /* #167: @33 = @17[:7] */
  for (rr=w33, ss=w17+0; ss!=w17+7; ss+=1) *rr++ = *ss;
  /* #168: (@28[4, 5, 6, 7, 19, 20, 21] = @33) */
  for (cii=casadi_s10, rr=w28, ss=w33; cii!=casadi_s10+7; ++cii, ++ss) if (*cii>=0) rr[*cii] = *ss;
  /* #169: @33 = @17[:7] */
  for (rr=w33, ss=w17+0; ss!=w17+7; ss+=1) *rr++ = *ss;
  /* #170: (@28[1, 5, 9, 13, 17, 20, 23] = @33) */
  for (cii=casadi_s11, rr=w28, ss=w33; cii!=casadi_s11+7; ++cii, ++ss) if (*cii>=0) rr[*cii] = *ss;
  /* #171: @22 = zeros(1x4) */
  casadi_clear(w22, 4);
  /* #172: @30 = 00 */
  /* #173: @31 = 00 */
  /* #174: @34 = ones(15x1,2nz) */
  casadi_fill(w34, 2, 1.);
  /* #175: {NULL, NULL, @8, NULL, NULL, NULL, @9, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL} = vertsplit(@34) */
  w8 = w34[0];
  w9 = w34[1];
  /* #176: @32 = 00 */
  /* #177: @7 = vertcat(@30, @31, @8, @32) */
  rr=(&w7);
  *rr++ = w8;
  /* #178: @8 = @7' */
  casadi_copy((&w7), 1, (&w8));
  /* #179: @22 = mac(@8,@25,@22) */
  casadi_mtimes((&w8), casadi_s12, w25, casadi_s1, w22, casadi_s0, w, 0);
  /* #180: @22 = @22' */
  /* #181: @24 = zeros(1x4) */
  casadi_clear(w24, 4);
  /* #182: @7 = @7' */
  /* #183: @24 = mac(@7,@26,@24) */
  casadi_mtimes((&w7), casadi_s12, w26, casadi_s1, w24, casadi_s0, w, 0);
  /* #184: @24 = @24' */
  /* #185: @22 = (@22+@24) */
  for (i=0, rr=w22, cs=w24; i<4; ++i) (*rr++) += (*cs++);
  /* #186: {@7, @8, @6, @10} = vertsplit(@22) */
  w7 = w22[0];
  w8 = w22[1];
  w6 = w22[2];
  w10 = w22[3];
  /* #187: @19 = zeros(11x1) */
  casadi_clear(w19, 11);
  /* #188: @1 = zeros(1x3) */
  casadi_clear(w1, 3);
  /* #189: @11 = @9[0] */
  for (rr=(&w11), ss=(&w9)+0; ss!=(&w9)+1; ss+=1) *rr++ = *ss;
  /* #190: @11 = (-@11) */
  w11 = (- w11 );
  /* #191: @9 = @11' */
  casadi_copy((&w11), 1, (&w9));
  /* #192: @1 = mac(@9,@21,@1) */
  casadi_mtimes((&w9), casadi_s13, w21, casadi_s4, w1, casadi_s3, w, 0);
  /* #193: @1 = @1' */
  /* #194: @20 = zeros(1x3) */
  casadi_clear(w20, 3);
  /* #195: @11 = @11' */
  /* #196: @20 = mac(@11,@27,@20) */
  casadi_mtimes((&w11), casadi_s13, w27, casadi_s4, w20, casadi_s3, w, 0);
  /* #197: @20 = @20' */
  /* #198: @1 = (@1+@20) */
  for (i=0, rr=w1, cs=w20; i<3; ++i) (*rr++) += (*cs++);
  /* #199: @1 = (-@1) */
  for (i=0, rr=w1, cs=w1; i<3; ++i) *rr++ = (- *cs++ );
  /* #200: (@19[:3] += @1) */
  for (rr=w19+0, ss=w1; rr!=w19+3; rr+=1) *rr += *ss++;
  /* #201: {@11, @9, @5, @0, @2, @3, @4, @12, @13, @14, @15} = vertsplit(@19) */
  w11 = w19[0];
  w9 = w19[1];
  w5 = w19[2];
  w0 = w19[3];
  w2 = w19[4];
  w3 = w19[5];
  w4 = w19[6];
  w12 = w19[7];
  w13 = w19[8];
  w14 = w19[9];
  w15 = w19[10];
  /* #202: @17 = vertcat(@7, @8, @6, @10, @11, @9, @5, @0, @2, @3, @4, @12, @13, @14, @15) */
  rr=w17;
  *rr++ = w7;
  *rr++ = w8;
  *rr++ = w6;
  *rr++ = w10;
  *rr++ = w11;
  *rr++ = w9;
  *rr++ = w5;
  *rr++ = w0;
  *rr++ = w2;
  *rr++ = w3;
  *rr++ = w4;
  *rr++ = w12;
  *rr++ = w13;
  *rr++ = w14;
  *rr++ = w15;
  /* #203: @33 = @17[:7] */
  for (rr=w33, ss=w17+0; ss!=w17+7; ss+=1) *rr++ = *ss;
  /* #204: (@28[8, 9, 10, 11, 22, 23, 24] = @33) */
  for (cii=casadi_s14, rr=w28, ss=w33; cii!=casadi_s14+7; ++cii, ++ss) if (*cii>=0) rr[*cii] = *ss;
  /* #205: @33 = @17[:7] */
  for (rr=w33, ss=w17+0; ss!=w17+7; ss+=1) *rr++ = *ss;
  /* #206: (@28[2, 6, 10, 14, 18, 21, 24] = @33) */
  for (cii=casadi_s15, rr=w28, ss=w33; cii!=casadi_s15+7; ++cii, ++ss) if (*cii>=0) rr[*cii] = *ss;
  /* #207: @22 = zeros(1x4) */
  casadi_clear(w22, 4);
  /* #208: @30 = 00 */
  /* #209: @31 = 00 */
  /* #210: @32 = 00 */
  /* #211: @7 = ones(15x1,1nz) */
  w7 = 1.;
  /* #212: {NULL, NULL, NULL, @8, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL} = vertsplit(@7) */
  w8 = w7;
  /* #213: @7 = vertcat(@30, @31, @32, @8) */
  rr=(&w7);
  *rr++ = w8;
  /* #214: @8 = @7' */
  casadi_copy((&w7), 1, (&w8));
  /* #215: @22 = mac(@8,@25,@22) */
  casadi_mtimes((&w8), casadi_s16, w25, casadi_s1, w22, casadi_s0, w, 0);
  /* #216: @22 = @22' */
  /* #217: @24 = zeros(1x4) */
  casadi_clear(w24, 4);
  /* #218: @7 = @7' */
  /* #219: @24 = mac(@7,@26,@24) */
  casadi_mtimes((&w7), casadi_s16, w26, casadi_s1, w24, casadi_s0, w, 0);
  /* #220: @24 = @24' */
  /* #221: @22 = (@22+@24) */
  for (i=0, rr=w22, cs=w24; i<4; ++i) (*rr++) += (*cs++);
  /* #222: {@7, @8, @6, @10} = vertsplit(@22) */
  w7 = w22[0];
  w8 = w22[1];
  w6 = w22[2];
  w10 = w22[3];
  /* #223: @30 = 00 */
  /* #224: @31 = 00 */
  /* #225: @32 = 00 */
  /* #226: @35 = 00 */
  /* #227: @36 = 00 */
  /* #228: @37 = 00 */
  /* #229: @38 = 00 */
  /* #230: @39 = 00 */
  /* #231: @40 = 00 */
  /* #232: @41 = 00 */
  /* #233: @42 = 00 */
  /* #234: @22 = vertcat(@7, @8, @6, @10, @30, @31, @32, @35, @36, @37, @38, @39, @40, @41, @42) */
  rr=w22;
  *rr++ = w7;
  *rr++ = w8;
  *rr++ = w6;
  *rr++ = w10;
  /* #235: @24 = @22[:4] */
  for (rr=w24, ss=w22+0; ss!=w22+4; ss+=1) *rr++ = *ss;
  /* #236: (@28[12:16] = @24) */
  for (rr=w28+12, ss=w24; rr!=w28+16; rr+=1) *rr = *ss++;
  /* #237: @24 = @22[:4] */
  for (rr=w24, ss=w22+0; ss!=w22+4; ss+=1) *rr++ = *ss;
  /* #238: (@28[3:19:4] = @24) */
  for (rr=w28+3, ss=w24; rr!=w28+19; rr+=4) *rr = *ss++;
  /* #239: @43 = @28' */
  casadi_trans(w28,casadi_s17, w43, casadi_s17, iw);
  /* #240: output[2][0] = @43 */
  casadi_copy(w43, 25, res[2]);
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
    case 0: return casadi_s18;
    case 1: return casadi_s19;
    case 2: return casadi_s20;
    case 3: return casadi_s21;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* Drone_ode_cost_ext_cost_0_fun_jac_hess_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s22;
    case 1: return casadi_s21;
    case 2: return casadi_s17;
    case 3: return casadi_s20;
    case 4: return casadi_s23;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int Drone_ode_cost_ext_cost_0_fun_jac_hess_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 19;
  if (sz_res) *sz_res = 20;
  if (sz_iw) *sz_iw = 16;
  if (sz_w) *sz_w = 183;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif

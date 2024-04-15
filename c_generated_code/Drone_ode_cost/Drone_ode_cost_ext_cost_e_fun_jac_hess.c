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
  #define CASADI_PREFIX(ID) Drone_ode_cost_ext_cost_e_fun_jac_hess_ ## ID
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
#define casadi_dot CASADI_PREFIX(dot)
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

casadi_real casadi_dot(casadi_int n, const casadi_real* x, const casadi_real* y) {
  casadi_int i;
  casadi_real r = 0;
  for (i=0; i<n; ++i) r += *x++ * *y++;
  return r;
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

static const casadi_int casadi_s0[9] = {1, 3, 0, 1, 2, 3, 0, 0, 0};
static const casadi_int casadi_s1[15] = {3, 3, 0, 3, 6, 9, 0, 1, 2, 0, 1, 2, 0, 1, 2};
static const casadi_int casadi_s2[7] = {1, 3, 0, 1, 1, 1, 0};
static const casadi_int casadi_s3[7] = {0, 1, 2, 9, 10, 11, 12};
static const casadi_int casadi_s4[7] = {0, 3, 6, 9, 13, 17, 21};
static const casadi_int casadi_s5[7] = {1, 3, 0, 0, 1, 1, 0};
static const casadi_int casadi_s6[7] = {3, 4, 5, 13, 14, 15, 16};
static const casadi_int casadi_s7[7] = {1, 4, 7, 10, 14, 18, 22};
static const casadi_int casadi_s8[7] = {1, 3, 0, 0, 0, 1, 0};
static const casadi_int casadi_s9[7] = {6, 7, 8, 17, 18, 19, 20};
static const casadi_int casadi_s10[7] = {2, 5, 8, 11, 15, 19, 23};
static const casadi_int casadi_s11[39] = {11, 11, 0, 3, 6, 9, 13, 17, 21, 25, 25, 25, 25, 25, 0, 1, 2, 0, 1, 2, 0, 1, 2, 3, 4, 5, 6, 3, 4, 5, 6, 3, 4, 5, 6, 3, 4, 5, 6};
static const casadi_int casadi_s12[15] = {11, 1, 0, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
static const casadi_int casadi_s13[3] = {0, 0, 0};
static const casadi_int casadi_s14[19] = {15, 1, 0, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14};
static const casadi_int casadi_s15[5] = {1, 1, 0, 1, 0};
static const casadi_int casadi_s16[14] = {0, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

/* Drone_ode_cost_ext_cost_e_fun_jac_hess:(i0[11],i1[],i2[],i3[15])->(o0,o1[11],o2[11x11,25nz],o3[],o4[0x11]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_int i, j, k;
  casadi_real *rr, *ss, *tt;
  const casadi_int *cii;
  const casadi_real *cs;
  casadi_real w0, *w1=w+2, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15, w16, *w17=w+20, *w18=w+35, *w19=w+38, *w20=w+49, *w21=w+52, *w22=w+61, *w23=w+65, *w24=w+69, *w25=w+72, *w26=w+75, w27, w28, w29, *w30=w+87, *w31=w+112, w32, w33, w34, *w35=w+121, *w36=w+128, *w44=w+130;
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
  /* #22: @9 = input[0][3] */
  w9 = arg[0] ? arg[0][3] : 0;
  /* #23: @10 = input[0][4] */
  w10 = arg[0] ? arg[0][4] : 0;
  /* #24: @11 = input[0][5] */
  w11 = arg[0] ? arg[0][5] : 0;
  /* #25: @12 = input[0][6] */
  w12 = arg[0] ? arg[0][6] : 0;
  /* #26: @13 = input[0][7] */
  w13 = arg[0] ? arg[0][7] : 0;
  /* #27: @14 = input[0][8] */
  w14 = arg[0] ? arg[0][8] : 0;
  /* #28: @15 = input[0][9] */
  w15 = arg[0] ? arg[0][9] : 0;
  /* #29: @16 = input[0][10] */
  w16 = arg[0] ? arg[0][10] : 0;
  /* #30: @19 = vertcat(@2, @3, @4, @9, @10, @11, @12, @13, @14, @15, @16) */
  rr=w19;
  *rr++ = w2;
  *rr++ = w3;
  *rr++ = w4;
  *rr++ = w9;
  *rr++ = w10;
  *rr++ = w11;
  *rr++ = w12;
  *rr++ = w13;
  *rr++ = w14;
  *rr++ = w15;
  *rr++ = w16;
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
  /* #43: @2 = 1 */
  w2 = 1.;
  /* #44: @6 = (-@6) */
  w6 = (- w6 );
  /* #45: @7 = (-@7) */
  w7 = (- w7 );
  /* #46: @8 = (-@8) */
  w8 = (- w8 );
  /* #47: @22 = vertcat(@5, @6, @7, @8) */
  rr=w22;
  *rr++ = w5;
  *rr++ = w6;
  *rr++ = w7;
  *rr++ = w8;
  /* #48: @23 = @17[3:7] */
  for (rr=w23, ss=w17+3; ss!=w17+7; ss+=1) *rr++ = *ss;
  /* #49: @5 = ||@23||_F */
  w5 = sqrt(casadi_dot(4, w23, w23));
  /* #50: @22 = (@22/@5) */
  for (i=0, rr=w22; i<4; ++i) (*rr++) /= w5;
  /* #51: @5 = @22[0] */
  for (rr=(&w5), ss=w22+0; ss!=w22+1; ss+=1) *rr++ = *ss;
  /* #52: @6 = (@5*@9) */
  w6  = (w5*w9);
  /* #53: @7 = @22[1] */
  for (rr=(&w7), ss=w22+1; ss!=w22+2; ss+=1) *rr++ = *ss;
  /* #54: @8 = (@7*@10) */
  w8  = (w7*w10);
  /* #55: @6 = (@6-@8) */
  w6 -= w8;
  /* #56: @8 = @22[2] */
  for (rr=(&w8), ss=w22+2; ss!=w22+3; ss+=1) *rr++ = *ss;
  /* #57: @3 = (@8*@11) */
  w3  = (w8*w11);
  /* #58: @6 = (@6-@3) */
  w6 -= w3;
  /* #59: @3 = @22[3] */
  for (rr=(&w3), ss=w22+3; ss!=w22+4; ss+=1) *rr++ = *ss;
  /* #60: @4 = (@3*@12) */
  w4  = (w3*w12);
  /* #61: @6 = (@6-@4) */
  w6 -= w4;
  /* #62: @2 = (@2-@6) */
  w2 -= w6;
  /* #63: @0 = (@0+@2) */
  w0 += w2;
  /* #64: @2 = 0 */
  w2 = 0.;
  /* #65: @4 = 7 */
  w4 = 7.;
  /* #66: @13 = (@5*@10) */
  w13  = (w5*w10);
  /* #67: @14 = (@7*@9) */
  w14  = (w7*w9);
  /* #68: @13 = (@13+@14) */
  w13 += w14;
  /* #69: @14 = (@8*@12) */
  w14  = (w8*w12);
  /* #70: @13 = (@13+@14) */
  w13 += w14;
  /* #71: @14 = (@3*@11) */
  w14  = (w3*w11);
  /* #72: @13 = (@13-@14) */
  w13 -= w14;
  /* #73: @14 = (@5*@11) */
  w14  = (w5*w11);
  /* #74: @15 = (@7*@12) */
  w15  = (w7*w12);
  /* #75: @14 = (@14-@15) */
  w14 -= w15;
  /* #76: @15 = (@8*@9) */
  w15  = (w8*w9);
  /* #77: @14 = (@14+@15) */
  w14 += w15;
  /* #78: @15 = (@3*@10) */
  w15  = (w3*w10);
  /* #79: @14 = (@14+@15) */
  w14 += w15;
  /* #80: @12 = (@5*@12) */
  w12  = (w5*w12);
  /* #81: @11 = (@7*@11) */
  w11  = (w7*w11);
  /* #82: @12 = (@12+@11) */
  w12 += w11;
  /* #83: @10 = (@8*@10) */
  w10  = (w8*w10);
  /* #84: @12 = (@12-@10) */
  w12 -= w10;
  /* #85: @9 = (@3*@9) */
  w9  = (w3*w9);
  /* #86: @12 = (@12+@9) */
  w12 += w9;
  /* #87: @22 = vertcat(@6, @13, @14, @12) */
  rr=w22;
  *rr++ = w6;
  *rr++ = w13;
  *rr++ = w14;
  *rr++ = w12;
  /* #88: @20 = @22[1:4] */
  for (rr=w20, ss=w22+1; ss!=w22+4; ss+=1) *rr++ = *ss;
  /* #89: @20 = @20' */
  /* #90: @20 = (@4*@20) */
  for (i=0, rr=w20, cs=w20; i<3; ++i) (*rr++)  = (w4*(*cs++));
  /* #91: @24 = @22[1:4] */
  for (rr=w24, ss=w22+1; ss!=w22+4; ss+=1) *rr++ = *ss;
  /* #92: @2 = mac(@20,@24,@2) */
  for (i=0, rr=(&w2); i<1; ++i) for (j=0; j<1; ++j, ++rr) for (k=0, ss=w20+j, tt=w24+i*3; k<3; ++k) *rr += ss[k*1]**tt++;
  /* #93: @0 = (@0+@2) */
  w0 += w2;
  /* #94: output[0][0] = @0 */
  if (res[0]) res[0][0] = w0;
  /* #95: @19 = zeros(11x1) */
  casadi_clear(w19, 11);
  /* #96: @1 = @1' */
  /* #97: @25 = zeros(1x3) */
  casadi_clear(w25, 3);
  /* #98: @18 = @18' */
  /* #99: @26 = @21' */
  for (i=0, rr=w26, cs=w21; i<3; ++i) for (j=0; j<3; ++j) rr[i+j*3] = *cs++;
  /* #100: @25 = mac(@18,@26,@25) */
  for (i=0, rr=w25; i<3; ++i) for (j=0; j<1; ++j, ++rr) for (k=0, ss=w18+j, tt=w26+i*3; k<3; ++k) *rr += ss[k*1]**tt++;
  /* #101: @25 = @25' */
  /* #102: @1 = (@1+@25) */
  for (i=0, rr=w1, cs=w25; i<3; ++i) (*rr++) += (*cs++);
  /* #103: @1 = (-@1) */
  for (i=0, rr=w1, cs=w1; i<3; ++i) *rr++ = (- *cs++ );
  /* #104: (@19[:3] += @1) */
  for (rr=w19+0, ss=w1; rr!=w19+3; rr+=1) *rr += *ss++;
  /* #105: {@0, @2, @6, @13, @14, @12, @9, @10, @11, @15, @16} = vertsplit(@19) */
  w0 = w19[0];
  w2 = w19[1];
  w6 = w19[2];
  w13 = w19[3];
  w14 = w19[4];
  w12 = w19[5];
  w9 = w19[6];
  w10 = w19[7];
  w11 = w19[8];
  w15 = w19[9];
  w16 = w19[10];
  /* #106: output[1][0] = @0 */
  if (res[1]) res[1][0] = w0;
  /* #107: output[1][1] = @2 */
  if (res[1]) res[1][1] = w2;
  /* #108: output[1][2] = @6 */
  if (res[1]) res[1][2] = w6;
  /* #109: @22 = zeros(4x1) */
  casadi_clear(w22, 4);
  /* #110: @20 = @20' */
  /* #111: (@22[1:4] += @20) */
  for (rr=w22+1, ss=w20; rr!=w22+4; rr+=1) *rr += *ss++;
  /* #112: @24 = @24' */
  /* #113: @24 = (@4*@24) */
  for (i=0, rr=w24, cs=w24; i<3; ++i) (*rr++)  = (w4*(*cs++));
  /* #114: @24 = @24' */
  /* #115: (@22[1:4] += @24) */
  for (rr=w22+1, ss=w24; rr!=w22+4; rr+=1) *rr += *ss++;
  /* #116: {@6, @2, @0, @27} = vertsplit(@22) */
  w6 = w22[0];
  w2 = w22[1];
  w0 = w22[2];
  w27 = w22[3];
  /* #117: @28 = (@3*@27) */
  w28  = (w3*w27);
  /* #118: @29 = (@8*@0) */
  w29  = (w8*w0);
  /* #119: @28 = (@28+@29) */
  w28 += w29;
  /* #120: @29 = (@7*@2) */
  w29  = (w7*w2);
  /* #121: @28 = (@28+@29) */
  w28 += w29;
  /* #122: @29 = -1 */
  w29 = -1.;
  /* #123: @29 = (@29+@6) */
  w29 += w6;
  /* #124: @6 = (@5*@29) */
  w6  = (w5*w29);
  /* #125: @28 = (@28+@6) */
  w28 += w6;
  /* #126: @28 = (@28+@13) */
  w28 += w13;
  /* #127: output[1][3] = @28 */
  if (res[1]) res[1][3] = w28;
  /* #128: @28 = (@3*@0) */
  w28  = (w3*w0);
  /* #129: @13 = (@8*@27) */
  w13  = (w8*w27);
  /* #130: @28 = (@28-@13) */
  w28 -= w13;
  /* #131: @13 = (@5*@2) */
  w13  = (w5*w2);
  /* #132: @28 = (@28+@13) */
  w28 += w13;
  /* #133: @13 = (@7*@29) */
  w13  = (w7*w29);
  /* #134: @28 = (@28-@13) */
  w28 -= w13;
  /* #135: @28 = (@28+@14) */
  w28 += w14;
  /* #136: output[1][4] = @28 */
  if (res[1]) res[1][4] = w28;
  /* #137: @28 = (@7*@27) */
  w28  = (w7*w27);
  /* #138: @14 = (@5*@0) */
  w14  = (w5*w0);
  /* #139: @28 = (@28+@14) */
  w28 += w14;
  /* #140: @14 = (@3*@2) */
  w14  = (w3*w2);
  /* #141: @28 = (@28-@14) */
  w28 -= w14;
  /* #142: @14 = (@8*@29) */
  w14  = (w8*w29);
  /* #143: @28 = (@28-@14) */
  w28 -= w14;
  /* #144: @28 = (@28+@12) */
  w28 += w12;
  /* #145: output[1][5] = @28 */
  if (res[1]) res[1][5] = w28;
  /* #146: @27 = (@5*@27) */
  w27  = (w5*w27);
  /* #147: @0 = (@7*@0) */
  w0  = (w7*w0);
  /* #148: @27 = (@27-@0) */
  w27 -= w0;
  /* #149: @2 = (@8*@2) */
  w2  = (w8*w2);
  /* #150: @27 = (@27+@2) */
  w27 += w2;
  /* #151: @29 = (@3*@29) */
  w29  = (w3*w29);
  /* #152: @27 = (@27-@29) */
  w27 -= w29;
  /* #153: @27 = (@27+@9) */
  w27 += w9;
  /* #154: output[1][6] = @27 */
  if (res[1]) res[1][6] = w27;
  /* #155: output[1][7] = @10 */
  if (res[1]) res[1][7] = w10;
  /* #156: output[1][8] = @11 */
  if (res[1]) res[1][8] = w11;
  /* #157: output[1][9] = @15 */
  if (res[1]) res[1][9] = w15;
  /* #158: output[1][10] = @16 */
  if (res[1]) res[1][10] = w16;
  /* #159: @30 = zeros(11x11,25nz) */
  casadi_clear(w30, 25);
  /* #160: @19 = zeros(11x1) */
  casadi_clear(w19, 11);
  /* #161: @24 = zeros(1x3) */
  casadi_clear(w24, 3);
  /* #162: @31 = ones(11x1,6nz) */
  casadi_fill(w31, 6, 1.);
  /* #163: {@16, NULL, NULL, @15, NULL, NULL, NULL, NULL, NULL, NULL, NULL} = vertsplit(@31) */
  w16 = w31[0];
  w15 = w31[1];
  /* #164: @11 = @16[0] */
  for (rr=(&w11), ss=(&w16)+0; ss!=(&w16)+1; ss+=1) *rr++ = *ss;
  /* #165: @11 = (-@11) */
  w11 = (- w11 );
  /* #166: @16 = @11' */
  casadi_copy((&w11), 1, (&w16));
  /* #167: @24 = mac(@16,@21,@24) */
  casadi_mtimes((&w16), casadi_s2, w21, casadi_s1, w24, casadi_s0, w, 0);
  /* #168: @24 = @24' */
  /* #169: @20 = zeros(1x3) */
  casadi_clear(w20, 3);
  /* #170: @11 = @11' */
  /* #171: @20 = mac(@11,@26,@20) */
  casadi_mtimes((&w11), casadi_s2, w26, casadi_s1, w20, casadi_s0, w, 0);
  /* #172: @20 = @20' */
  /* #173: @24 = (@24+@20) */
  for (i=0, rr=w24, cs=w20; i<3; ++i) (*rr++) += (*cs++);
  /* #174: @24 = (-@24) */
  for (i=0, rr=w24, cs=w24; i<3; ++i) *rr++ = (- *cs++ );
  /* #175: (@19[:3] += @24) */
  for (rr=w19+0, ss=w24; rr!=w19+3; rr+=1) *rr += *ss++;
  /* #176: {@11, @16, @10, @27, @9, @29, @2, @0, @28, @12, @14} = vertsplit(@19) */
  w11 = w19[0];
  w16 = w19[1];
  w10 = w19[2];
  w27 = w19[3];
  w9 = w19[4];
  w29 = w19[5];
  w2 = w19[6];
  w0 = w19[7];
  w28 = w19[8];
  w12 = w19[9];
  w14 = w19[10];
  /* #177: @22 = zeros(4x1) */
  casadi_clear(w22, 4);
  /* #178: @13 = (@5*@15) */
  w13  = (w5*w15);
  /* #179: @6 = (@7*@15) */
  w6  = (w7*w15);
  /* #180: @32 = (@8*@15) */
  w32  = (w8*w15);
  /* #181: @15 = (@3*@15) */
  w15  = (w3*w15);
  /* #182: @23 = vertcat(@13, @6, @32, @15) */
  rr=w23;
  *rr++ = w13;
  *rr++ = w6;
  *rr++ = w32;
  *rr++ = w15;
  /* #183: @24 = @23[1:4] */
  for (rr=w24, ss=w23+1; ss!=w23+4; ss+=1) *rr++ = *ss;
  /* #184: @24 = @24' */
  /* #185: @24 = (@4*@24) */
  for (i=0, rr=w24, cs=w24; i<3; ++i) (*rr++)  = (w4*(*cs++));
  /* #186: @24 = @24' */
  /* #187: (@22[1:4] += @24) */
  for (rr=w22+1, ss=w24; rr!=w22+4; rr+=1) *rr += *ss++;
  /* #188: @24 = @23[1:4] */
  for (rr=w24, ss=w23+1; ss!=w23+4; ss+=1) *rr++ = *ss;
  /* #189: @24 = @24' */
  /* #190: @24 = (@4*@24) */
  for (i=0, rr=w24, cs=w24; i<3; ++i) (*rr++)  = (w4*(*cs++));
  /* #191: @24 = @24' */
  /* #192: (@22[1:4] += @24) */
  for (rr=w22+1, ss=w24; rr!=w22+4; rr+=1) *rr += *ss++;
  /* #193: {@13, @6, @32, @15} = vertsplit(@22) */
  w13 = w22[0];
  w6 = w22[1];
  w32 = w22[2];
  w15 = w22[3];
  /* #194: @33 = (@3*@15) */
  w33  = (w3*w15);
  /* #195: @34 = (@8*@32) */
  w34  = (w8*w32);
  /* #196: @33 = (@33+@34) */
  w33 += w34;
  /* #197: @34 = (@7*@6) */
  w34  = (w7*w6);
  /* #198: @33 = (@33+@34) */
  w33 += w34;
  /* #199: @34 = (@5*@13) */
  w34  = (w5*w13);
  /* #200: @33 = (@33+@34) */
  w33 += w34;
  /* #201: @33 = (@33+@27) */
  w33 += w27;
  /* #202: @27 = (@3*@32) */
  w27  = (w3*w32);
  /* #203: @34 = (@8*@15) */
  w34  = (w8*w15);
  /* #204: @27 = (@27-@34) */
  w27 -= w34;
  /* #205: @34 = (@5*@6) */
  w34  = (w5*w6);
  /* #206: @27 = (@27+@34) */
  w27 += w34;
  /* #207: @34 = (@7*@13) */
  w34  = (w7*w13);
  /* #208: @27 = (@27-@34) */
  w27 -= w34;
  /* #209: @27 = (@27+@9) */
  w27 += w9;
  /* #210: @9 = (@7*@15) */
  w9  = (w7*w15);
  /* #211: @34 = (@5*@32) */
  w34  = (w5*w32);
  /* #212: @9 = (@9+@34) */
  w9 += w34;
  /* #213: @34 = (@3*@6) */
  w34  = (w3*w6);
  /* #214: @9 = (@9-@34) */
  w9 -= w34;
  /* #215: @34 = (@8*@13) */
  w34  = (w8*w13);
  /* #216: @9 = (@9-@34) */
  w9 -= w34;
  /* #217: @9 = (@9+@29) */
  w9 += w29;
  /* #218: @15 = (@5*@15) */
  w15  = (w5*w15);
  /* #219: @32 = (@7*@32) */
  w32  = (w7*w32);
  /* #220: @15 = (@15-@32) */
  w15 -= w32;
  /* #221: @6 = (@8*@6) */
  w6  = (w8*w6);
  /* #222: @15 = (@15+@6) */
  w15 += w6;
  /* #223: @13 = (@3*@13) */
  w13  = (w3*w13);
  /* #224: @15 = (@15-@13) */
  w15 -= w13;
  /* #225: @15 = (@15+@2) */
  w15 += w2;
  /* #226: @19 = vertcat(@11, @16, @10, @33, @27, @9, @15, @0, @28, @12, @14) */
  rr=w19;
  *rr++ = w11;
  *rr++ = w16;
  *rr++ = w10;
  *rr++ = w33;
  *rr++ = w27;
  *rr++ = w9;
  *rr++ = w15;
  *rr++ = w0;
  *rr++ = w28;
  *rr++ = w12;
  *rr++ = w14;
  /* #227: @35 = @19[:7] */
  for (rr=w35, ss=w19+0; ss!=w19+7; ss+=1) *rr++ = *ss;
  /* #228: (@30[0, 1, 2, 9, 10, 11, 12] = @35) */
  for (cii=casadi_s3, rr=w30, ss=w35; cii!=casadi_s3+7; ++cii, ++ss) if (*cii>=0) rr[*cii] = *ss;
  /* #229: @35 = @19[:7] */
  for (rr=w35, ss=w19+0; ss!=w19+7; ss+=1) *rr++ = *ss;
  /* #230: (@30[0, 3, 6, 9, 13, 17, 21] = @35) */
  for (cii=casadi_s4, rr=w30, ss=w35; cii!=casadi_s4+7; ++cii, ++ss) if (*cii>=0) rr[*cii] = *ss;
  /* #231: @19 = zeros(11x1) */
  casadi_clear(w19, 11);
  /* #232: @24 = zeros(1x3) */
  casadi_clear(w24, 3);
  /* #233: @36 = ones(11x1,2nz) */
  casadi_fill(w36, 2, 1.);
  /* #234: {NULL, @11, NULL, NULL, @16, NULL, NULL, NULL, NULL, NULL, NULL} = vertsplit(@36) */
  w11 = w36[0];
  w16 = w36[1];
  /* #235: @10 = @11[0] */
  for (rr=(&w10), ss=(&w11)+0; ss!=(&w11)+1; ss+=1) *rr++ = *ss;
  /* #236: @10 = (-@10) */
  w10 = (- w10 );
  /* #237: @11 = @10' */
  casadi_copy((&w10), 1, (&w11));
  /* #238: @24 = mac(@11,@21,@24) */
  casadi_mtimes((&w11), casadi_s5, w21, casadi_s1, w24, casadi_s0, w, 0);
  /* #239: @24 = @24' */
  /* #240: @20 = zeros(1x3) */
  casadi_clear(w20, 3);
  /* #241: @10 = @10' */
  /* #242: @20 = mac(@10,@26,@20) */
  casadi_mtimes((&w10), casadi_s5, w26, casadi_s1, w20, casadi_s0, w, 0);
  /* #243: @20 = @20' */
  /* #244: @24 = (@24+@20) */
  for (i=0, rr=w24, cs=w20; i<3; ++i) (*rr++) += (*cs++);
  /* #245: @24 = (-@24) */
  for (i=0, rr=w24, cs=w24; i<3; ++i) *rr++ = (- *cs++ );
  /* #246: (@19[:3] += @24) */
  for (rr=w19+0, ss=w24; rr!=w19+3; rr+=1) *rr += *ss++;
  /* #247: {@10, @11, @33, @27, @9, @15, @0, @28, @12, @14, @2} = vertsplit(@19) */
  w10 = w19[0];
  w11 = w19[1];
  w33 = w19[2];
  w27 = w19[3];
  w9 = w19[4];
  w15 = w19[5];
  w0 = w19[6];
  w28 = w19[7];
  w12 = w19[8];
  w14 = w19[9];
  w2 = w19[10];
  /* #248: @22 = zeros(4x1) */
  casadi_clear(w22, 4);
  /* #249: @13 = (@7*@16) */
  w13  = (w7*w16);
  /* #250: @13 = (-@13) */
  w13 = (- w13 );
  /* #251: @6 = (@5*@16) */
  w6  = (w5*w16);
  /* #252: @32 = (@3*@16) */
  w32  = (w3*w16);
  /* #253: @16 = (@8*@16) */
  w16  = (w8*w16);
  /* #254: @16 = (-@16) */
  w16 = (- w16 );
  /* #255: @23 = vertcat(@13, @6, @32, @16) */
  rr=w23;
  *rr++ = w13;
  *rr++ = w6;
  *rr++ = w32;
  *rr++ = w16;
  /* #256: @24 = @23[1:4] */
  for (rr=w24, ss=w23+1; ss!=w23+4; ss+=1) *rr++ = *ss;
  /* #257: @24 = @24' */
  /* #258: @24 = (@4*@24) */
  for (i=0, rr=w24, cs=w24; i<3; ++i) (*rr++)  = (w4*(*cs++));
  /* #259: @24 = @24' */
  /* #260: (@22[1:4] += @24) */
  for (rr=w22+1, ss=w24; rr!=w22+4; rr+=1) *rr += *ss++;
  /* #261: @24 = @23[1:4] */
  for (rr=w24, ss=w23+1; ss!=w23+4; ss+=1) *rr++ = *ss;
  /* #262: @24 = @24' */
  /* #263: @24 = (@4*@24) */
  for (i=0, rr=w24, cs=w24; i<3; ++i) (*rr++)  = (w4*(*cs++));
  /* #264: @24 = @24' */
  /* #265: (@22[1:4] += @24) */
  for (rr=w22+1, ss=w24; rr!=w22+4; rr+=1) *rr += *ss++;
  /* #266: {@13, @6, @32, @16} = vertsplit(@22) */
  w13 = w22[0];
  w6 = w22[1];
  w32 = w22[2];
  w16 = w22[3];
  /* #267: @29 = (@3*@16) */
  w29  = (w3*w16);
  /* #268: @34 = (@8*@32) */
  w34  = (w8*w32);
  /* #269: @29 = (@29+@34) */
  w29 += w34;
  /* #270: @34 = (@7*@6) */
  w34  = (w7*w6);
  /* #271: @29 = (@29+@34) */
  w29 += w34;
  /* #272: @34 = (@5*@13) */
  w34  = (w5*w13);
  /* #273: @29 = (@29+@34) */
  w29 += w34;
  /* #274: @29 = (@29+@27) */
  w29 += w27;
  /* #275: @27 = (@3*@32) */
  w27  = (w3*w32);
  /* #276: @34 = (@8*@16) */
  w34  = (w8*w16);
  /* #277: @27 = (@27-@34) */
  w27 -= w34;
  /* #278: @34 = (@5*@6) */
  w34  = (w5*w6);
  /* #279: @27 = (@27+@34) */
  w27 += w34;
  /* #280: @34 = (@7*@13) */
  w34  = (w7*w13);
  /* #281: @27 = (@27-@34) */
  w27 -= w34;
  /* #282: @27 = (@27+@9) */
  w27 += w9;
  /* #283: @9 = (@7*@16) */
  w9  = (w7*w16);
  /* #284: @34 = (@5*@32) */
  w34  = (w5*w32);
  /* #285: @9 = (@9+@34) */
  w9 += w34;
  /* #286: @34 = (@3*@6) */
  w34  = (w3*w6);
  /* #287: @9 = (@9-@34) */
  w9 -= w34;
  /* #288: @34 = (@8*@13) */
  w34  = (w8*w13);
  /* #289: @9 = (@9-@34) */
  w9 -= w34;
  /* #290: @9 = (@9+@15) */
  w9 += w15;
  /* #291: @16 = (@5*@16) */
  w16  = (w5*w16);
  /* #292: @32 = (@7*@32) */
  w32  = (w7*w32);
  /* #293: @16 = (@16-@32) */
  w16 -= w32;
  /* #294: @6 = (@8*@6) */
  w6  = (w8*w6);
  /* #295: @16 = (@16+@6) */
  w16 += w6;
  /* #296: @13 = (@3*@13) */
  w13  = (w3*w13);
  /* #297: @16 = (@16-@13) */
  w16 -= w13;
  /* #298: @16 = (@16+@0) */
  w16 += w0;
  /* #299: @19 = vertcat(@10, @11, @33, @29, @27, @9, @16, @28, @12, @14, @2) */
  rr=w19;
  *rr++ = w10;
  *rr++ = w11;
  *rr++ = w33;
  *rr++ = w29;
  *rr++ = w27;
  *rr++ = w9;
  *rr++ = w16;
  *rr++ = w28;
  *rr++ = w12;
  *rr++ = w14;
  *rr++ = w2;
  /* #300: @35 = @19[:7] */
  for (rr=w35, ss=w19+0; ss!=w19+7; ss+=1) *rr++ = *ss;
  /* #301: (@30[3, 4, 5, 13, 14, 15, 16] = @35) */
  for (cii=casadi_s6, rr=w30, ss=w35; cii!=casadi_s6+7; ++cii, ++ss) if (*cii>=0) rr[*cii] = *ss;
  /* #302: @35 = @19[:7] */
  for (rr=w35, ss=w19+0; ss!=w19+7; ss+=1) *rr++ = *ss;
  /* #303: (@30[1, 4, 7, 10, 14, 18, 22] = @35) */
  for (cii=casadi_s7, rr=w30, ss=w35; cii!=casadi_s7+7; ++cii, ++ss) if (*cii>=0) rr[*cii] = *ss;
  /* #304: @19 = zeros(11x1) */
  casadi_clear(w19, 11);
  /* #305: @24 = zeros(1x3) */
  casadi_clear(w24, 3);
  /* #306: @36 = ones(11x1,2nz) */
  casadi_fill(w36, 2, 1.);
  /* #307: {NULL, NULL, @10, NULL, NULL, @11, NULL, NULL, NULL, NULL, NULL} = vertsplit(@36) */
  w10 = w36[0];
  w11 = w36[1];
  /* #308: @33 = @10[0] */
  for (rr=(&w33), ss=(&w10)+0; ss!=(&w10)+1; ss+=1) *rr++ = *ss;
  /* #309: @33 = (-@33) */
  w33 = (- w33 );
  /* #310: @10 = @33' */
  casadi_copy((&w33), 1, (&w10));
  /* #311: @24 = mac(@10,@21,@24) */
  casadi_mtimes((&w10), casadi_s8, w21, casadi_s1, w24, casadi_s0, w, 0);
  /* #312: @24 = @24' */
  /* #313: @20 = zeros(1x3) */
  casadi_clear(w20, 3);
  /* #314: @33 = @33' */
  /* #315: @20 = mac(@33,@26,@20) */
  casadi_mtimes((&w33), casadi_s8, w26, casadi_s1, w20, casadi_s0, w, 0);
  /* #316: @20 = @20' */
  /* #317: @24 = (@24+@20) */
  for (i=0, rr=w24, cs=w20; i<3; ++i) (*rr++) += (*cs++);
  /* #318: @24 = (-@24) */
  for (i=0, rr=w24, cs=w24; i<3; ++i) *rr++ = (- *cs++ );
  /* #319: (@19[:3] += @24) */
  for (rr=w19+0, ss=w24; rr!=w19+3; rr+=1) *rr += *ss++;
  /* #320: {@33, @10, @29, @27, @9, @16, @28, @12, @14, @2, @0} = vertsplit(@19) */
  w33 = w19[0];
  w10 = w19[1];
  w29 = w19[2];
  w27 = w19[3];
  w9 = w19[4];
  w16 = w19[5];
  w28 = w19[6];
  w12 = w19[7];
  w14 = w19[8];
  w2 = w19[9];
  w0 = w19[10];
  /* #321: @22 = zeros(4x1) */
  casadi_clear(w22, 4);
  /* #322: @13 = (@8*@11) */
  w13  = (w8*w11);
  /* #323: @13 = (-@13) */
  w13 = (- w13 );
  /* #324: @6 = (@3*@11) */
  w6  = (w3*w11);
  /* #325: @6 = (-@6) */
  w6 = (- w6 );
  /* #326: @32 = (@5*@11) */
  w32  = (w5*w11);
  /* #327: @11 = (@7*@11) */
  w11  = (w7*w11);
  /* #328: @23 = vertcat(@13, @6, @32, @11) */
  rr=w23;
  *rr++ = w13;
  *rr++ = w6;
  *rr++ = w32;
  *rr++ = w11;
  /* #329: @24 = @23[1:4] */
  for (rr=w24, ss=w23+1; ss!=w23+4; ss+=1) *rr++ = *ss;
  /* #330: @24 = @24' */
  /* #331: @24 = (@4*@24) */
  for (i=0, rr=w24, cs=w24; i<3; ++i) (*rr++)  = (w4*(*cs++));
  /* #332: @24 = @24' */
  /* #333: (@22[1:4] += @24) */
  for (rr=w22+1, ss=w24; rr!=w22+4; rr+=1) *rr += *ss++;
  /* #334: @24 = @23[1:4] */
  for (rr=w24, ss=w23+1; ss!=w23+4; ss+=1) *rr++ = *ss;
  /* #335: @24 = @24' */
  /* #336: @24 = (@4*@24) */
  for (i=0, rr=w24, cs=w24; i<3; ++i) (*rr++)  = (w4*(*cs++));
  /* #337: @24 = @24' */
  /* #338: (@22[1:4] += @24) */
  for (rr=w22+1, ss=w24; rr!=w22+4; rr+=1) *rr += *ss++;
  /* #339: {@13, @6, @32, @11} = vertsplit(@22) */
  w13 = w22[0];
  w6 = w22[1];
  w32 = w22[2];
  w11 = w22[3];
  /* #340: @15 = (@3*@11) */
  w15  = (w3*w11);
  /* #341: @34 = (@8*@32) */
  w34  = (w8*w32);
  /* #342: @15 = (@15+@34) */
  w15 += w34;
  /* #343: @34 = (@7*@6) */
  w34  = (w7*w6);
  /* #344: @15 = (@15+@34) */
  w15 += w34;
  /* #345: @34 = (@5*@13) */
  w34  = (w5*w13);
  /* #346: @15 = (@15+@34) */
  w15 += w34;
  /* #347: @15 = (@15+@27) */
  w15 += w27;
  /* #348: @27 = (@3*@32) */
  w27  = (w3*w32);
  /* #349: @34 = (@8*@11) */
  w34  = (w8*w11);
  /* #350: @27 = (@27-@34) */
  w27 -= w34;
  /* #351: @34 = (@5*@6) */
  w34  = (w5*w6);
  /* #352: @27 = (@27+@34) */
  w27 += w34;
  /* #353: @34 = (@7*@13) */
  w34  = (w7*w13);
  /* #354: @27 = (@27-@34) */
  w27 -= w34;
  /* #355: @27 = (@27+@9) */
  w27 += w9;
  /* #356: @9 = (@7*@11) */
  w9  = (w7*w11);
  /* #357: @34 = (@5*@32) */
  w34  = (w5*w32);
  /* #358: @9 = (@9+@34) */
  w9 += w34;
  /* #359: @34 = (@3*@6) */
  w34  = (w3*w6);
  /* #360: @9 = (@9-@34) */
  w9 -= w34;
  /* #361: @34 = (@8*@13) */
  w34  = (w8*w13);
  /* #362: @9 = (@9-@34) */
  w9 -= w34;
  /* #363: @9 = (@9+@16) */
  w9 += w16;
  /* #364: @11 = (@5*@11) */
  w11  = (w5*w11);
  /* #365: @32 = (@7*@32) */
  w32  = (w7*w32);
  /* #366: @11 = (@11-@32) */
  w11 -= w32;
  /* #367: @6 = (@8*@6) */
  w6  = (w8*w6);
  /* #368: @11 = (@11+@6) */
  w11 += w6;
  /* #369: @13 = (@3*@13) */
  w13  = (w3*w13);
  /* #370: @11 = (@11-@13) */
  w11 -= w13;
  /* #371: @11 = (@11+@28) */
  w11 += w28;
  /* #372: @19 = vertcat(@33, @10, @29, @15, @27, @9, @11, @12, @14, @2, @0) */
  rr=w19;
  *rr++ = w33;
  *rr++ = w10;
  *rr++ = w29;
  *rr++ = w15;
  *rr++ = w27;
  *rr++ = w9;
  *rr++ = w11;
  *rr++ = w12;
  *rr++ = w14;
  *rr++ = w2;
  *rr++ = w0;
  /* #373: @35 = @19[:7] */
  for (rr=w35, ss=w19+0; ss!=w19+7; ss+=1) *rr++ = *ss;
  /* #374: (@30[6, 7, 8, 17, 18, 19, 20] = @35) */
  for (cii=casadi_s9, rr=w30, ss=w35; cii!=casadi_s9+7; ++cii, ++ss) if (*cii>=0) rr[*cii] = *ss;
  /* #375: @35 = @19[:7] */
  for (rr=w35, ss=w19+0; ss!=w19+7; ss+=1) *rr++ = *ss;
  /* #376: (@30[2, 5, 8, 11, 15, 19, 23] = @35) */
  for (cii=casadi_s10, rr=w30, ss=w35; cii!=casadi_s10+7; ++cii, ++ss) if (*cii>=0) rr[*cii] = *ss;
  /* #377: @37 = 00 */
  /* #378: @38 = 00 */
  /* #379: @39 = 00 */
  /* #380: @22 = zeros(4x1) */
  casadi_clear(w22, 4);
  /* #381: @33 = ones(11x1,1nz) */
  w33 = 1.;
  /* #382: {NULL, NULL, NULL, NULL, NULL, NULL, @10, NULL, NULL, NULL, NULL} = vertsplit(@33) */
  w10 = w33;
  /* #383: @33 = (@3*@10) */
  w33  = (w3*w10);
  /* #384: @33 = (-@33) */
  w33 = (- w33 );
  /* #385: @29 = (@8*@10) */
  w29  = (w8*w10);
  /* #386: @15 = (@7*@10) */
  w15  = (w7*w10);
  /* #387: @15 = (-@15) */
  w15 = (- w15 );
  /* #388: @10 = (@5*@10) */
  w10  = (w5*w10);
  /* #389: @23 = vertcat(@33, @29, @15, @10) */
  rr=w23;
  *rr++ = w33;
  *rr++ = w29;
  *rr++ = w15;
  *rr++ = w10;
  /* #390: @24 = @23[1:4] */
  for (rr=w24, ss=w23+1; ss!=w23+4; ss+=1) *rr++ = *ss;
  /* #391: @24 = @24' */
  /* #392: @24 = (@4*@24) */
  for (i=0, rr=w24, cs=w24; i<3; ++i) (*rr++)  = (w4*(*cs++));
  /* #393: @24 = @24' */
  /* #394: (@22[1:4] += @24) */
  for (rr=w22+1, ss=w24; rr!=w22+4; rr+=1) *rr += *ss++;
  /* #395: @24 = @23[1:4] */
  for (rr=w24, ss=w23+1; ss!=w23+4; ss+=1) *rr++ = *ss;
  /* #396: @24 = @24' */
  /* #397: @24 = (@4*@24) */
  for (i=0, rr=w24, cs=w24; i<3; ++i) (*rr++)  = (w4*(*cs++));
  /* #398: @24 = @24' */
  /* #399: (@22[1:4] += @24) */
  for (rr=w22+1, ss=w24; rr!=w22+4; rr+=1) *rr += *ss++;
  /* #400: {@4, @33, @29, @15} = vertsplit(@22) */
  w4 = w22[0];
  w33 = w22[1];
  w29 = w22[2];
  w15 = w22[3];
  /* #401: @10 = (@3*@15) */
  w10  = (w3*w15);
  /* #402: @27 = (@8*@29) */
  w27  = (w8*w29);
  /* #403: @10 = (@10+@27) */
  w10 += w27;
  /* #404: @27 = (@7*@33) */
  w27  = (w7*w33);
  /* #405: @10 = (@10+@27) */
  w10 += w27;
  /* #406: @27 = (@5*@4) */
  w27  = (w5*w4);
  /* #407: @10 = (@10+@27) */
  w10 += w27;
  /* #408: @27 = (@3*@29) */
  w27  = (w3*w29);
  /* #409: @9 = (@8*@15) */
  w9  = (w8*w15);
  /* #410: @27 = (@27-@9) */
  w27 -= w9;
  /* #411: @9 = (@5*@33) */
  w9  = (w5*w33);
  /* #412: @27 = (@27+@9) */
  w27 += w9;
  /* #413: @9 = (@7*@4) */
  w9  = (w7*w4);
  /* #414: @27 = (@27-@9) */
  w27 -= w9;
  /* #415: @9 = (@7*@15) */
  w9  = (w7*w15);
  /* #416: @11 = (@5*@29) */
  w11  = (w5*w29);
  /* #417: @9 = (@9+@11) */
  w9 += w11;
  /* #418: @11 = (@3*@33) */
  w11  = (w3*w33);
  /* #419: @9 = (@9-@11) */
  w9 -= w11;
  /* #420: @11 = (@8*@4) */
  w11  = (w8*w4);
  /* #421: @9 = (@9-@11) */
  w9 -= w11;
  /* #422: @5 = (@5*@15) */
  w5 *= w15;
  /* #423: @7 = (@7*@29) */
  w7 *= w29;
  /* #424: @5 = (@5-@7) */
  w5 -= w7;
  /* #425: @8 = (@8*@33) */
  w8 *= w33;
  /* #426: @5 = (@5+@8) */
  w5 += w8;
  /* #427: @3 = (@3*@4) */
  w3 *= w4;
  /* #428: @5 = (@5-@3) */
  w5 -= w3;
  /* #429: @40 = 00 */
  /* #430: @41 = 00 */
  /* #431: @42 = 00 */
  /* #432: @43 = 00 */
  /* #433: @22 = vertcat(@37, @38, @39, @10, @27, @9, @5, @40, @41, @42, @43) */
  rr=w22;
  *rr++ = w10;
  *rr++ = w27;
  *rr++ = w9;
  *rr++ = w5;
  /* #434: @23 = @22[:4] */
  for (rr=w23, ss=w22+0; ss!=w22+4; ss+=1) *rr++ = *ss;
  /* #435: (@30[21:25] = @23) */
  for (rr=w30+21, ss=w23; rr!=w30+25; rr+=1) *rr = *ss++;
  /* #436: @23 = @22[:4] */
  for (rr=w23, ss=w22+0; ss!=w22+4; ss+=1) *rr++ = *ss;
  /* #437: (@30[12:28:4] = @23) */
  for (rr=w30+12, ss=w23; rr!=w30+28; rr+=4) *rr = *ss++;
  /* #438: @44 = @30' */
  casadi_trans(w30,casadi_s11, w44, casadi_s11, iw);
  /* #439: output[2][0] = @44 */
  casadi_copy(w44, 25, res[2]);
  return 0;
}

CASADI_SYMBOL_EXPORT int Drone_ode_cost_ext_cost_e_fun_jac_hess(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int Drone_ode_cost_ext_cost_e_fun_jac_hess_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int Drone_ode_cost_ext_cost_e_fun_jac_hess_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void Drone_ode_cost_ext_cost_e_fun_jac_hess_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int Drone_ode_cost_ext_cost_e_fun_jac_hess_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void Drone_ode_cost_ext_cost_e_fun_jac_hess_release(int mem) {
}

CASADI_SYMBOL_EXPORT void Drone_ode_cost_ext_cost_e_fun_jac_hess_incref(void) {
}

CASADI_SYMBOL_EXPORT void Drone_ode_cost_ext_cost_e_fun_jac_hess_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int Drone_ode_cost_ext_cost_e_fun_jac_hess_n_in(void) { return 4;}

CASADI_SYMBOL_EXPORT casadi_int Drone_ode_cost_ext_cost_e_fun_jac_hess_n_out(void) { return 5;}

CASADI_SYMBOL_EXPORT casadi_real Drone_ode_cost_ext_cost_e_fun_jac_hess_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* Drone_ode_cost_ext_cost_e_fun_jac_hess_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* Drone_ode_cost_ext_cost_e_fun_jac_hess_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    case 1: return "o1";
    case 2: return "o2";
    case 3: return "o3";
    case 4: return "o4";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* Drone_ode_cost_ext_cost_e_fun_jac_hess_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s12;
    case 1: return casadi_s13;
    case 2: return casadi_s13;
    case 3: return casadi_s14;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* Drone_ode_cost_ext_cost_e_fun_jac_hess_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s15;
    case 1: return casadi_s12;
    case 2: return casadi_s11;
    case 3: return casadi_s13;
    case 4: return casadi_s16;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int Drone_ode_cost_ext_cost_e_fun_jac_hess_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 19;
  if (sz_res) *sz_res = 16;
  if (sz_iw) *sz_iw = 12;
  if (sz_w) *sz_w = 155;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif

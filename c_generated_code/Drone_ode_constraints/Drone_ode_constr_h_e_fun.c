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
  #define CASADI_PREFIX(ID) Drone_ode_constr_h_e_fun_ ## ID
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
#define casadi_s0 CASADI_PREFIX(s0)
#define casadi_s1 CASADI_PREFIX(s1)
#define casadi_s2 CASADI_PREFIX(s2)
#define casadi_s3 CASADI_PREFIX(s3)

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

static const casadi_int casadi_s0[12] = {8, 1, 0, 8, 0, 1, 2, 3, 4, 5, 6, 7};
static const casadi_int casadi_s1[3] = {0, 0, 0};
static const casadi_int casadi_s2[16] = {12, 1, 0, 12, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
static const casadi_int casadi_s3[5] = {1, 1, 0, 1, 0};

/* Drone_ode_constr_h_e_fun:(i0[8],i1[],i2[],i3[12])->(o0) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_int i, j, k;
  casadi_real *rr, *ss, *tt;
  const casadi_real *cs;
  casadi_real w0, *w1=w+5, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, *w14=w+21, *w15=w+33, *w16=w+37, *w17=w+45, *w18=w+49, *w19=w+65, *w20=w+81;
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
  /* #28: @17 = (-@17) */
  for (i=0, rr=w17, cs=w17; i<4; ++i) *rr++ = (- *cs++ );
  /* #29: @18 = zeros(4x4) */
  casadi_clear(w18, 16);
  /* #30: @2 = 2 */
  w2 = 2.;
  /* #31: (@18[0] = @2) */
  for (rr=w18+0, ss=(&w2); rr!=w18+1; rr+=1) *rr = *ss++;
  /* #32: @2 = 2 */
  w2 = 2.;
  /* #33: (@18[5] = @2) */
  for (rr=w18+5, ss=(&w2); rr!=w18+6; rr+=1) *rr = *ss++;
  /* #34: @2 = 2 */
  w2 = 2.;
  /* #35: (@18[10] = @2) */
  for (rr=w18+10, ss=(&w2); rr!=w18+11; rr+=1) *rr = *ss++;
  /* #36: @2 = 2 */
  w2 = 2.;
  /* #37: (@18[15] = @2) */
  for (rr=w18+15, ss=(&w2); rr!=w18+16; rr+=1) *rr = *ss++;
  /* #38: @1 = mac(@17,@18,@1) */
  for (i=0, rr=w1; i<4; ++i) for (j=0; j<1; ++j, ++rr) for (k=0, ss=w17+j, tt=w18+i*4; k<4; ++k) *rr += ss[k*1]**tt++;
  /* #39: @17 = zeros(4x1) */
  casadi_clear(w17, 4);
  /* #40: @19 = zeros(4x4) */
  casadi_clear(w19, 16);
  /* #41: @2 = cos(@5) */
  w2 = cos( w5 );
  /* #42: (@19[0] = @2) */
  for (rr=w19+0, ss=(&w2); rr!=w19+1; rr+=1) *rr = *ss++;
  /* #43: @2 = sin(@5) */
  w2 = sin( w5 );
  /* #44: @2 = (-@2) */
  w2 = (- w2 );
  /* #45: (@19[4] = @2) */
  for (rr=w19+4, ss=(&w2); rr!=w19+5; rr+=1) *rr = *ss++;
  /* #46: @2 = 0 */
  w2 = 0.;
  /* #47: (@19[8] = @2) */
  for (rr=w19+8, ss=(&w2); rr!=w19+9; rr+=1) *rr = *ss++;
  /* #48: @2 = 0 */
  w2 = 0.;
  /* #49: (@19[12] = @2) */
  for (rr=w19+12, ss=(&w2); rr!=w19+13; rr+=1) *rr = *ss++;
  /* #50: @2 = sin(@5) */
  w2 = sin( w5 );
  /* #51: (@19[1] = @2) */
  for (rr=w19+1, ss=(&w2); rr!=w19+2; rr+=1) *rr = *ss++;
  /* #52: @5 = cos(@5) */
  w5 = cos( w5 );
  /* #53: (@19[5] = @5) */
  for (rr=w19+5, ss=(&w5); rr!=w19+6; rr+=1) *rr = *ss++;
  /* #54: @5 = 0 */
  w5 = 0.;
  /* #55: (@19[9] = @5) */
  for (rr=w19+9, ss=(&w5); rr!=w19+10; rr+=1) *rr = *ss++;
  /* #56: @5 = 0 */
  w5 = 0.;
  /* #57: (@19[13] = @5) */
  for (rr=w19+13, ss=(&w5); rr!=w19+14; rr+=1) *rr = *ss++;
  /* #58: @5 = 0 */
  w5 = 0.;
  /* #59: (@19[2] = @5) */
  for (rr=w19+2, ss=(&w5); rr!=w19+3; rr+=1) *rr = *ss++;
  /* #60: @5 = 0 */
  w5 = 0.;
  /* #61: (@19[6] = @5) */
  for (rr=w19+6, ss=(&w5); rr!=w19+7; rr+=1) *rr = *ss++;
  /* #62: @5 = 1 */
  w5 = 1.;
  /* #63: (@19[10] = @5) */
  for (rr=w19+10, ss=(&w5); rr!=w19+11; rr+=1) *rr = *ss++;
  /* #64: @5 = 0 */
  w5 = 0.;
  /* #65: (@19[14] = @5) */
  for (rr=w19+14, ss=(&w5); rr!=w19+15; rr+=1) *rr = *ss++;
  /* #66: @5 = 0 */
  w5 = 0.;
  /* #67: (@19[3] = @5) */
  for (rr=w19+3, ss=(&w5); rr!=w19+4; rr+=1) *rr = *ss++;
  /* #68: @5 = 0 */
  w5 = 0.;
  /* #69: (@19[7] = @5) */
  for (rr=w19+7, ss=(&w5); rr!=w19+8; rr+=1) *rr = *ss++;
  /* #70: @5 = 0 */
  w5 = 0.;
  /* #71: (@19[11] = @5) */
  for (rr=w19+11, ss=(&w5); rr!=w19+12; rr+=1) *rr = *ss++;
  /* #72: @5 = 1 */
  w5 = 1.;
  /* #73: (@19[15] = @5) */
  for (rr=w19+15, ss=(&w5); rr!=w19+16; rr+=1) *rr = *ss++;
  /* #74: @20 = @16[4:8] */
  for (rr=w20, ss=w16+4; ss!=w16+8; ss+=1) *rr++ = *ss;
  /* #75: @17 = mac(@19,@20,@17) */
  for (i=0, rr=w17; i<1; ++i) for (j=0; j<4; ++j, ++rr) for (k=0, ss=w19+j, tt=w20+i*4; k<4; ++k) *rr += ss[k*4]**tt++;
  /* #76: @0 = mac(@1,@17,@0) */
  for (i=0, rr=(&w0); i<1; ++i) for (j=0; j<1; ++j, ++rr) for (k=0, ss=w1+j, tt=w17+i*4; k<4; ++k) *rr += ss[k*1]**tt++;
  /* #77: @5 = 0.1 */
  w5 = 1.0000000000000001e-01;
  /* #78: @2 = 0 */
  w2 = 0.;
  /* #79: @1 = zeros(1x4) */
  casadi_clear(w1, 4);
  /* #80: @3 = 0.5 */
  w3 = 5.0000000000000000e-01;
  /* #81: @17 = @15' */
  casadi_copy(w15, 4, w17);
  /* #82: @17 = (@3*@17) */
  for (i=0, rr=w17, cs=w17; i<4; ++i) (*rr++)  = (w3*(*cs++));
  /* #83: @1 = mac(@17,@18,@1) */
  for (i=0, rr=w1; i<4; ++i) for (j=0; j<1; ++j, ++rr) for (k=0, ss=w17+j, tt=w18+i*4; k<4; ++k) *rr += ss[k*1]**tt++;
  /* #84: @2 = mac(@1,@15,@2) */
  for (i=0, rr=(&w2); i<1; ++i) for (j=0; j<1; ++j, ++rr) for (k=0, ss=w1+j, tt=w15+i*4; k<4; ++k) *rr += ss[k*1]**tt++;
  /* #85: @5 = (@5*@2) */
  w5 *= w2;
  /* #86: @0 = (@0+@5) */
  w0 += w5;
  /* #87: output[0][0] = @0 */
  if (res[0]) res[0][0] = w0;
  return 0;
}

CASADI_SYMBOL_EXPORT int Drone_ode_constr_h_e_fun(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int Drone_ode_constr_h_e_fun_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int Drone_ode_constr_h_e_fun_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void Drone_ode_constr_h_e_fun_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int Drone_ode_constr_h_e_fun_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void Drone_ode_constr_h_e_fun_release(int mem) {
}

CASADI_SYMBOL_EXPORT void Drone_ode_constr_h_e_fun_incref(void) {
}

CASADI_SYMBOL_EXPORT void Drone_ode_constr_h_e_fun_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int Drone_ode_constr_h_e_fun_n_in(void) { return 4;}

CASADI_SYMBOL_EXPORT casadi_int Drone_ode_constr_h_e_fun_n_out(void) { return 1;}

CASADI_SYMBOL_EXPORT casadi_real Drone_ode_constr_h_e_fun_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* Drone_ode_constr_h_e_fun_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* Drone_ode_constr_h_e_fun_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* Drone_ode_constr_h_e_fun_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    case 2: return casadi_s1;
    case 3: return casadi_s2;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* Drone_ode_constr_h_e_fun_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s3;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int Drone_ode_constr_h_e_fun_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 16;
  if (sz_res) *sz_res = 2;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 85;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif

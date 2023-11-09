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
  #define CASADI_PREFIX(ID) Drone_ode_cost_ext_cost_e_fun_ ## ID
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

static const casadi_int casadi_s0[15] = {11, 1, 0, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
static const casadi_int casadi_s1[3] = {0, 0, 0};
static const casadi_int casadi_s2[19] = {15, 1, 0, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14};
static const casadi_int casadi_s3[5] = {1, 1, 0, 1, 0};

/* Drone_ode_cost_ext_cost_e_fun:(i0[11],i1[],i2[],i3[15])->(o0) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_int i, j, k;
  casadi_real *rr, *ss, *tt;
  const casadi_real *cs;
  casadi_real w0, *w1=w+2, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15, w16, *w17=w+20, *w18=w+35, *w19=w+38, *w20=w+49, *w21=w+52;
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
  /* #43: output[0][0] = @0 */
  if (res[0]) res[0][0] = w0;
  return 0;
}

CASADI_SYMBOL_EXPORT int Drone_ode_cost_ext_cost_e_fun(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int Drone_ode_cost_ext_cost_e_fun_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int Drone_ode_cost_ext_cost_e_fun_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void Drone_ode_cost_ext_cost_e_fun_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int Drone_ode_cost_ext_cost_e_fun_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void Drone_ode_cost_ext_cost_e_fun_release(int mem) {
}

CASADI_SYMBOL_EXPORT void Drone_ode_cost_ext_cost_e_fun_incref(void) {
}

CASADI_SYMBOL_EXPORT void Drone_ode_cost_ext_cost_e_fun_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int Drone_ode_cost_ext_cost_e_fun_n_in(void) { return 4;}

CASADI_SYMBOL_EXPORT casadi_int Drone_ode_cost_ext_cost_e_fun_n_out(void) { return 1;}

CASADI_SYMBOL_EXPORT casadi_real Drone_ode_cost_ext_cost_e_fun_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* Drone_ode_cost_ext_cost_e_fun_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* Drone_ode_cost_ext_cost_e_fun_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* Drone_ode_cost_ext_cost_e_fun_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    case 2: return casadi_s1;
    case 3: return casadi_s2;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* Drone_ode_cost_ext_cost_e_fun_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s3;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int Drone_ode_cost_ext_cost_e_fun_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 19;
  if (sz_res) *sz_res = 2;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 61;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif

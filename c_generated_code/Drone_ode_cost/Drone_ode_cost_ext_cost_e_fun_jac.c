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
  #define CASADI_PREFIX(ID) Drone_ode_cost_ext_cost_e_fun_jac_ ## ID
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
static const casadi_int casadi_s2[5] = {1, 1, 0, 1, 0};

/* Drone_ode_cost_ext_cost_e_fun_jac:(i0[8],i1[],i2[],i3[8])->(o0,o1[8]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_int i, j, k;
  casadi_real *rr, *ss, *tt;
  const casadi_real *cs;
  casadi_real w0, *w1=w+2, w2, w3, w4, w5, w6, w7, w8, w9, *w10=w+14, *w11=w+22, *w12=w+26, *w13=w+30, *w14=w+46;
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
  /* #35: output[0][0] = @0 */
  if (res[0]) res[0][0] = w0;
  /* #36: @10 = zeros(8x1) */
  casadi_clear(w10, 8);
  /* #37: @1 = @1' */
  /* #38: @12 = zeros(1x4) */
  casadi_clear(w12, 4);
  /* #39: @11 = @11' */
  /* #40: @14 = @13' */
  for (i=0, rr=w14, cs=w13; i<4; ++i) for (j=0; j<4; ++j) rr[i+j*4] = *cs++;
  /* #41: @12 = mac(@11,@14,@12) */
  for (i=0, rr=w12; i<4; ++i) for (j=0; j<1; ++j, ++rr) for (k=0, ss=w11+j, tt=w14+i*4; k<4; ++k) *rr += ss[k*1]**tt++;
  /* #42: @12 = @12' */
  /* #43: @1 = (@1+@12) */
  for (i=0, rr=w1, cs=w12; i<4; ++i) (*rr++) += (*cs++);
  /* #44: @1 = (-@1) */
  for (i=0, rr=w1, cs=w1; i<4; ++i) *rr++ = (- *cs++ );
  /* #45: (@10[:4] += @1) */
  for (rr=w10+0, ss=w1; rr!=w10+4; rr+=1) *rr += *ss++;
  /* #46: output[1][0] = @10 */
  casadi_copy(w10, 8, res[1]);
  return 0;
}

CASADI_SYMBOL_EXPORT int Drone_ode_cost_ext_cost_e_fun_jac(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int Drone_ode_cost_ext_cost_e_fun_jac_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int Drone_ode_cost_ext_cost_e_fun_jac_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void Drone_ode_cost_ext_cost_e_fun_jac_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int Drone_ode_cost_ext_cost_e_fun_jac_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void Drone_ode_cost_ext_cost_e_fun_jac_release(int mem) {
}

CASADI_SYMBOL_EXPORT void Drone_ode_cost_ext_cost_e_fun_jac_incref(void) {
}

CASADI_SYMBOL_EXPORT void Drone_ode_cost_ext_cost_e_fun_jac_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int Drone_ode_cost_ext_cost_e_fun_jac_n_in(void) { return 4;}

CASADI_SYMBOL_EXPORT casadi_int Drone_ode_cost_ext_cost_e_fun_jac_n_out(void) { return 2;}

CASADI_SYMBOL_EXPORT casadi_real Drone_ode_cost_ext_cost_e_fun_jac_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* Drone_ode_cost_ext_cost_e_fun_jac_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* Drone_ode_cost_ext_cost_e_fun_jac_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    case 1: return "o1";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* Drone_ode_cost_ext_cost_e_fun_jac_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    case 2: return casadi_s1;
    case 3: return casadi_s0;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* Drone_ode_cost_ext_cost_e_fun_jac_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s2;
    case 1: return casadi_s0;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int Drone_ode_cost_ext_cost_e_fun_jac_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 12;
  if (sz_res) *sz_res = 3;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 62;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif

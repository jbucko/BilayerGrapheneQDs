#include <complex.h>
#include <stdio.h>
#include "arb-2.17.0/arbcmath.h"


double * hyper_u(double a_re,double a_im, double b_re, double b_im, double z_re, double z_im)
{

static double s[2];
complex double a,b,z;
a = a_re+a_im*I;
b = b_re+b_im*I;
z = z_re+z_im*I;

complex double c = ac_hyperu(a,b,z);
s[0] = creal(c);
s[1] = cimag(c);
return s;
}

double * hyper_m(double a_re,double a_im, double b_re, double b_im, double z_re, double z_im)
{

static double s[2];
complex double a,b,z;
a = a_re+a_im*I;
b = b_re+b_im*I;
z = z_re+z_im*I;

complex double c = ac_hyp1f1(a,b,z);
s[0] = creal(c);
s[1] = cimag(c);
return s;
}



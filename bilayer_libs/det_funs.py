import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import null_space
from scipy.special import hyperu as hpu
from scipy.optimize import root,brentq
from mpmath import hyp1f1,hyperu,gamma 
import mpmath as mm
from mpmath import *
from numpy.linalg import norm
from matplotlib import cm
import matplotlib
from mpl_toolkits.axes_grid1 import AxesGrid
import time,sys
import ctypes
from scipy.interpolate import interp2d

"""
calling C function for calculations of confluent hypergeometric functions
"""
# try:
confluent = ctypes.CDLL('/Users/jozef/Desktop/master_thesis/cm-bilayerboundstates/HamiltonianModel/my_libs/confluent.so')
# except:
	# print('file confluent.so not found')

confluent.hyper_m.argtypes = [ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_double]
confluent.hyper_m.restype = ctypes.POINTER(ctypes.c_double*2)

confluent.hyper_u.argtypes = [ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_double]
confluent.hyper_u.restype = ctypes.POINTER(ctypes.c_double*2)


def shiftedColorMap(name='shiftedcmap'):
	"""
	colormap for energy maps is defined
	return: newmap - colormap used for plotting
	"""

	tpl = np.zeros(shape=(14,3))
	tpl[0] = [189, 101, 67]
	tpl[1] = [162, 127, 108]
	tpl[2] = [134, 166, 163]
	tpl[3] = [133, 188, 191]
	tpl[4] = [151, 204, 208]
	tpl[5] = [172, 219, 224]
	tpl[6] = [192, 229, 230]
	tpl[7] = [210, 235, 227]
	tpl[8] = [222, 237, 214]
	tpl[9] = [221, 236, 198]
	tpl[10] = [216, 231, 171]
	tpl[11] = [201, 221, 136]
	tpl[12] = [185, 212, 103]
	tpl[13] = [170, 200, 84]

	red_part = []
	green_part = []
	blue_part = []

	for i in range(14):
		red_part.append((i/13,tpl[i][0]/255,tpl[i][0]/255))
		green_part.append((i/13,tpl[i][1]/255,tpl[i][1]/255))
		blue_part.append((i/13,tpl[i][2]/255,tpl[i][2]/255))

	cdict = {'red':   red_part, 'green': green_part, 'blue':  blue_part}

	newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
	plt.register_cmap(cmap=newcmap)

	return newcmap



def ksq(s,r,E,V,t,tau,n):
	return 2*s - r**(-2)*(E**2+V**2/4+(-1)**n*np.sqrt(t**2*(E**2 - V**2/4) + (tau*E*V-2*s*r**2)**2+0j))


def as1_in(s,r,m,E,V,t,tau,n):
	if m>=1:
		return ksq(s,r,E,V,t,tau,n)/2
	if m==0:
		return 2
	if m<=-1:
		return 2


def as1_out(s,r,E,U0,V,t,tau,n):
	return -(1+s+ksq(s,r,E-U0,V,t,tau,n)*(1-s)/4)


def as2_in(s,r,m,E,V,t,tau,n):
	if m>=1:
		return 2
	if m==0:
		return ksq(s,r,E,V,t,tau,n)/2
	if m<=-1:
		return ksq(s,r,E,V,t,tau,n)/2


def as2_out(s,r,E,U0,V,t,tau,n):
	return -(1-s+ksq(s,r,E-U0,V,t,tau,n)*(1+s)/4)


def as3_in(s,r,m,E,V,t,tau,n):
	if m>=1:
		return 2
	if m==0:
		return 2
	if m<=-1:
		return (ksq(s,r,E,V,t,tau,n)-4*s)/2


def as3_out(s,r,E,U0,V,t,tau,n):
	return -(1-s+(ksq(s,r,E-U0,V,t,tau,n)/4-1)*(1+s))


def as4_in(s,r,m,E,V,t,tau,n):
	if m>=1:
		return (ksq(s,r,E,V,t,tau,n)-4*s)/2
	if m==0:
		return (ksq(s,r,E,V,t,tau,n)-4*s)/2
	if m<=-1:
		return 2


def as4_out(s,r,E,U0,V,t,tau,n):
	return -(1+s+(ksq(s,r,E-U0,V,t,tau,n)/4+1)*(1-s))


def HmE_matrix_in(s,r,m,E,V,t,tau,n):
	matrix = np.array([
					[tau*V/2-E,-as1_in(s,r,m,E,V,t,tau,n)*r*1j,t,0],
					[-as2_in(s,r,m,E,V,t,tau,n)*r*1j,tau*V/2-E,0,0],
					[t,0,-tau*V/2-E,-as3_in(s,r,m,E,V,t,tau,n)*r*1j],
					[0,0,-as4_in(s,r,m,E,V,t,tau,n)*r*1j,-tau*V/2-E]
					])
	#print('matrix_in',matrix)
	return matrix 


def HmE_matrix_out(s,r,m,E,U0,V,t,tau,n):
	matrix = np.array([
					[tau*V/2-E+U0,-as1_out(s,r,E,U0,V,t,tau,n)*r*1j,t,0],
					[-as2_out(s,r,E,U0,V,t,tau,n)*r*1j,tau*V/2-E+U0,0,0],
					[t,0,-tau*V/2-E+U0,-as3_out(s,r,E,U0,V,t,tau,n)*r*1j],
					[0,0,-as4_out(s,r,E,U0,V,t,tau,n)*r*1j,-tau*V/2-E+U0]
					])
	return matrix 


def psi2_in(s,r,m,E,V,t,tau,n):
	output = null_space(HmE_matrix_in(s,r,m,E,V,t,tau,n))
	#print('psi2_in',t2-t1)
	return output


def psi2_out(s,r,m,E,U0,V,t,tau,n):
	output = null_space(HmE_matrix_out(s,r,m,E,U0,V,t,tau,n))
	return output


def det_matrix(s,r,E,U0,V,t,tau,n):
	matrix = np.zeros((4,4))
	matrix[:,0],matrix[:,1],matrix[:,2],matrix[:,3] = psi2_in(s,r,m,E,V,t,tau,1),\
													psi2_in(s,r,m,E,V,t,tau,2),\
													psi2_out(s,r,m,E,U0,V,t,tau,1),\
													psi2_out(s,r,m,E,U0,V,t,tau,2)
	return np.linalg.det(matrix)


def phi_in(alpha,ksi,s,r,m,E,V,t,tau,n):
	#print(alpha,ksi,s,r,m,E,V,t,tau,n)
	a = (abs(m+alpha)+1+s*(m-1-alpha))/2+ksq(s,r,E,V,t,tau,n)/4
	b = 1 + abs(m+alpha)
	z = ksi**2
	hyp1f1_abz_from_C = confluent.hyper_m(a.real,a.imag,b.real,b.imag,z.real,z.imag)
	hyp1f1_abz = hyp1f1_abz_from_C.contents[0] + hyp1f1_abz_from_C.contents[1]*1j
	#print('a,b,z,hyp1f1',a,b,z,hyp1f1_abz)
	return np.exp(-ksi**2/2)*ksi**(abs(m+alpha)+1/2)*hyp1f1_abz/gamma(1+abs(m+alpha))


def phi_out(alpha,ksi,s,r,m,E,U0,V,t,tau,n):
	#print(alpha,ksi,s,r,m,E,U0,V,t,tau,n)
	a = (abs(m+alpha)+1+s*(m-1-alpha))/2+ksq(s,r,E-U0,V,t,tau,n)/4
	b = 1 + abs(m+alpha)
	z = ksi**2
	hyperu_abz_from_C = confluent.hyper_u(a.real,a.imag,b.real,b.imag,z.real,z.imag)
	hyperu_abz = hyperu_abz_from_C.contents[0] + hyperu_abz_from_C.contents[1]*1j
	output = np.exp(-ksi**2/2)*ksi**(abs(m+alpha)+1/2)*hyperu_abz
	return output


def M1_in(ksi,s,r,m,E,V,t,tau,n):
	m1_in = mm.matrix([
		[phi_in(0,ksi,s,r,m,E,V,t,tau,n),0,0,0],
		[0,phi_in(-1,ksi,s,r,m,E,V,t,tau,n),0,0],
		[0,0,phi_in(0,ksi,s,r,m,E,V,t,tau,n),0],
		[0,0,0,phi_in(1,ksi,s,r,m,E,V,t,tau,n)]
		])
	#print('m1_in',m1_in,m1_in)
	m1_in = np.array(m1_in.tolist(),dtype = complex)
	#print('m1_in',m1_in)
	return m1_in


def M1_out(ksi,s,r,m,E,U0,V,t,tau,n):
	m1_out = np.array(mm.matrix([
		[phi_out(0,ksi,s,r,m,E,U0,V,t,tau,n),0,0,0],
		[0,phi_out(-1,ksi,s,r,m,E,U0,V,t,tau,n),0,0],
		[0,0,phi_out(0,ksi,s,r,m,E,U0,V,t,tau,n),0],
		[0,0,0,phi_out(1,ksi,s,r,m,E,U0,V,t,tau,n)]
		]).tolist(),dtype = complex)
	#print('m1_out,',m1_out)
	return m1_out


def psi1_in(ksi,s,r,m,E,V,t,tau,n):
	output = M1_in(ksi,s,r,m,E,V,t,tau,n).dot(psi2_in(s,r,m,E,V,t,tau,n))
	#print('psi1_in: ',t2-t1)
	return output


def psi1_out(ksi,s,r,m,E,U0,V,t,tau,n):
	output = M1_out(ksi,s,r,m,E,U0,V,t,tau,n).dot(psi2_out(s,r,m,E,U0,V,t,tau,n))
	#print('psi1_out: ',t2-t1)
	return output


def det(E,s,r,m,U0,V,t,tau,null_sp = False,svd = False,rcond = 0.01,save_matrix = False):
	v1 = psi1_in(r,s,r,m,E,V,t,tau,1)
	v2 = psi1_in(r,s,r,m,E,V,t,tau,2)
	v3 = psi1_out(r,s,r,m,E,U0,V,t,tau,1)
	v4 = psi1_out(r,s,r,m,E,U0,V,t,tau,2)
	matrix = np.zeros((4,4),dtype = complex)
	matrix[:,0:1],matrix[:,1:2],matrix[:,2:3],matrix[:,3:4] = 1/norm(v1,ord = 2,axis = 0)*v1,1/norm(v2,ord = 2,axis = 0)*v2,\
													1/norm(v3,ord = 2,axis = 0)*v3,1/norm(v4, ord = 2,axis = 0)*v4
	output = np.absolute(np.linalg.det(matrix))
	
	if not save_matrix:
		if not null_sp and not svd:
			return output
		if null_sp:
			return null_space(matrix,rcond = rcond)
		if svd:
			return np.linalg.svd(matrix)
	if save_matrix:
		if not null_sp and not svd:
			return output,matrix
		if null_sp:
			return null_space(matrix,rcond = rcond),matrix
		if svd:
			return np.linalg.svd(matrix),matrix

def U0_t(UinmeV, Rinnm):
	return UinmeV/(6582/10)*Rinnm


def V_t(VinmeV, Rinnm):
	return VinmeV/(6582/10)*Rinnm


def t_t(tinmeV, Rinnm):
	return tinmeV/(6582/10)*Rinnm


def r_t(BinT,Rinnm):
	return 275599/1e7*Rinnm*np.sqrt(BinT)


def E_t(EinmeV, Rinnm):
	return EinmeV/(6582/10)*Rinnm


def zero_sqrt_in(EinmeV,BinT, s,Rinnm,UinmeV,VinmeV,tinmeV,tau):
	return t_t(tinmeV, Rinnm)**2*(E_t(EinmeV, Rinnm)**2-V_t(VinmeV, Rinnm)**2/4)\
	+ (tau*E_t(EinmeV, Rinnm)*V_t(VinmeV, Rinnm)-2*s*r_t(BinT,Rinnm)**2)**2


def zero_sqrt_out(EinmeV,BinT,s,Rinnm,UinmeV,VinmeV,tinmeV,tau):
	return t_t(tinmeV, Rinnm)**2*((E_t(EinmeV, Rinnm)-U0_t(UinmeV, Rinnm))**2-V_t(VinmeV, Rinnm)**2/4)\
	+ (tau*(E_t(EinmeV, Rinnm)-U0_t(UinmeV, Rinnm))*V_t(VinmeV, Rinnm)-2*s*r_t(BinT,Rinnm)**2)**2




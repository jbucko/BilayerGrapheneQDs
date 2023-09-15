"""
identical to energy_lines_with_dets - only renamed
"""
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
import time
from scipy.interpolate import interp2d
from scipy.optimize import minimize_scalar
from scipy.signal import argrelextrema
from matplotlib import rc
import multiprocessing
#rc('text', usetex=True)

# own routines
sys.path.append('.')
from det_funs import*
from waveft_class import*

def sorting_parallel(j,dets_resolved,EinmeVs_resolved,BinTs_resolved,s,m,Rinnm,UinmeV,VinmeV,tinmeV,tau):
	"""
	calculates approximate minima along E axis
	params:
		j: defines cut along B-axis
		dets_resolved: determinant map
		EinmeVs_resolved: energies for which dets_resolved is calculated
		BinTs_resolved: magnetic field values for which dets_resolved is calculated
		s,m,Rinnm,Uinmev,VinmeV,tinmeV,tau: parameters from the model
	"""
	minima_fixed_B = []
	dets_for_fixed_B = dets_resolved[:,j]
	#print(dets_for_fixed_B)
	minima = argrelextrema(dets_for_fixed_B,np.less)[0] # local minima along the E-axis for specific B
	black_contour_slice = zero_sqrt_out(EinmeVs_resolved,BinTs_resolved[j],s,Rinnm,UinmeV,VinmeV,tinmeV,tau)
	# gap boundaries
	indices = [i for i in range(len(EinmeVs_resolved)-1) if np.sign(black_contour_slice[i]) == -np.sign(black_contour_slice[i+1])]

	#print(indices)
	for i in range(len(minima)):
		if minima[i]>indices[0] and minima[i]<indices[1]:
			lb = E_t(EinmeVs_resolved[minima[i]-6],Rinnm)
			ub = E_t(EinmeVs_resolved[minima[i]+6],Rinnm)
			res = minimize_scalar(det,args = (s,r_t(BinTs_resolved[j],Rinnm),m,U0_t(UinmeV,Rinnm),V_t(VinmeV,Rinnm),t_t(tinmeV,Rinnm),tau),method = 'Bounded',bounds = (lb,ub),options={'maxiter': 5})
			Emin = res.x*(6582/10)/Rinnm
			minima_fixed_B.append(Emin)
			#print(Emin)
	return minima_fixed_B

class energy_minima():
	def __init__(self,m,UinmeV,VinmeV,tau,s,Rinnm,tinmeV,BinTmin,BinTmax,nB,nE):
		"""
		class for extraction of energy lines from deerminant maps
		includes parallelization
		"""
		self.m = m
		self.UinmeV = UinmeV
		self.VinmeV = VinmeV
		self.tau = tau
		self.s = s
		self.Rinnm = Rinnm
		self.tinmeV = tinmeV
		self.BinTmin = BinTmin
		self.BinTmax = BinTmax
		self.nB = nB
		self.nE = nE

		self.BinTs = np.linspace(BinTmin, BinTmax ,nB)
		self.EinmeVs = np.linspace(self.UinmeV-self.VinmeV/2-1,self.UinmeV + self.VinmeV/2+1,nE)


	def dets_calc(self):
		"""
		determinant calculations
		"""
		print('calculating determinant...')
		ts = time.time()
		i = 0
		dets = []

		num_cores = multiprocessing.cpu_count()
		pool = multiprocessing.Pool(processes=num_cores)
		for E in self.EinmeVs:
			results = [pool.apply_async(det, args=(E_t(E,self.Rinnm),self.s,r_t(B,self.Rinnm),self.m,U0_t(self.UinmeV,self.Rinnm),V_t(self.VinmeV,self.Rinnm),t_t(self.tinmeV,self.Rinnm),self.tau)) for B in self.BinTs]
			along_B = [p.get() for p in results]
			dets.append(along_B)
		pool.close()

		"""
		data postprocessing and interpolation
		"""
		ti = time.time()
		dets = np.reshape(dets,(self.nE,self.nB))
		f = interp2d(self.BinTs,self.EinmeVs,dets, kind = 'linear')
		BinTs_resolved = np.linspace(self.BinTmin, self.BinTmax ,self.nB)
		EinmeVs_resolved = np.linspace(self.UinmeV-self.VinmeV/2-1,self.UinmeV + self.VinmeV/2+1,360)
		dets_resolved = f(BinTs_resolved,EinmeVs_resolved)
		te = time.time()
		print('determinant calculation finished. Ellapsed time: {:.4f},{:.4f}\n'.format(ti-ts,te - ti,'\n'))
		return BinTs_resolved, EinmeVs_resolved, dets_resolved, dets

	def search_minima(self):
		"""
		energy search from density map
		for each field value B we find local minima in the 1D array of interpolated energies
		and then also position of black line (dot edges).
		Within the found range we then do resolved search of the minima
		"""
		t1 = time.time()
		BinTs_resolved,EinmeVs_resolved,dets_resolved,_ = self.dets_calc()
		t2 = time.time()
		all_minima = []

		num_cores = multiprocessing.cpu_count()
		pool = multiprocessing.Pool(processes=num_cores)
		results = [pool.apply_async(sorting_parallel, args=(j,dets_resolved,EinmeVs_resolved,BinTs_resolved,self.s,self.m,self.Rinnm,self.UinmeV,self.VinmeV,self.tinmeV,self.tau)) for j in range(len(BinTs_resolved))]
		minima_fixed_B = [p.get() for p in results]
		all_minima.extend(minima_fixed_B)
		pool.close()

		t3 = time.time()
		print('search_minima times:',t2-t1,t3-t2)
		return BinTs_resolved,EinmeVs_resolved,dets_resolved, all_minima

	def calc_lines(self):
		"""
		here we devide found minima into separate curves
		"""

		placed = False
		all_minima_sorted = []

		t1 = time.time()
		BinTs_resolved,EinmeVs_resolved, dets_resolved, all_minima = self.search_minima()
		t2 = time.time()
		#print(all_minima,len(all_minima))
		l = 0
		while len(all_minima_sorted) == 0 and l<len(all_minima):
			for i in range(len(all_minima[l])):
				#print(all_minima[0])
				#print(det(E_t(all_minima[0][i],self.Rinnm),s,r_t(BinTs_resolved[0],self.Rinnm),m,U0_t(self.UinmeV,self.Rinnm),V_t(self.VinmeV,self.Rinnm),t_t(self.tinmeV,self.Rinnm),tau))
				if det(E_t(all_minima[l][i],self.Rinnm),self.s,r_t(BinTs_resolved[0],self.Rinnm),self.m,U0_t(self.UinmeV,self.Rinnm),V_t(self.VinmeV,self.Rinnm),t_t(self.tinmeV,self.Rinnm),self.tau)<0.006:
					all_minima_sorted.append([all_minima[l][i]])
			l+=1
		#print('l:',l)
		if l<len(all_minima):
			for i in range(l,len(all_minima)):
				occupied = []
				for j in range(len(all_minima[i])):
					diff = np.array([abs(all_minima[i][j]-all_minima_sorted[k][-1]) for k in range(len(all_minima_sorted))])
					idx = np.where(diff == np.min(diff))[0][0]
					# print('diff, idx;',diff,idx)
					if diff[idx] < 0.5 and idx not in occupied:
						all_minima_sorted[idx].append(all_minima[i][j])
						occupied.extend([idx])
					else:
						if det(E_t(all_minima[i][j],self.Rinnm),self.s,r_t(BinTs_resolved[i],self.Rinnm),self.m,U0_t(self.UinmeV,self.Rinnm),V_t(self.VinmeV,self.Rinnm),t_t(self.tinmeV,self.Rinnm),self.tau)<0.006:
							all_minima_sorted.append([all_minima[i][j]])
		t3 = time.time()
		all_minima_sorted_valid = []
		for i in range(len(all_minima_sorted)):
			length = len(all_minima_sorted[i])
			print('length comparisons:',length,0.999*len(BinTs_resolved))
			if length >= 0.999*len(BinTs_resolved) or length == len(BinTs_resolved) - 1:
				#print('length:',length,'valid line:',all_minima_sorted[i])
				if length == len(BinTs_resolved) - 1:
					print('extending line ',i)
					print(all_minima_sorted[i])
					all_minima_sorted[i].append(2*all_minima_sorted[i][-1] - all_minima_sorted[i][-2])
					print(2*all_minima_sorted[i][-1] - all_minima_sorted[i][-2])
					print(all_minima_sorted)
				all_minima_sorted_valid.append(all_minima_sorted[i])

		return BinTs_resolved,EinmeVs_resolved,dets_resolved, all_minima_sorted_valid




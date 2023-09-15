import numpy as np
import math
import argparse
import torch,sys
from torch import optim
import sobol_seq
import torch.nn as nn
import torch.nn.functional as F

import shutil
from scipy.signal import gaussian

from scipy.optimize import minimize
import nlopt
import os

#own routines that have to be loaded from my_lib folder
sys.path.append('./../my_libs')
from energy_lines import energy_minima
from waveft_class_optim import*
from classify_lines import classify


lr = 0.03
"""
general
"""
allowed_states = [2] # which state from the maps assume if there is more of them ([2] or [3], [1] is very close to the boundary)

#arrays for storing important quantities
losses_CRS = []
UV_CRS = []
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
dtype = torch.float
# reproducibility is good
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
rng = np.random.RandomState(seed)
if torch.cuda.is_available():
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


# main class
class ParametrizedHamiltonian(nn.Module):
	def __init__(self,m,UinmeV,VinmeV,tau, Uinit,Vinit,target = False,Bmin = 0.06,Bmax = 2.5):
		super().__init__()

		"""
		governing parameters
		"""
		self.Uinit = Uinit # initial point for descent (no here)
		self.Vinit = Vinit # initial point for descent (no here)
		
		##############----model variables----#############
		self.s = 1
		self.m = m
		self.tau = tau
		if tau ==2:
			self.tau = -1
		self.Rinnm = 20
		self.tinmeV = 400
		self.UinmeV = UinmeV
		self.VinmeV = VinmeV
		
		self.dimxi = 50
		self.dxi = 2/(self.dimxi+1)
		self.xi = torch.linspace(0.0, 3.0, self.dimxi, device=device, requires_grad=False) # radial points where to evaluate wavefunction

		self.dimB = 30 #grid along magnetic field
		self.nE = 70 #grid along energy

		self.BinTmin = Bmin
		self.BinTmax = Bmax
		self.dB = 2/(self.dimB+1)
		self.BinT = torch.linspace(self.BinTmin, self.BinTmax, self.dimB, device=device, requires_grad=False)
		self.ones = torch.ones((self.dimB,self.dimxi,4))
		###############################################


		##########---target state energies----##########
		if target:
			print('------------------------------------------------------\ntarget energies calculation...\n')
			t = time.time()

			#initialize class for energy lines extraction from determinant maps
			target_energies_class = energy_minima(self.m,self.UinmeV,self.VinmeV,self.tau,self.s,self.Rinnm,self.tinmeV,self.BinTmin,self.BinTmax,self.dimB,self.nE)
			# calculates energies
			target_energies = target_energies_class.calc_lines()[-1]

			# if there is more lines in the map, determines which to chhose
			clf = classify(self.m,self.UinmeV,self.VinmeV,self.tau,target_energies,self.BinTmax)

			for state in allowed_states:
				if state in clf:
					idx = np.where(np.array(clf)==state)[0][0]
					self.target_E = target_energies[idx]
				else:
					print('this state is not found')
			
			tt = time.time()
			print('\ntarget energies calculation finished after {:.4f} s...\n------------------------------------------------------\n'.format(tt-t))
			self.target_E_diff = torch.ones((self.dimB - 1),device = device, dtype = dtype, requires_grad = False)
			for i in  range(self.dimB -1):
				self.target_E_diff[i] = self.target_E[i+1] - self.target_E[i]
		##########################################



		# ##########----define parameters and hamiltonian derivatives----#########
		# self.params = nn.Parameter(torch.tensor([self.Uinit,self.Vinit],device=device), requires_grad=True)
		# self.HU = torch.eye(4,device = device, requires_grad = False,dtype = dtype)
		# self.HV = self.tau/2*torch.diag(torch.tensor([1,1,-1,-1],device = device, requires_grad = False,dtype = dtype))
		# #########################################################################

	def loss(self,target_E):
		gE = self.E - target_E
		return torch.matmul(gE,gE), gE*2*self.dB

	def loss_UV(self,UV,grad,phi,Uc,Vc,target_E):

		"""
		params:
			UV = [U,V]
			grad = [grad_U,grad_V]
			phi: rotation angle by domain transform
			Uc,Vc: shift of the domain center by the transformation
			target_E: target state energy
		returns:
			square loss between target and actual states
		"""
		
		#global losses_CRS
		global UV_CRS
		global allowed_states
		#transform
		Ut = UV[0]*np.cos(phi) - UV[1]*np.sin(phi) + Uc
		Vt = UV[0]*np.sin(phi) + UV[1]*np.cos(phi) + Vc
		
		if grad.size >0:
			grad = [0.,0.]

		#initialize class for energy lines extraction from determinant maps
		params_energies_class = energy_minima(self.m,Ut,Vt,self.tau,self.s,self.Rinnm,self.tinmeV,self.BinTmin,self.BinTmax,self.dimB,self.nE)
		#energies calculation
		energies = params_energies_class.calc_lines()[-1]
		
		# if there is more lines in the map, determines which to chhose
		clf = classify(self.m,Ut,Vt,self.tau,energies,self.BinTmax)
		print('params:',self.m,self.tau,Ut,Vt)

		for state in allowed_states:
			if state in clf:
				idx = np.where(np.array(clf)==state)[0][0]
				self.E = energies[idx]
			else:
				print('this state is not found')
				self.E = [-2*i for i in range(self.dimB)]


		self.E = torch.tensor(self.E,device = device, dtype = dtype, requires_grad = False)[:self.dimB]

		self.E_diff = torch.ones((self.dimB - 1),device = device, dtype = dtype, requires_grad = False)
		# calculates differences of consecutive points
		for i in  range(self.dimB -1):
			self.E_diff[i] = self.E[i+1] - self.E[i]

		print('self.E',self.E.shape)
		gE = self.E_diff - target_E
		loss = torch.matmul(gE,gE)/(self.dimB - 1)
		print('UV:',UV,' -> ',[Ut,Vt])
		UV_CRS.extend(UV)

		lll = float(loss.detach().numpy())
		print('loss:',lll)
		losses_CRS.append(lll)
		return lll


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('-e', '--epochs', type=int, default=100,
		 help="How many iterations for the optimizer")
	parser.add_argument('-m', '--m', type=int, default=0,
		 help="hyperparameret m - angular momentum")
	parser.add_argument('-Udef', '--UinmeV', type=float, default=60.0,
		 help="hyperparameter U - confining potential in meV")
	parser.add_argument('-Vdef', '--VinmeV', type=float, default=50.0,
		 help="hyperparameter V - gapping potential in meV")
	parser.add_argument('-Udef2', '--UinmeV2', type=float, default=60.0,
		 help="hyperparameter U - confining potential in meV")
	parser.add_argument('-Vdef2', '--VinmeV2', type=float, default=50.0,
		 help="hyperparameter V - gapping potential in meV")
	parser.add_argument('-tau', '--tau', type=int, default=1,
		 help="hyperparameter tau - valley number")
	parser.add_argument('-d', '--dot', type=str, default='CG3',
		 help="quantum dot")
	parser.add_argument('-sU1', '--sectionU1', type=int,default = 0,
		 help="section of U range to be searched")
	parser.add_argument('-sV1', '--sectionV1', type=int,default = 0,
		 help="section of V range to be searched")
	parser.add_argument('-np', '--npartitions', type=int,default = 5,
		 help="to how many subdomains shold original domain be split")
	print('!!! PLEASE, NO MORE OF TAU = 2, USE \-1 INSTEAD IN BASH')

	args = parser.parse_args()
	# if args.tau == 2:
	# 	args.tau = -1
	# if args.m == 3:
	# 	args.m = 


	t_start = time.time()

	"""
	read smoothened experimental data
	"""
	path = '.'
	data_E = []
	data_B = []
	for i in range(1):
		with open(path + '/B_lev_' + args.dot + '_e_smoothened.csv') as f:
			for line in f:
				data_B.append([float(x) for x in line[:-2].split()])

	for i in range(1):
		with open(path + '/E_lev_' + args.dot + '_e_smoothened.csv') as f:
			for line in f:
				data_E.append([float(x) for x in line[:-2].split()])
	print(len(data_E))

	"""
	choose a lowest couple
	"""
	target_B_down = data_B[0]
	target_B_up = data_B[1]
	from scipy.interpolate import interp1d
	target_E_down = data_E[0]
	target_E_up = data_E[1]
	f_B_down = interp1d(target_B_down,target_E_down, 3)
	f_B_up = interp1d(target_B_up,target_E_up, 3)

	"""
	restrict B up to 1.2 T
	"""
	for i in range(len(target_B_down)):
		if target_B_down[i]>=0.06:
			break
	for j in range(len(target_B_down)):
		if target_B_down[j]>=1.2:
			break
	Bmin = target_B_down[i]
	Bmax = target_B_down[j-1]
	print('Bmin: {}, \nBmax: {}'.format(Bmin,Bmax))
	target_B = np.linspace(Bmin,Bmax,ParametrizedHamiltonian(args.m,args.UinmeV,args.VinmeV, args.tau,60.,60.).dimB)
	target_E_down_nondiff = torch.tensor(f_B_down(target_B),dtype = dtype)
	target_E_up_nondiff = torch.tensor(f_B_up(target_B),dtype = dtype)

	"""
	go from lines to derivatives
	"""
	length_B = target_B.shape[0]
	target_E_down = torch.ones((length_B - 1),device = device, dtype = dtype, requires_grad = False)
	target_E_up = torch.ones((length_B - 1),device = device, dtype = dtype, requires_grad = False)
	for i in  range(length_B -1):
		target_E_down[i] = target_E_down_nondiff[i+1] - target_E_down_nondiff[i]
		target_E_up[i] = target_E_up_nondiff[i+1] - target_E_up_nondiff[i]

	# define a fraction of original interval to be searched over
	interval = (70-50)/args.npartitions

	t1 = time.time()
	res_UV = np.zeros(4)
	min_loss = np.zeros(2)
	evals = np.zeros(2)

	"""
	main loop - iterates over lower branch (j=0) and upper branch (j=1)
	"""
	for j in range(0,2):

		#######################################
		if j==0:
			target_E = target_E_down
			arg_m = args.m
			arg_tau = args.tau
		if j==1:
			target_E = target_E_up
			arg_m = -args.m
			arg_tau = -args.tau

		##################------NLopt-------______------______-----###########


		loss_hamil2 = ParametrizedHamiltonian(arg_m,args.UinmeV,args.VinmeV, arg_tau,float(60),float(60),False,Bmin,Bmax).loss_UV

		#define optimizer (controlled random search)
		opt = nlopt.opt(nlopt.GN_CRS2_LM, 2)

		# define subdomains of full interval 50-70 meV
		Umin = 50 + args.sectionU1*interval
		Umax = 50 + (args.sectionU1 + 1)*interval
		Uinit = 0.5*(Umin + Umax)

		Vmin = 50 + args.sectionV1*interval
		Vmax = 50 + (args.sectionV1 + 1)*interval
		Vinit = 0.5*(Vmin + Vmax)
		init = [Uinit,Vinit]

		"""
		setting the following three quantities to zero means we do not want to transform the domain (thesis)
		"""
		phi = 0.
		Uc,Vc = 0.,0.
		
		opt.set_min_objective(lambda x, grad: loss_hamil2(x,grad,phi,Uc,Vc,target_E)) # set loss function
		
		# intervals
		opt.set_lower_bounds([Umin,Vmin])
		opt.set_upper_bounds([Umax,Vmax])


		#opt.set_xtol_rel(1e-4)
		opt.set_ftol_rel(1e-4) 
		opt.set_maxeval(600)
		opt.verbose = 1
		x = opt.optimize(np.array(init,dtype = np.float64))
		res_UV[2*j] = x[0]*np.cos(phi) - x[1]*np.sin(phi) + Uc # transformation (nothing happens when phi,Uc,Vc = 0,0,0)
		res_UV[2*j+1] = x[0]*np.sin(phi) + x[1]*np.cos(phi) + Vc
		minf = opt.last_optimum_value()
		min_loss[j] = minf

		print("optimum at ", x[0], x[1],' -> ',res_UV[2*j],res_UV[2*j+1])
		print("minimum value = ", minf)
		print("result code = ", opt.last_optimize_result())
		
		evals[j] = opt.get_numevals()
		t_end = time.time()

		#print('line definition time:',t_inter - t_start)
		print(evals[j])
		print('total time:',t_end - t_start)

	#where to store results
	path = './down_at_m_{}_tau_{}_secU1_{}_secV1_{}'.format(args.m,args.tau,args.sectionU1,args.sectionV1)


	if os.path.exists(path) and os.path.isdir(path):
		shutil.rmtree(path)
	os.mkdir(path)

	UV_CRS = np.reshape(np.array(UV_CRS),(-1,2))

	# save results
	np.savetxt(path+'/losses_global.csv',losses_CRS)
	np.savetxt(path+'/UV_global.csv',UV_CRS)

	res = np.zeros(shape = (2,4))
	res[0,0] = res_UV[0] #*np.cos(phi) - x[1]*np.sin(phi) + Uc
	res[0,1] = res_UV[1]#*np.sin(phi) + x[1]*np.cos(phi) + Vc
	res[0,2] = res_UV[2]#*np.cos(phi) - x[1]*np.sin(phi) + Uc
	res[0,3] = res_UV[3]#*np.sin(phi) + x[1]*np.cos(phi) + Vc
	res[1,0] = min_loss[0]
	res[1,1] = min_loss[1]
	res[1,2] = evals[0]
	res[1,3] = evals[1]
	
	np.savetxt(path+'/results.csv',res)
	print('---number of evaluations is {} and {}'.format(evals[0],evals[1]))

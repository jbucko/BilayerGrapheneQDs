import numpy as np
import math
import argparse
# PyTorch utilities
import torch,sys,os
from torch import optim
import sobol_seq
import torch.nn as nn
import torch.nn.functional as F
import shutil
from scipy.signal import gaussian
from scipy.optimize import minimize

import nlopt
# from numpy import *

# own routines
sys.path.append('./../my_libs')
from waveft_class_optim import*
from energy_lines import energy_minima


lr = 0.03
allowed_states = [2]
iteration = 0

losses_CRS = []
UV_CRS = []
offsets_CRS = []
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
dtype = torch.float
# reproducibility is good



# main class
class ParametrizedHamiltonian(nn.Module):
	def __init__(self,m,UinmeV,VinmeV,tau, Uinit,Vinit,target = False,Bmin = 0.06,Bmax = 2.5):
		super().__init__()
		"""
		governing parameters
		"""
		self.Uinit = Uinit # initial point for descent (no here)
		self.Vinit = Vinit # initial point for descent (no here)
		##############----variables----#############
		self.s = 1
		self.m = m
		self.tau = tau
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



		##########----define parameters and hamiltonian derivatives----#########
		self.params = nn.Parameter(torch.tensor([self.Uinit,self.Vinit],device=device), requires_grad=True)
		self.HU = torch.eye(4,device = device, requires_grad = False,dtype = dtype)
		self.HV = self.tau/2*torch.diag(torch.tensor([1,1,-1,-1],device = device, requires_grad = False,dtype = dtype))
		#########################################################################

	def loss(self,target_E):
		gE = self.E - target_E
		return torch.matmul(gE,gE), gE*2*self.dB # return square loss and its derivative

	def loss_UV(self,UV,grad,phi,Uc,Vc,target_E):
		"""
		function used for CRS optimization
		params:
			UV = [U,V]
			grad = [grad_U,grad_V]
			phi: rotation angle by domain transform
			Uc,Vc: shift of the domain center by the transformation
			target_E: target state energy
		returns:
			square loss between target and actual states
		"""

		global UV_CRS
		global allowed_states
		global iteration
		iteration +=1

		#domain transform
		Ut = UV[0]*np.cos(phi) - UV[1]*np.sin(phi) + Uc
		Vt = UV[0]*np.sin(phi) + UV[1]*np.cos(phi) + Vc
		
		if grad.size >0:
			grad = [0.,0.]

		#initialize class for energy lines extraction from determinant maps
		params_energies_class = energy_minima(self.m,Ut,Vt,self.tau,self.s,self.Rinnm,self.tinmeV,self.BinTmin,self.BinTmax,self.dimB,self.nE)
		#energies calculation
		energies = params_energies_class.calc_lines()[-1]

		print('params:',self.m,self.tau,Ut,Vt)

		# find the most suitable line from a single map - the one with smallest loss
		loss_min = 1e8
		i_min = 100
		for i in range(len(energies)):
			self.E = torch.tensor(energies[i],device = device, dtype = dtype, requires_grad = False)

			gE = self.E - target_E
			loss = torch.matmul(gE,gE)/self.dimB
			if loss < loss_min:
				loss_min = loss
				i_min = i
		
		print('state:',i_min,'UV:',UV,' -> ',[Ut,Vt])
		UV_CRS.extend(UV)
		
		lll = float(loss_min)#0].astype(np.float64)
		print('loss:',lll)
		losses_CRS.append(lll)
		
		return lll


	def adjoint_gradient(self,state = False,target_E = None):
		"""
		function calculating gradient of the loss wrt. U and V
		params:
			target_E: target state
			state: if true, wavefunction if evaluated and used to calculate gradient, otherwise only loss is returned
		returns:
			loss, grad_loss_U,grad_loss_V - if state true
			loss - if state false
		"""
		global allowed_states

		#########----energies and eigenstates for actual state----###########
		print('------------------------------------------------------\nparams energies and eigenstates calculation...\n')
		t = time.time()

		#map calculation and energy lines extraction
		params_energies_class = energy_minima(self.m,self.params[0].detach().numpy(),self.params[1].detach().numpy(),self.tau,self.s,self.Rinnm,self.tinmeV,self.BinTmin,self.BinTmax,self.dimB,self.nE)
		energies = params_energies_class.calc_lines()[-1]

		if len(energies) == 0:
			print('no sufficient line found')
			return np.nan

		#find optimal line
		loss_min = 1e8
		i_min = 100
		for i in range(len(energies)):
			self.E = torch.tensor(energies[i],device = device, dtype = dtype, requires_grad = False)

			gE = self.E - target_E
			loss = torch.matmul(gE,gE)/self.dimB
			print('i: {}, loss: {}'.format(i,loss))
			if loss < loss_min:
				loss_min = loss
				i_min = i

		print('params:',self.m,self.tau,self.params[0].detach().numpy(),self.params[1].detach().numpy())
		# self.E = torch.tensor(energies[1],device = device, dtype = dtype, requires_grad = False)
		self.E = energies[i_min]
		print(self.E)

		#calculates wavefunction at array points xi
		if state:
			params_wf_arr = []
			# print(self.BinT,self.E)
			for i in range(self.dimB):
				print(i)
				params_wf = psi_complete(self.E[i],self.BinT[i].detach().numpy(),self.s,self.m,self.tau,self.Rinnm,self.tinmeV,self.params[0].detach().numpy(),self.params[1].detach().numpy(),1)
				params_wf_arr.append([params_wf.psisq_joint_elements(xi.detach().numpy()) for xi in self.xi]) # without normalization


			self.params_wf_tensor = torch.tensor(params_wf_arr,dtype = dtype, requires_grad = False).squeeze(3)
			#print(self.params_wf_tensor.shape)
		tt = time.time()
		print('params energies and eigenstates calculation finished after {:.4f} s...\n------------------------------------------------------\n'.format(tt-t))
		self.E = torch.tensor(energies[i_min],device = device, dtype = dtype, requires_grad = False)
		###################################################################################


		# evaluate loss
		loss, lossE = self.loss(target_E)

		if state:
			#########----derivatives of hamiltonian and losses----##########
			EU = torch.einsum('ijk,kl,ijl->i', self.params_wf_tensor, self.HU, self.ones)
			EV = torch.einsum('ijk,kl,ijl->i', self.params_wf_tensor, self.HV, self.ones)
			#print(Ep.shape,lossE.shape)
			lossU = torch.matmul(lossE,EU)#.unsqueeze(0) # dL/dp = (del)L/(del)E * (del)E/(del)p
			lossV = torch.matmul(lossE,EV)#.unsqueeze(0) # dL/dp = (del)L/(del)E * (del)E/(del)p
			print('------------------------------------------------------\nloss: {}, lossU: {}, lossV: {}'.format(loss.data/self.dimB, lossU.data, lossV.data))
			###############################################################
			return loss/self.dimB, lossU, lossV
		else:
			print(loss/self.dimB)
			return loss/self.dimB#, torch.tensor([0]),torch.tensor([0])

def gradient_based(hamil,optimizer,n_epochs,iteration,path,Umin,Umax,Vmin,Vmax, no_point,save = False,target_E = None):
	loss_save = -torch.ones((args.epochs))
	paramU_save = -torch.ones((args.epochs+1))
	paramV_save = -torch.ones((args.epochs+1))
	#initialize parameters
	for p in hamil.parameters():
		paramU_save[0] = p[0]
		paramV_save[0] = p[1]

	#perform gradient descent
	for i in range(n_epochs):

		print('######################################################\n##############---------epoch {}--------################\n######################################################\n'.format(i))
		loss, lossU, lossV = hamil.adjoint_gradient(True,target_E)
		if np.isnan(loss):
			return 0,0,loss
			break

		if loss > no_point:
			print('loss: {} greater than allowed value {}'.format(loss,no_point))
			return 0,0,loss
			break

		# Reverse mode AD
		optimizer.zero_grad()
		# loss.backward()

		# Adjoint sensitivity method
		for p in hamil.parameters():
			print('parameters:',p.data)
			p.grad = torch.tensor([lossU.clone(),1*lossV.clone()],dtype = dtype)
			optimizer.step() # here we can put inside the loop as it loops once only
			#p[0],p[1] = parameter_check(p[0],p[1])
			print('after update:',p.data,'\n------------------------------------------------------\n')
			if p[0]>Umax:
				p[0] = Umax
			if p[0] < Umin:
				p[0] = Umin
			if p[1]>Vmax:
				p[1] = Vmax
			if p[1] < Vmin:
				p[1] = Vmin
			print('after update and boundary check:',p.data,'\n------------------------------------------------------\n')
			paramU_save[i+1] = p[0]
			paramV_save[i+1] = p[1]
		loss_save[i] = loss

		#stopping
		if i>5:
			print('ratio:',abs(loss_save[i]-loss_save[i-1])/loss_save[i])
			if abs(loss_save[i]-loss_save[i-1])/loss_save[i]<0.02:
				print('DECREASING TOO SLOW - EXITTING')
				break

	#save data
	if save:
		np.savetxt(path+'/'+'losses_U_iter_{}.csv'.format(iteration),loss_save.detach().numpy())
		np.savetxt(path+'/'+'paramU_U_iter_{}.csv'.format(iteration),paramU_save.detach().numpy())
		np.savetxt(path+'/'+'paramV_U_iter_{}.csv'.format(iteration),paramV_save.detach().numpy())

	U1_res = paramU_save[i+1].detach().numpy()
	V1_res = paramV_save[i+1].detach().numpy()
	loss_res = loss_save[i].detach().numpy()
	return U1_res,V1_res,loss_res

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
	parser.add_argument('-tau', '--tau', type=int, default=1,
		 help="hyperparameter tau - valley number")
	parser.add_argument('-d', '--dot', type=str, default='CG3',
		 help="quantum dot")
	parser.add_argument('-off', '--offset', type=float,
		 help="offset for given m and tau found in ground state search")
	parser.add_argument('-rg', '--range', type=float,default = 50,
		 help="range for U to optimize over")
	parser.add_argument('-seed', '--seed', type=int,default = 0,
		 help="random seed")


	print('!!! PLEASE, NO MORE OF TAU = 2, USE \-1 INSTEAD IN BASH')

	args = parser.parse_args()


	seed = args.seed
	np.random.seed(seed)
	torch.manual_seed(seed)
	rng = np.random.RandomState(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False

	####################################
	"""
	results of the lowest state searches for different QDs
	bottom_index: index of the bottom branch -index of the line in file containing smoothened data (0 for ground state, 2 for first higher,...)
	top_index: index of the top branch -index of the line in file containing smoothened data (1 for ground state, 3 for first higher,...)
	"""
	if args.dot == 'CG2':
		off_down, off_up = 44.303493261055706, 40.4374974551569
		Ugr_down, Vgr_down = 53.723, 69.1456
		Ugr_up, Vgr_up = 60.3219, 50.5569
		bottom_index = 6
		top_index = 7
	if args.dot == 'CG3':
		off_down, off_up = 41.172988347488406, 39.954204076123084
		Ugr_down, Vgr_down = 62.0299, 50.0615
		Ugr_up, Vgr_up = 61.2954, 51.2555
		bottom_index = 6
		top_index = 7
	if args.dot == 'CG9':
		off_down, off_up = 40.64315502131403, 40.22771175499579
		Ugr_down, Vgr_down = 60.0874, 50.1455
		Ugr_up, Vgr_up = 61.3245, 51.6202
		bottom_index = 6
		top_index = 7
	Vmin = 50
	Vmax = 70

	"""
	optimization controls
	"""
	U_range = args.range
	n_sobol = 15
	n_ask = 2
	max_eval = 600
	rel_tol = 1e-4
	no_point = 5e2 # artificial increase of loss for when a line from one iteration to another ceases to exists points (thesis)

	path_dot = './dot_{}'.format(args.dot)

	if os.path.exists(path_dot) and os.path.isdir(path_dot):
		print(path_dot,'exists')
	else:
		try:
			os.mkdir(path_dot)
		except:
			print('folder could not be created')

	path_seed = './dot_{}/seed_{}_U_range_{}'.format(args.dot,args.seed,args.range)

	if os.path.exists(path_seed) and os.path.isdir(path_seed):
		print(path_seed,'exists')
	else:
		try:
			os.mkdir(path_seed)
		except:
			print('folder could not be created')



	t_start = time.time()


	"""
	reading target states (from experimental data)
	"""

	try:
		path = '.'
		open(path + '/B_lev_' + args.dot + '_e_smoothened.csv')
	except:
			print('no data found')
	data_E = []
	data_B = []
	for i in range(1):
		with open(path + '/B_lev_' + args.dot + '_e_smoothened.csv') as f:
			for line in f:
				data_B.append([float(x) for x in line[:-2].split()])

		with open(path + '/E_lev_' + args.dot + '_e_smoothened.csv') as f:
			for line in f:
				data_E.append([float(x) for x in line[:-2].split()])
	print(len(data_E))

	target_B_down = data_B[bottom_index]
	target_B_up = data_B[top_index]
	from scipy.interpolate import interp1d
	target_E_down = data_E[bottom_index]
	target_E_up = data_E[top_index]
	f_B_down = interp1d(target_B_down,target_E_down, 3)
	f_B_up = interp1d(target_B_up,target_E_up, 3)

	# cutting the data
	for i in range(len(target_B_down)):
		if target_B_down[i]>=0.06:
			break
	for j in range(len(target_B_down)):
		if target_B_down[j]>=1.2:
			break
	Bmin = target_B_down[i]
	Bmax = target_B_down[j-1]
	print('Bmin: {}, \nBmax: {}'.format(Bmin,Bmax))
	dimB = ParametrizedHamiltonian(args.m,args.UinmeV,args.VinmeV, args.tau,50.,50.).dimB
	target_B = np.linspace(Bmin,Bmax,dimB)
	target_E_down = torch.tensor(f_B_down(target_B),dtype = dtype)
	target_E_up = torch.tensor(f_B_up(target_B),dtype = dtype)


	t1 = time.time()
	res_UV = np.zeros(4)
	min_loss = np.zeros(2)
	evals = np.zeros(2)

	##########################################################################################

	"""
	j=0 fits bottom branch, j=1 fits top branch
	"""
	for j in range(0,2):

		no_descent = False # whether to descent to the shallow valley of the landscape before running CRS
		#######################################
		if j==0:
			target_E = target_E_down + off_down*torch.ones((dimB))
			arg_m = args.m
			arg_tau = args.tau
			offset = off_down
			Umin = Ugr_down - 5
			Umax = Ugr_down + U_range - 5
		if j==1:
			target_E = target_E_up + off_up*torch.ones((dimB))
			arg_m = -args.m
			arg_tau = -args.tau
			offset = off_up
			Umin = Ugr_up - 5
			Umax = Ugr_up + U_range - 5


		print('optimization region for U: {:.2f} - {:.2f}'.format(Umin,Umax))
		print('optimization region for V: {:.2f} - {:.2f}'.format(Vmin,Vmax))


		"""
		candidates for the descent initializations - 4 corners of the optimization region 
		and then n_sobol more points covering the region
		"""
		rand = np.random.randint(0,1000)
		U_sobol = np.zeros((n_sobol + 4,1))
		V_sobol = np.zeros((n_sobol + 4,1))
		for r in range(4,n_sobol+4):
			U_sobol[r,:] = sobol_seq.i4_sobol(1,10*seed + r)[0]*(Umax - Umin) + Umin
			V_sobol[r,:] = sobol_seq.i4_sobol(1,10*seed + r)[0]*(Vmax - Vmin) + Vmin
		UV_init_sobol = np.concatenate((U_sobol,V_sobol),axis = 1)
		UV_init_sobol[:4,0] = Umin,Umin,Umax,Umax
		UV_init_sobol[:4,1] = Vmin,Vmax,Vmin,Vmax
		print(UV_init_sobol, UV_init_sobol.shape)
		losses_sobol = []
		# sys.exit()

		##########################################
		"""
		attempt to descent to the valley
		"""
		UU = []
		VV = []
		index_init = 0
		while index_init < UV_init_sobol.shape[0] and len(UU)<2:
			print('index_init',index_init)
			hamil = ParametrizedHamiltonian(arg_m,args.UinmeV,args.VinmeV,arg_tau,UV_init_sobol[index_init,0],UV_init_sobol[index_init,1],False,Bmin,Bmax)
			optimizer = optim.SGD(hamil.parameters(), lr=lr, momentum=0.3)
			hamil.zero_grad()
			try:
				U,V,loss = gradient_based(hamil,optimizer,args.epochs,i,'none',Umin,Umax,Vmin,Vmax,no_point, False,target_E)
				if np.isnan(loss):
					# index_init +=1
					print('no sufficient state found')
					raise ValueError('no sufficient state found')
				if loss > no_point:
					# index_init +=1
					print('loss probably coming from wrong state')
					raise ValueError('loss probably coming from wrong state')
				UU.append(U)
				VV.append(V)
				print('converged points:',U,V,'loss:',loss)
				index_init += 1
			except:
				index_init +=1
				# print(error)
				print('gradient descent not succesfull')

		if index_init >= UV_init_sobol.shape[0]:
			no_descent =True

		print(UU,VV)
		print('no_descent',no_descent)


		"""
		define the valley region and consider region boundaries
		"""
		special = False
		if not no_descent:
			try:
				if abs(VV[0] - VV[1])<1e-3:
					Vmin = max(Vmin,VV[0] - 4)
					Vmax = min(Vmax,VV[0] + 4)
					special = True
				if abs(UU[0] - UU[1])<1e-3:
					Umin = max(Umin,UU[0] - 4)
					Umax = min(Umax,UU[0] + 4)
					special = True
				if abs(VV[0] - VV[1])>1e-6 and abs(UU[0] - UU[1])>1e-6:
					U1,V1=UU[0],VV[0]
					U2,V2=UU[1],VV[1]
					if V1==V2:
						if V1>(Vmin+Vmax)/2:
							V1 -=0.05
						else:
							V1 +=0.05
					a = (U2-U1)/(V2-V1)
					b=U1-a*V1
					Us = np.linspace(Umin,Umax,500)
					Vs = (Us-b)/a
					Us_cut,Vs_cut = [],[]
					for i in range(len(Us)):
						if Vs[i]>=Vmin and Vs[i]<=Vmax:
							Us_cut.append(Us[i])
							Vs_cut.append(Vs[i])
					# print('Vs:',Vs)
					print(Us_cut[0],Us_cut[-1])
					print(Vs_cut[0],Vs_cut[-1])

					Uc,Vc = (Us_cut[0]+ Us_cut[-1])/2,(Vs_cut[0]+ Vs_cut[-1])/2
					Uth = np.sqrt((Us_cut[0] - Us_cut[-1])**2 + (Vs_cut[0] - Vs_cut[-1])**2)/2
					phi = -np.pi/2 - np.arctan(a)
					print('a: {}, phi = {}'.format(a,phi))
			except:
				no_descent = True
				print('no descent set to True')

		print('status special:',special)

		##################------NLopt-------______------______-----###########
		"""
		performs Controlled Random Search
		a) if descent was succesfull, in the restricted region
		b) if descent not succesfull, perform on the whole region
		"""


		loss_hamil2 = ParametrizedHamiltonian(arg_m,args.UinmeV,args.VinmeV, arg_tau,float(50),float(50),False,Bmin,Bmax).loss_UV

		opt = nlopt.opt(nlopt.GN_CRS2_LM, 2)
		print(target_E,target_E + offset*torch.ones((dimB),dtype = dtype))

		if no_descent or special:
			# no transform of domain
			print('optimization domain for NLopt is (Umin,Umnax,Vmin,Vmax):',Umin,Umax,Vmin,Vmax)
			phi = 0.
			Uc,Vc = 0.,0.
			opt.set_min_objective(lambda x, grad: loss_hamil2(x,grad,phi,Uc,Vc,target_E))
			opt.set_lower_bounds([Umin,Vmin])
			opt.set_upper_bounds([Umax,Vmax])
			l1 = np.random.uniform(0,1)
			Uinit = (1-l1)*Umin + l1*Umax # randomization in descent initialization
			l2 = np.random.uniform(0,1)
			Vinit = (1-l2)*Vmin + l2*Vmax # randomization in descent initialization
			init = [Uinit, Vinit]
		else:
			#transform of the domain
			opt.set_min_objective(lambda x, grad: loss_hamil2(x,grad,phi,Uc,Vc,target_E))
			opt.set_lower_bounds([-Uth,-2])
			opt.set_upper_bounds([Uth,2])
			l1 = np.random.uniform(0,1)
			Uinit = (1-l1)*(-1*Uth) + l1*Uth # randomization in descent initialization
			l2 = np.random.uniform(0,1)
			Vinit = (1-l2)*(-2) + l2*2 # randomization in descent initialization
			init = [Uinit, Vinit]


		#opt.set_xtol_rel(1e-4)
		opt.set_ftol_rel(rel_tol)
		opt.set_maxeval(max_eval)
		opt.verbose = 1
		x = opt.optimize(np.array(init,dtype = np.float64))
		res_UV[2*j] = x[0]*np.cos(phi) - x[1]*np.sin(phi) + Uc
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

	#hamil = ParametrizedHamiltonian(arg_m,args.UinmeV,args.VinmeV, arg_tau,50.,50.)
	path = './dot_{}/seed_{}_U_range_{}/down_at_m_{}_tau_{}'.format(args.dot,args.seed,args.range,args.m,args.tau)

	if os.path.exists(path) and os.path.isdir(path):
		shutil.rmtree(path)
	os.mkdir(path)

	UV_CRS = np.reshape(np.array(UV_CRS),(-1,2))
	#UV_CRS[0] = init
	#print(UV_CRS)
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

import logging
logging.basicConfig(level=logging.WARNING)

import argparse

import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres

from hpbandster.optimizers import BOHB as BOHB
from hpbandster.examples.commons import MyWorker
from MLP_worker import MLP_Worker
import numpy as np
import pickle



parser = argparse.ArgumentParser(description='Example 1 - sequential and local execution.')
parser.add_argument('--min_budget',   type=float, help='Minimum budget used during the optimization.',    default=15)
parser.add_argument('--max_budget',   type=float, help='Maximum budget used during the optimization.',    default=900)
parser.add_argument('--n_iterations', type=int,   help='Number of iterations performed by the optimizer', default=50)
parser.add_argument('--n_workers', type=int,   help='Number of workers to run in parallel.', default=20)

parser.add_argument('-m', '--m', type=int, default=0,
	 help="hyperparameret m - angular momentum")
parser.add_argument('-tau', '--tau', type=int, default=1,
	 help="hyperparameter tau - valley number")
parser.add_argument('-d', '--dot', type=str, default='CG3',
	 help="quantum dot")
parser.add_argument('-r', '--randomstate', type=int, default=0,
	 help="random state initialization of neural net")


args = parser.parse_args()

np.random.seed(args.randomstate)

l_rate = .001
n_epochs = 5000
batch_size = 100
patience = 200
data_x_load = []
data_y_load = []
data_x_load_permuted = []
data_y_load_permuted = []

correlations = 1 # if set to True (1), dataset with crosscorrelations is loaded, otherwise only line data are used

"""
reading and processing data (see thesis)
"""
for m in [-2,-1,0,1,2]:
	for tau in [-1,1]:
		"""read data"""
		which_data = 'm_{}_tau_{}'.format(m,tau)
		print(which_data)
		try:
			# local
			PATH_DATA = "./../line_data_storing_v2_50-70/"
			data_x_load.append(np.genfromtxt(PATH_DATA+'lines_'+which_data+'.csv'))
			data_y_load.append(np.genfromtxt(PATH_DATA+'labels_'+which_data+'.csv'))
			print('x and y data loaded')
		except:
			print('no data available')


	length_down = data_x_load[-2].shape[0]
	length_up = data_x_load[-1].shape[0]
	dim = data_x_load[-1].shape[1]
	print(length_down,length_up,dim)
	data_permuted = np.zeros((length_down+length_up,2*dim - 2))
	data_y_permuted = np.zeros(length_down+length_up)
	data_y_permuted[:length_down] = data_y_load[-2][:,0]
	# append random up line behind down line
	for i in range(length_down):
		p = np.random.randint(length_up)
		# print(p)
		for j in range(dim - 1):
			data_permuted[i,j] = data_x_load[-2][i,j + 1] - data_x_load[-2][i,j]
		for j in range(dim - 1):
			data_permuted[i,j + dim - 1] = data_x_load[-1][p,j + 1] - data_x_load[-1][p,j]

	# prepend random down line in front of up line
	for i in range(length_up):
		p = np.random.randint(length_down)
		# print(p)
		data_y_permuted[length_down + i] = data_y_load[-2][p,0]
		for j in range(dim - 1):
			data_permuted[length_down + i,j] = data_x_load[-2][p,j + 1] - data_x_load[-2][p,j]
		for j in range(dim - 1):
			data_permuted[i,j + dim - 1] = data_x_load[-1][i,j + 1] - data_x_load[-1][i,j]

	data_x_load_permuted.append(data_permuted)
	data_y_load_permuted.append(data_y_permuted)
	print(data_permuted.shape,data_y_permuted.shape)
# sys.exit()


STORE_MODEL_PATH = './'
data_x = np.concatenate((data_x_load_permuted[0],data_x_load_permuted[1],data_x_load_permuted[2],data_x_load_permuted[3],data_x_load_permuted[4]),axis = 0)
data_y = np.concatenate((data_y_load_permuted[0],data_y_load_permuted[1],data_y_load_permuted[2],data_y_load_permuted[3],data_y_load_permuted[4]),axis = 0)
print(data_x.shape)
print(data_y.shape,data_y[0])


"""
calculate cross-correlations
"""
data_corr = np.zeros((data_x.shape[0],2*data_x.shape[1] - 1))

for i in range(data_x.shape[0]):
	data_corr[i,:] = np.correlate(data_x[i],data_x[i],mode = 'full')


if correlations:
	data_x_merged = np.concatenate((data_x,data_corr),axis = 1)
else:
	data_x_merged = data_x


data = np.concatenate((data_x_merged,np.reshape(data_y,(-1,1))),axis = 1)
np.random.shuffle(data) # shuffling data



x_train = data[:,:data_x_merged.shape[1]]
y_train_temp = data[:,data_x_merged.shape[1]:]

"""
transforming labels to 0 -4 
"""
y_train = np.zeros(x_train.shape[0])
for i in range(len(y_train)):
	if y_train_temp[i]<0:
		y_train[i] = (abs(y_train_temp[i,0])+2)
	else:
		y_train[i] = y_train_temp[i]


###############------counting and splitting the dataset--------###########################
print(x_train.shape,y_train.shape)

u, yc = np.unique(y_train, return_counts=True)
testcounts = dict(zip(u,yc))
print('final dataset',testcounts)

split1 = int(x_train.shape[0]*0.8)
split2 = int(x_train.shape[0]*0.9)


x_test, y_test = x_train[split2:],y_train[split2:]
x_validate, y_validate = x_train[split1:split2],y_train[split1:split2]
x_train, y_train = x_train[:split1],y_train[:split1]
############################################################################

# Step 1: Start a nameserver (see example_1)
NS = hpns.NameServer(run_id='MLPClassifier', host='127.0.0.1', port=None)
NS.start()

# Step 2: Start the workers
# Now we can instantiate the specified number of workers. To emphasize the effect,
# we introduce a sleep_interval of one second, which makes every function evaluation
# take a bit of time. Note the additional id argument that helps separating the
# individual workers. This is necessary because every worker uses its processes
# ID which is the same for all threads here.
workers=[]
for i in range(args.n_workers):
    w = MLP_Worker(sleep_interval = 0.5, x_train = x_train, y_train = y_train, x_validate = x_validate, y_validate = y_validate, nameserver='127.0.0.1',run_id='MLPClassifier', id=i)
    w.run(background=True)
    workers.append(w)

# Step 3: Run an optimizer
# Now we can create an optimizer object and start the run.
# We add the min_n_workers argument to the run methods to make the optimizer wait
# for all workers to start. This is not mandatory, and workers can be added
# at any time, but if the timing of the run is essential, this can be used to
# synchronize all workers right at the start.
bohb = BOHB(  configspace = w.get_configspace(),
              run_id = 'MLPClassifier',
              min_budget=args.min_budget, max_budget=args.max_budget
           )
res = bohb.run(n_iterations=args.n_iterations, min_n_workers=args.n_workers)

# Step 4: Shutdown
# After the optimizer run, we must shutdown the master and the nameserver.
bohb.shutdown(shutdown_workers=True)
NS.shutdown()

# Step 5: Analysis
# Each optimizer returns a hpbandster.core.result.Result object.
# It holds informations about the optimization run like the incumbent (=best) configuration.
# For further details about the Result object, see its documentation.
# Here we simply print out the best config and some statistics about the performed runs.
id2config = res.get_id2config_mapping()
incumbent = res.get_incumbent_id()
traj = res.get_incumbent_trajectory()
all_runs = res.get_all_runs()
curves = res.get_learning_curves()



print('Best found configuration:', id2config[incumbent]['config'])
print('A total of %i unique configurations where sampled.' % len(id2config.keys()))
print('A total of %i runs where executed.' % len(res.get_all_runs()))
print('Total budget corresponds to %.1f full function evaluations.'%(sum([r.budget for r in all_runs])/args.max_budget))
print('Total budget corresponds to %.1f full function evaluations.'%(sum([r.budget for r in all_runs])/args.max_budget))
print('The run took  %.1f seconds to complete.'%(all_runs[-1].time_stamps['finished'] - all_runs[0].time_stamps['started']))


"""
print and save the data
"""
print('id2config',id2config)
pkl_filename = "id2config.pkl"
with open(pkl_filename, 'wb') as file:
	pickle.dump(id2config, file)


print('incumbent',incumbent)
pkl_filename = "incumbent.pkl"
with open(pkl_filename, 'wb') as file:
	pickle.dump(incumbent, file)


print('traj',traj)
pkl_filename = "traj.pkl"
with open(pkl_filename, 'wb') as file:
	pickle.dump(traj, file)


print('all_runs',all_runs)
pkl_filename = "all_runs.pkl"
with open(pkl_filename, 'wb') as file:
	pickle.dump(all_runs, file)


print('curves',curves)
pkl_filename = "curves.pkl"
with open(pkl_filename, 'wb') as file:
	pickle.dump(curves, file)



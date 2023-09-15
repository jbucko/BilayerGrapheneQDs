import sklearn.neural_network as NN
import logging
logging.basicConfig(level=logging.WARNING)
import argparse
import numpy as np
import pickle



parser = argparse.ArgumentParser()

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

"""
general settings
"""
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
data loading and processing
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


data_corr = np.zeros((data_x.shape[0],2*data_x.shape[1] - 1))

for i in range(data_x.shape[0]):
	data_corr[i,:] = np.correlate(data_x[i],data_x[i],mode = 'full')


if correlations:
	data_x_merged = np.concatenate((data_x,data_corr),axis = 1)
else:
	data_x_merged = data_x

data = np.concatenate((data_x_merged,np.reshape(data_y,(-1,1))),axis = 1)
np.random.shuffle(data)
print(data.shape)


x_train = data[:,:data_x_merged.shape[1]]
y_train_temp = data[:,data_x_merged.shape[1]:]


"""
transform label to 0-4
"""
y_train = np.zeros(x_train.shape[0])
for i in range(len(y_train)):
	if y_train_temp[i]<0:
		y_train[i] = (abs(y_train_temp[i,0])+2)
	else:
		y_train[i] = y_train_temp[i]


######################---count and split dataset ----####################################
#
print(x_train.shape,y_train.shape)

u, yc = np.unique(y_train, return_counts=True)
testcounts = dict(zip(u,yc))
print('final dataset',testcounts)

split1 = int(x_train.shape[0]*0.8)
split2 = int(x_train.shape[0]*0.9)

# for i in range(len(y_train)):
# 	print(y_train[i])

x_test, y_test = x_train[split2:],y_train[split2:]
x_validate, y_validate = x_train[split1:split2],y_train[split1:split2]
x_train, y_train = x_train[:split1],y_train[:split1]
########################----define classifier and train---#################

# define classifier
clf = NN.MLPClassifier(verbose = 1,learning_rate = 'constant',hidden_layer_sizes = (800,800),max_iter = 60,n_iter_no_change=30,\
                early_stopping = True, alpha = 3.077287226244102e-07, random_state = args.randomstate,learning_rate_init = 0.018174243713252908)

clf.fit(x_train,y_train)

print(clf.validation_scores_)
y_pred = clf.predict(x_test)


score = clf.score(x_test,y_test)
print(score)

u, yc = np.unique(y_test, return_counts=True)
testcounts = dict(zip(u,yc))
print('y_test:',testcounts)

u, yc = np.unique(y_pred, return_counts=True)
testcounts = dict(zip(u,yc))
print('y_pred:',testcounts)

#save data
np.savetxt('ytest_{}.txt'.format(args.randomstate),y_test)
np.savetxt('ypred_{}.txt'.format(args.randomstate),y_pred)

#store model
pkl_filename = "MLPClassifier_2layers_bohb_optim_{}.pkl".format(args.randomstate)
with open(pkl_filename, 'wb') as file:
	pickle.dump(clf, file)


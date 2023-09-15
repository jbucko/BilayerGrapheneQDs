import torch
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import DataLoader,random_split
import numpy as np
#import folder_label_from_name_separated as read_data
import sys
from sklearn.decomposition import PCA,KernelPCA 
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE


"""read data"""

"""
path to data
"""
PATH_DATA = "./train/"

"""
reading the data
we read single component
"""
component = 'red' #('green','blue')

from PIL import Image
import glob
image_list = []
mt_list = []
r = []

print('reading data...')
for filename in glob.glob(PATH_DATA +'*.jpg'): #assuming gif
	im=Image.open(filename)
	im1 = im.copy()
	im1 = np.array(im1)
	"""
	extracting m and tau from file name and appending to label array mt_list
	"""
	label = filename.split('/')[-1][:-4]
	label = label.split('_')
	if label[9]=='1':
		mt_list.append([int(label[1]),1])
	else:
		mt_list.append([int(label[1]),-1])
	if component == 'red':
		r.append(im1[:,:,0])
	elif component == 'green':
		r.append(im1[:,:,1])
	elif component == 'blue':
		r.append(im1[:,:,2])
	im.close()


print("data reading finished...")
memory_limit = 4410 # number of training instrances


dataset = np.zeros(shape = (memory_limit,360*360))



for i in range(memory_limit):
	dataset[i] = np.reshape(r[i],(1,360*360))

#scale from 0 to 1
dataset/=255
# dataset = dataset[:,:20]

print(dataset.shape)

"""
data scaling
"""
scaler = StandardScaler()
scaler.fit(dataset)
scaler.transform(dataset)
print('transform finished...')

###########---tSNE---#############
n_components = 2

tsne = TSNE(n_components = n_components)
dataset = tsne.fit_transform(dataset)

# save components and labels
np.savetxt('tsne_two_principal.txt',dataset)
np.savetxt('tsne_labels.txt',mt_list)
print('number of iterations to converge: ',tsne.n_iter_)
print('KL_divergence value: ',tsne.kl_divergence_)

#########---store quantities for future prediction---##########
scaler_data_ = np.array([scaler.mean_,scaler.var_])
np.save("tSNE_scaler.npy",scaler_data_)
np.save("embedding_vectors.npy",tsne.embedding_)

print(scaler_data_,'vectors:',tsne.embedding_)

import torch
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import DataLoader,random_split,Subset
from itertools import accumulate
import numpy as np
import sys

def random_split_my(dataset, lengths):
	r"""
	Randomly split a dataset into non-overlapping new datasets of given lengths.

	Arguments:
		dataset (Dataset): Dataset to be split
		lengths (sequence): lengths of splits to be produced
	"""
	if sum(lengths) != len(dataset):
		raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

	indices = torch.randperm(sum(lengths)).tolist()
	return [Subset(dataset, indices[offset - length:offset]) for offset, length in zip(accumulate(lengths), lengths)],indices

"""
general settings
"""

l_rate = .001
n_epochs = 150
batch_size = 100

"""read data"""
which_data = "m_0_tau_1" #should specify a folder with the dataset containing only relevant m and tau maps
PATH_DATA = "./"+which_data+'/' # complete path to dataset
STORE_MODEL_PATH = "./" # where to store the trained model

"""
data readining and splitting
"""
data = torchvision.datasets.ImageFolder_label_from_name_separated(root = PATH_DATA,transform = transforms.ToTensor())
[data_train,data_validation,data_test],indices = random_split_my(data,(int(0.8*data.__len__()),int(0.1*data.__len__()),data.__len__()-int(0.8*data.__len__())-int(0.1*data.__len__())))
test_indices = indices[-(data.__len__()-int(0.8*data.__len__())-int(0.1*data.__len__())):]
np.savetxt('test_indices.txt',test_indices)

"""
saves names of test instances
"""
test = open('test_instances.txt','w')
for j in test_indices:
	#print(data.imgs[j][0],j)
	test.write(data.imgs[j][0]+'\n')
test.close()


"""
loading of training and validation data into an iterative structure
"""
data_train_loader = DataLoader(dataset = data_train,shuffle = True,batch_size = batch_size)
data_validation_loader = DataLoader(dataset = data_validation,batch_size = 1)



"""how to print and image"""
x,y = data[0]

print('labels of the first instance:',data[0][-1])
print('format of the data and label:',x.shape,y.shape)
# plt.imshow(x.permute(1,2,0))
# plt.show()


"""
CNN definition
"""
class ConvNet(nn.Module):
	def __init__(self):
		super(ConvNet,self).__init__()
		self.layer1 = nn.Sequential(
				nn.Conv2d(3,16,kernel_size = (5,5),stride = 1,padding = (2,2)),
				nn.ReLU(),
				nn.MaxPool2d(kernel_size = (5,5),stride = (5,5))
				)
		self.layer2 = nn.Sequential(
				nn.Conv2d(16,32,kernel_size = (3,3),stride = 1,padding = (1,1)),
				nn.ReLU(),
				nn.MaxPool2d(kernel_size = (3,3),stride = (3,3))
				)
		self.drop_out = nn.Dropout()
		self.fc1 = nn.Linear(360*360//3//3//5//5*32,250)
		self.fc2 = nn.Linear(250,50)
		self.fc3 = nn.Linear(50,2)
		self.relu =  torch.nn.ReLU(inplace=True)

	def forward(self,x):
		out = self.layer1(x)
		out = self.layer2(out)
		out = out.reshape(out.size(0),-1)
		out = self.drop_out(out)
		out = self.fc1(out)
		out = self.relu(out)
		out = self.fc2(out)
		out = self.relu(out)
		out = self.fc3(out)
		return out

model = ConvNet()

# loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr = l_rate)

total_step = len(data_train_loader)
print('total steps:',total_step)

loss_list = []
loss_global = []
acc_list = []
val_loss_list = []


"""training"""
for epoch in range(n_epochs):
	for i,(images,labels) in enumerate(data_train_loader):
		model.train()
		#print(images.shape)
		#print(labels.shape)
		image1 = images[0,:,:,:]
		#print(labels)
		#print(image1.shape)
		outputs = model(images)
		#print(outputs.data)
		#print(labels.type())
		labels = labels.float()
		loss = criterion(outputs,labels)
		print('epoch {}/{}, batch {}/{}, loss: {}'.format(epoch+1,n_epochs,i+1,total_step,loss.item()))

		loss_list.append(loss.item())

		# Backprop and perform Adam optimisation
		optimizer.zero_grad() # initialize gradients to zero before backproparagtion
		loss.backward()
		optimizer.step()

		# Track the accuracy
		total = labels.size(0)


	# validation accuracy
	val_loss = []
	for i,(images,labels) in enumerate(data_validation_loader):
		model.eval()
		outputs = model(images)
		val_loss.append(criterion(labels,outputs).item())
		#print(outputs,labels)

	val_loss_list.append(np.mean(val_loss))
	loss_global.append(np.mean(loss_list[-total_step:]))
	print('After epoch [{}/{}]: training loss: {:.4f}, validation loss: {:.4f}'.format(epoch+1,n_epochs,loss_global[-1],val_loss_list[-1]))

# Save the model learning curves
torch.save(model.state_dict(), STORE_MODEL_PATH + 'supervised_CNN_'+which_data+'.ckpt')
np.savetxt('training_'+which_data+'.txt',loss_global)
np.savetxt('validation_'+which_data+'.txt',val_loss_list)






import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
from torch.utils.data import Dataset
from torch.autograd import Variable

workspace_dir = "/mnt/d/Documents/gan/MNIST"

input_size = 4
N_training = 10000
hidden_sizes = 10
output_size = 2
learning_rate = 0.003
momentum = 0.9
epochs = 15
cardinality = 1000

mean = 4.0
sigma = 1.25

activation_function = torch.sigmoid
criterion = nn.BCELoss()

def get_distribution_sampler(mu, sigma):
	return lambda n: torch.Tensor(np.random.normal(mu, sigma, (1,n)))

def get_generator_input_sampler():
	return lambda m, n: torch.rand(m, n)  # Uniform-dist data into generator, _NOT_ Gaussian

def get_moments(d):
    # Return the first 4 moments of the data provided
    mean = torch.mean(d)
    diffs = d - mean
    var = torch.mean(torch.pow(diffs, 2.0))
    std = torch.pow(var, 0.5)
    zscores = diffs / std
    skews = torch.mean(torch.pow(zscores, 3.0))
    kurtoses = torch.mean(torch.pow(zscores, 4.0)) - 3.0  # excess kurtosis, should be 0 for Gaussian
    final = torch.cat((mean.reshape(1,), std.reshape(1,), skews.reshape(1,), kurtoses.reshape(1,)))
    return final

class gauss_and_random_dataset(Dataset):
	def __init__(self, gauss_sampler, random_sampler, N, cardinality):
		self.samples = []
		g_sampler = gauss_sampler(mean, sigma)
		r_sampler = random_sampler()
		
		for k in range(N):
			this_g = g_sampler(cardinality)
			self.samples.append(this_g)
			this_r = r_sampler(cardinality, 1)
			self.samples.append(this_r)
			
	def __len__(self):
		return len(self.samples)
		
	def __getitem__(self, idx):
		return self.samples[idx]
		
trainingset = gauss_and_random_dataset(get_distribution_sampler, get_generator_input_sampler, N_training, cardinality)
validationset=gauss_and_random_dataset(get_distribution_sampler, get_generator_input_sampler, 2000, cardinality)

class Discriminator(nn.Module):
	def __init__(self, input_size, hidden_sizes, output_size, f):
		super(Discriminator, self).__init__()
		self.map1 = nn.Linear(input_size, hidden_sizes)
		self.map2 = nn.Linear(hidden_sizes, hidden_sizes)
		self.map3 = nn.Linear(hidden_sizes, output_size)
		self.f = f

	def forward(self, x):
		x = self.f(self.map1(x))
		x = self.f(self.map2(x))
		return self.f(self.map3(x))
		
def train():		
	d = Discriminator(input_size = input_size,
							hidden_sizes = hidden_sizes,
							output_size = output_size,
							f = activation_function)
							
	optimizer = optim.SGD(d.parameters(), lr=learning_rate, momentum=momentum)
	
	train_data = gauss_and_random_dataset(get_distribution_sampler, get_generator_input_sampler, N_training,cardinality)
	
	time0 = time()
	print("Beginning Training!")
	for epoch in range(epochs):
		running_loss = 0
		print("We are in epoch {}".format(epoch))
		for data in range(len(train_data)):
			d.zero_grad()
			
			moments = get_moments(train_data[data])
			output = d(moments) 
			
			labels = Variable(torch.tensor([int((data+1)%2),int((data)%2)])).float()
			
			loss = criterion(output, labels)
			
			loss.backward()
			
			optimizer.step()
			
			running_loss += loss.item()
	print("\nTraining Time (in minutes) = ",(time()-time0)/60)
	torch.save({'state_dict': d.state_dict()}, 'gaussian_discriminator.pt')
	
def validate():

	validation_data = gauss_and_random_dataset(get_distribution_sampler, get_generator_input_sampler, 2000, cardinality)

	correct_count, all_count = 0, 0

	d = Discriminator(input_size = input_size,
							hidden_sizes = hidden_sizes,
							output_size = output_size,
							f = activation_function)

	d_state_dict = torch.load('gaussian_discriminator.pt')['state_dict']
	d.load_state_dict(d_state_dict)
	
	for data in range(len(validation_data)):
		moments = get_moments(validation_data[data])
		with torch.no_grad():
			logps = d(moments)
		ps = torch.exp(logps)
		probab = list(ps.numpy())
		pred_label = probab.index(max(probab))
		
		if data%2 == 0:
			true_label_index = 0
		else:
			true_label_index = 1
		if(true_label_index == pred_label):
			correct_count += 1
		all_count += 1
			
	print("Number Of Images Tested =", all_count)
	print("\nModel Accuracy =", (correct_count/all_count))
	
def main():
	train()
	validate()
	
if __name__ == "__main__":
	main()

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from libs.mf_reg_common_a_b import *


### Definition of the model parameters ###
seed = 500  # seed
num = 6400  # number of elements
L = 6  # number of layers
dim = 500  # embedding size
numOfEpochs = 100  # number of epocs
learning_rate = 1e-1  # learning rate
l2_reg_coeff = 1e-2  # regularization term coefficient for l2
##########################################

# Set the seed value
np.random.seed(seed)

# Construct the input matrix
#M = np.random.randint(low=0, high=2, size=(L, num, num))

'''
M = np.zeros(shape=(L, num, num))
M = np.zeros(shape=(L, num, num))
for i in range(num):
  for j in range(i+1, num):
    M[0, i, j] = 1
    M[1, j, i] = 1
'''
M = np.random.randint(low=0, high=2, size=(L, num, num) )  #M = np.load('yeast_string_deepwalkmat.npy')
#M = sparse.csr_matrix(M)
print(M.shape)


# Convert the input matrix to tensor
M = torch.from_numpy(M)

# Define the model, loss function and the optimizer
model = MatrixFactorization(dim=dim, num=num)
loss_func = torch.nn.MSELoss()
#optimizer = torch.optim.ASGD(model.parameters(), lr=learning_rate, weight_decay=l2_reg_coeff)
optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate, weight_decay=l2_reg_coeff)

# Train the model
lossList = train(M, numOfEpochs, model, optimizer, loss_func)
'''
# Plot the learning curve
plt.figure()
plt.plot(range(numOfEpochs), lossList, 'r.')
plt.xlabel('Number of epochs')
plt.ylabel('Loss')
plt.show()

idxList = torch.tensor([ [l, i, j] for l in range(M.shape[0]) for i in range(M.shape[1]) for j in range(M.shape[2]) ])
currentLoss = computeCurrentLoss(loss_func, M, idxList, model)
print("Total Loss:" + str(currentLoss))

# Print the reconstructed matrix
modelParams = list(model.parameters())
A_pred = modelParams[0].detach().numpy()
B_pred = modelParams[1].detach().numpy()

np.save('yeast_deepwalkmat_mlne_common_a_b_500d.emb',A_pred,allow_pickle =False)

#C = np.dot(A_pred, B_pred.T)
#print(C)

'''
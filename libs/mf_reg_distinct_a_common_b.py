import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
import numpy as np

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))


# Definition of the class
class MatrixFactorization(torch.nn.Module):
    def __init__(self, num, dim, numOfLayers):
        super().__init__()
        self._num = num
        self._A = torch.nn.Embedding(num_embeddings=numOfLayers * self._num, embedding_dim=dim, sparse=False)
        self._B = torch.nn.Embedding(num_embeddings=self._num, embedding_dim=dim, sparse=False)

    def forward(self, layerIdx, aIdx, bIdx):
        return (self._A(layerIdx * self._num + aIdx) * self._B(bIdx)).sum(dim=1)


# Compute the current loss
def computeCurrentLoss(loss_func, M, model):
    loss = 0
    for layerIdx in range(M.shape[0]):
        for rowIdx in range(M.shape[1]):
            for columnIdx in range(M.shape[2]):
                layerIdx = torch.LongTensor([layerIdx])
                aIdx = torch.LongTensor([rowIdx])
                bIdx = torch.LongTensor([columnIdx])
                predictedValue = model(layerIdx, aIdx, bIdx)
                correctValue = torch.FloatTensor([M[layerIdx, rowIdx, columnIdx]])
                loss += loss_func(predictedValue, correctValue)

    return loss


# Train the model
def train(M, numOfEpochs, model, optimizer, loss_func):

    '''
    # Define the index list and shuffle the data
    idxList = torch.tensor([[l, i, j] for l in range(M.shape[0]) for i in range(M.shape[1]) for j in range(M.shape[2])])
    perm = np.random.permutation(len(idxList))
    idxList = idxList[perm]
    '''
    layer_indices = np.arange(M.shape[0])
    row_indices = np.arange(M.shape[1])
    column_indices = np.arange(M.shape[2])

    lossList = []
    for e in range(numOfEpochs):

        np.random.shuffle(layer_indices)
        np.random.shuffle(row_indices)
        np.random.shuffle(column_indices)

        for layerIdx in layer_indices:
            for rowIdx in row_indices:
                for colIdx in column_indices:
                    # Set gradients to zero
                    optimizer.zero_grad()

                    # Turn data into tensors
                    correctValue = torch.FloatTensor([M[layerIdx, rowIdx, colIdx]])
                    layerIdx = torch.LongTensor([layerIdx])
                    aIdx = torch.LongTensor([rowIdx])
                    bIdx = torch.LongTensor([colIdx])

                    # Predict and calculate loss
                    predictedValue = model(layerIdx, aIdx, bIdx)
                    loss = loss_func(predictedValue, correctValue)

                    # Backpropagate
                    loss.backward()

                    # Update the parameters
                    optimizer.step()

        currentLoss = computeCurrentLoss(loss_func, M, model)
        lossList.append(currentLoss)

        # print(f"Epoch {e+1}\n-------------------------------")
        # print(f"loss: {loss.item():>7f}")

    return lossList


if __name__ == "__main__":
    ### Definition of the model parameters ###
    seed = 500  # seed
    num = 4  # number of elements
    L = 3  # number of layers
    dim = 4  # embedding size
    numOfEpochs = 100  # number of epocs
    learning_rate = 1e-1  # learning rate
    l2_reg_coeff = 1e-2  # regularization term coefficient for l2
    ##########################################

    # Set the seed value
    np.random.seed(seed)

    # Construct the input matrix
    M = np.random.randint(low=0, high=10, size=(L, num, num))

    # Convert the input matrix to tensor
    M = torch.from_numpy(M)

    # Define the model, loss function and the optimizer
    model = MatrixFactorization(num, dim=dim, numOfLayers=L)
    loss_func = torch.nn.MSELoss()
    # optimizer = torch.optim.ASGD(model.parameters(), lr=learning_rate, weight_decay=l2_reg_coeff)
    optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate, weight_decay=l2_reg_coeff)

    # Train the model
    lossList = train(M, numOfEpochs, model, optimizer, loss_func)

    # Plot the learning curve
    plt.figure()
    plt.plot(range(numOfEpochs), lossList, 'r.')
    plt.xlabel('Number of epochs')
    plt.ylabel('Loss')
    plt.show()

    # Print the input matrix
    print(M)

    idxList = torch.tensor([[l, i, j] for l in range(M.shape[0]) for i in range(M.shape[1]) for j in range(M.shape[2])])
    currentLoss = computeCurrentLoss(loss_func, M, model)
    print(f"Total Loss: {currentLoss}")

    # Print the reconstructed matrix
    modelParams = list(model.parameters())
    A_pred = modelParams[0].detach().numpy()
    B_pred = modelParams[1].detach().numpy()

    print("------")
    print(np.dot(A_pred, B_pred.T), )


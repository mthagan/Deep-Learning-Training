# --------------------------------------------------------------------------------------------
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

#----------------------------------------------------------------------------
S1 = 20               # Number of neurons in the hidden layer
S2 = 10               # Number of outputs
num_epochs = 1       # Number of epochs
learning_rate = 0.1   # Learning rate
#----------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------
# download data from MNIST and create mini-batch data loader
torch.manual_seed(1122)

trainset = torchvision.datasets.MNIST(root='./mnist', train=True, download=True, transform=transforms.ToTensor())

trainloader = torch.utils.data.DataLoader(trainset, batch_size=250, shuffle=True)
# --------------------------------------------------------------------------------------------
testset = torchvision.datasets.MNIST(root='./mnist', train=False, download=True, transform=transforms.ToTensor())

testloader = torch.utils.data.DataLoader(testset, batch_size=250, shuffle=True)
# --------------------------------------------------------------------------------------------

# define and initialize a multilayer-perceptron, a criterion, and an optimizer
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.l1 = nn.Linear(1 * 28 * 28, S1)
        self.t1 = nn.Tanh()
        self.l2 = nn.Linear(S1, S2)
        self.t2 = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = x.view(-1, 1 * 28 * 28)
        x = self.t1(self.l1(x))
        x = self.t2(self.l2(x))
        return x
# --------------------------------------------------------------------------------------------
mlp = MLP()
criterion = nn.NLLLoss()
optimizer = optim.SGD(mlp.parameters(), lr=learning_rate, momentum=0.9)
# --------------------------------------------------------------------------------------------

# define a training epoch function
def trainEpoch(dataloader, epoch):
    print("Training Epoch %i" % (epoch + 1))
    mlp.train()
    running_loss = 0
    for i, data in enumerate(dataloader, 0):
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)
        optimizer.zero_grad()
        outputs = mlp(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.data
        if (i + 1) % 50 == 0:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 50))
            running_loss = 0.0
# --------------------------------------------------------------------------------------------

# define a testing function
def testModel(dataloader):
    mlp.eval()
    test_loss = 0
    correct = 0
    for inputs, targets in dataloader:
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = mlp(inputs)
        test_loss += F.nll_loss(outputs, targets, size_average=False).data[0]
        pred = outputs.data.max(1, keepdim=True)[1]
        correct += pred.eq(targets.data.view_as(pred)).sum()
    test_loss /= len(dataloader.dataset)
    print('Test set: Average loss: {:.3f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
          test_loss, correct, len(testloader.dataset),
          100. * correct / len(testloader.dataset)))

#--------------------------------------------------------------------------------------------

# Show results for one test image

def showOneTestResult(dataloader, b_number, testnumber):
    for i_batch, sample_batched in enumerate(dataloader):
        if i_batch == b_number:
            image1 = sample_batched[0]
            image1 = image1[testnumber,:,:,:]
            tar = sample_batched[1]
            tar = tar[testnumber]
            output = mlp(image1)
            image1 = image1.data.numpy()
            image1 = image1.squeeze()
            plt.imshow(image1)
            plt.show()
            output = output.data.numpy()
            output = np.exp(output)  # convert from logsoftmax to softmax
            print('The network output for item ' + np.str(testnumber) + ' in batch ' + np.str(i_batch) + ' is:')
            print(output)
            print('The target is')
            print(tar)



# --------------------------------------------------------------------------------------------
# run the training epoch 30 times and test the result
for epoch in range(num_epochs):
    trainEpoch(trainloader, epoch)

testModel(testloader)

# Show one example image, network output and target value
batch_view = 0  # Which batch to take the example
number_view = 0 # Which number of the batch to take the example
showOneTestResult(trainloader, batch_view, number_view)

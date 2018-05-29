#----------------------------------------------------------------------------
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import matplotlib.pyplot as plt
#----------------------------------------------------------------------------
R = 1                 # Input size
S1 = 10               # Number of neurons in the hidden layer
S2 = 1                # Network output size
num_epochs = 100000   # Number of epochs
learning_rate = 0.1  # Learning rate
#----------------------------------------------------------------------------

# Check for gpu
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

print('The device is a')
print(device)

# Create the data set - inputs and targets
inputs1 = np.linspace(-3,3,20, dtype=np.float32).reshape(-1,1)
targets1 = np.sin(inputs1,  dtype=np.float32).reshape(-1,1)

# Convert the NumPy ndarrays to Torch tensor variables
p = Variable(torch.from_numpy(inputs1))
t = Variable(torch.from_numpy(targets1))

p = p.to(device)
t = t.to(device)

# Create the network object
model = torch.nn.Sequential(
    torch.nn.Linear(R, S1),
    torch.nn.Tanh(),
    torch.nn.Linear(S1, S2),
)

model = model.to(device)

# Define the loss function (performance index) to mean square error
criterion = nn.MSELoss()

# Set the optimizer to stochastic gradient
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Array to save the loss for each iteration
total_loss = []

# Main training loop
for epoch in range(num_epochs):

    # Forward + Backward + Optimize
    optimizer.zero_grad()
    outputs = model(p)
    loss = criterion(outputs, t)
    loss.backward()
    optimizer.step()
    # Save loss for plotting
    total_loss.append(loss.data)

    if (epoch + 1) % 2000 == 0:
        print('Epoch [%d/%d], Loss: %.4f' % (epoch + 1, num_epochs, loss.data))

# Plot loss versus epoch number
plt.subplot(1, 2, 1)
plt.xlabel("Number of Epochs")
plt.ylabel("Sum Squared Error")
plt.loglog(total_loss, label="Training error")
plt.grid(True)
plt.legend()

# Calculate network output at finer spacing
inputs2 = np.linspace(-3,3,200, dtype=np.float32).reshape(-1,1)
p2 = Variable(torch.from_numpy(inputs2))
p2 = p2.to(device)
zz = model(p2)

# Pull data back into NumPy array on cpu for speed
zz1 = zz.data.cpu().numpy()

# Plot targets and network output at finer spacing
plt.subplot(1, 2, 2)
plt.scatter(inputs1, targets1, c='Red', label="Targets")
plt.plot(inputs2, zz1, label="Network response")
plt.grid(True)
plt.legend()

plt.show()


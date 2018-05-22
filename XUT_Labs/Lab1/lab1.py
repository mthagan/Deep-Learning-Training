# Solving Question 3 from Lab1 1 with Python
import matplotlib.pyplot as plt

import lab1module

# Main program

# Define the inputs and targets
p = [-1, 0, 1]
t = [-1.5, 0.0, 2.5]

# Assign the learning rate
lrn_rate0 = 0.1

# Create network object
net = lab1module.network(p, t)

# Train the network object
net.train_network(lrn_rate=lrn_rate0)

# Plot sum squared error versus epoch
plt.subplot(1, 2, 1)
plt.xlabel("Number of Epochs")
plt.ylabel("Sum Squared Error")
plt.plot(net.arrF, label="Training error")
plt.grid(True)
plt.legend()

# Create inputs to test the network response
test_p = []
test_a = []

for i in range(30):
    test_p.append(-1.5 + 0.1*i)
    test_a.append(net.weights*test_p[i] + net.bias)

# Plot network response vs inputs
plt.subplot(1, 2, 2)
plt.xlabel("Inputs")
plt.ylabel("Outputs")
plt.plot(test_p, test_a, "-g", label="Network response")
plt.scatter(p, t, label="Targets", c='red', marker='+', s=200)
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

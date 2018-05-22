import math


class network:
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

        self.weights = 0
        self.bias = 0

        self.arrF = []

    def train_network(self, lrn_rate):
        epoch = 0
        max_epoch = 50

        print 'Iteration = ' + str(epoch)
        print 'Weight = ' + str(self.weights)
        print 'Bias = ' + str(self.bias)

        while epoch < max_epoch:
            epoch = epoch + 1
            f = 0
            sum1 = 0
            sum2 = 0

            for i in range(len(self.inputs)):
                # Compute sum squared error
                f += (self.targets[i] - self.weights*self.inputs[i] - self.bias)**2

                # Derivative of F with respect to w
                sum1 += (self.targets[i] - self.weights*self.inputs[i] - self.bias)*self.inputs[i]

                # Derivative of F with respect to b
                sum2 += (self.targets[i] - self.weights*self.inputs[i] - self.bias)

            self.arrF.append(f)
            df_dw = -2*sum1
            df_db = -2*sum2

            # Compute the magnitude of the gradient of F
            mag_f = math.sqrt(df_dw**2 + df_db**2)

            # Update weight and bias
            self.weights = self.weights - lrn_rate*df_dw
            self.bias = self.bias - lrn_rate*df_db

            # Stop if gradient magnitude is small
            if mag_f < 0.01:
                break

            print 'Epoch = ' + str(epoch)
            print 'Weight = ' + str(self.weights)
            print 'Bias = ' + str(self.bias)

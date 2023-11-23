import numpy as np
import pandas as pd

class LinearRegression():
    def __init__(self, x, y):
        self.x = np.append(x, np.ones((x.shape[0], 1)),axis=1) # add ones column to capture intercept
        self.y = y
        # initialize weights randomly
        self.weights = np.random.randn(self.x.shape[1], 1)
    
    def train(self, learning_rate):
        print("Beginning training process...")
        opt_weights = self.weights
        alpha = learning_rate
        m = len(self.y)
        
        # gradient descent algorithm
        t = 0 # iteration tracker
        max_iters = 1000
        while t < max_iters:
            t += 1
            y_pred = self.predict(self.x, opt_weights) # current prediction for current weights
            xt = np.transpose(self.x)
            err = y_pred-self.y
            opt_weights = opt_weights - (alpha / m) *  xt@err # update based on gradient of loss fcn
        self.weights = opt_weights
    
    # generate prediction for given weights, used in training
    @staticmethod
    def predict(x,weights):
        return x@weights

    # make forecasts based on optimal weights
    def forecast(self):
        pass

    def r2(self):
        pass


def main():
    x = np.linspace(0,10,num=10).reshape(10,1)
    y = 5*x+17
    lr = LinearRegression(x,y)
    lr.train(learning_rate=0.01)
    print(lr.weights)


if __name__ == "__main__":
    main()
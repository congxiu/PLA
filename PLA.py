'''
An implementation of the Perceptron

Assumes data to be in panda dataframe format in which the
last column is the output, and the other columns are features
Constant column is automatically appened

Also requires numpy to function

Initial weights can be set by users, otherwise it's all 0's

Three version of learning algorithms are implemented

1:Deterministic PLA
2:Randomized PLA
3:Pocket PLA

Set flag = 1, 2, 3 to train different algos
In the case when data is not linear seperable, algo 1 and algo 2 is going to
terminate when the #updates >= size of data

Make sure to normalize data before training the model

Example:
p = Perceptron(data)
p.train(flag = 1, eta = 1.0)
p.predict(newData)
p.score(testSample, testLabel)
'''
import pandas as pd
import numpy as np

class Perceptron:
    def __init__(self, data, initialWeights = None):
        self.length = len(data)
        self.sample = data[data.columns[:-1]].join(pd.DataFrame({'const':[1]*self.length})).as_matrix()
        self.label = data.iloc[:, -1].as_matrix()
        if initialWeights:
            self.weights = initialWeights
        else:
            self.weights = np.array([0] * len(data.columns))

    def predict(self, ins):
        result = ins.dot(self.weights)
        if result.size > 1:
            result[result > 0] = 1
            result[result <= 0] = -1
        else:
            result = 1 if result > 0 else -1
        return result

    def score(self, ins, ots):
        return np.mean((self.predict(ins) * ots) > 0)
        
    def printModel(self):
        print self.weights
        
    def train(self, flag = 1, eta = 1.0, seed = 91):
        if flag == 1:
            self.trainPLA(eta)
        elif flag == 2:
            self.trainRandomizedPLA(eta, seed)
        elif flag == 3:
            self.trainPocketPLA(eta, seed)
        else:
            print "Invalid flag. Flag must be 1, 2, or 3"

    def trainPLA(self, eta = 1.0):
        idx = 0
        mistake = 0
        self.counter = 0
        while self.counter <= self.length:
            pred = self.predict(self.sample[idx]) * self.label[idx]
            if pred < 0:
                self.counter += 1
                self.weights = self.weights + eta * self.sample[idx] * self.label[idx]
                mistake = idx
                
            idx += 1
            if (idx >= self.length):
                idx = 0
                
            if idx == mistake:
                pred = self.predict(self.sample[idx]) * self.label[idx]
                if pred > 0:
                    print "Training complete, data is linear seperable!"
                    print "Total number of updates is", self.counter
                    break

        print "Accuracy is", self.score(self.sample, self.label)
        
    def trainRandomizedPLA(self, eta = 1.0, seed = 73):
        self.counter = 0
        np.random.seed(seed)
        indices = np.random.permutation(np.arange(self.length))
        if self.score(self.sample, self.label) == 1.0:
            perfect = True
        else:
            perfect = False
        
        while not perfect and self.counter <= self.length:
            perfect = True
            for idx in indices:
                pred = self.predict(self.sample[idx]) * self.label[idx]
                if pred < 0:
                    self.counter += 1
                    self.weights = self.weights + eta * self.sample[idx] * self.label[idx]
                    perfect = False
                    
        if perfect:
            print "Training complete, data is linear seperable!"
            print "Total number of updates is", self.counter
            
        print "Accuracy is", self.score(self.sample, self.label)
    
    def trainPocketPLA(self, eta = 1.0, seed = 83):
        self.counter = 0
        np.random.seed(seed)
        bestWeights = self.weights
        bestScore = self.score(self.sample, self.label)
        if bestScore == 1.0:
            perfect = True
        else:
            perfect = False
        
        while not perfect and self.counter <= 100:
            pred = 1
            while pred > 0:
                idx = np.random.randint(0, self.length)
                pred = self.predict(self.sample[idx]) * self.label[idx]
                
            self.weights = self.weights + eta * self.sample[idx] * self.label[idx]
            self.counter += 1
            newScore = self.score(self.sample, self.label)
            
            if newScore > bestScore:
                bestWeights = self.weights
                bestScore = newScore
                if newScore == 1.0:
                    perfect = True
                    
        if perfect:
            print "Training complete, data is linear seperable!"
            print "Total number of updates is", self.counter
            
        self.weights = bestWeights
        print "Accuracy is", bestScore
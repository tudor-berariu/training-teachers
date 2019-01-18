import numpy as np
import matplotlib.pyplot as plt


class Model(object):

    def __init__(self,dist):
        self.dist = dist

    def train(self, data, labels):
        raise NotImplementedError()

    def predict(self, data):
        raise NotImplementedError()

    def plot_decision(self,n_points=200):
        x = np.arange(-0.8,0.8,1/n_points)
        y = np.arange(-0.8,0.8,1/n_points)
        xx, yy = np.meshgrid(x,y)
        xxr = np.reshape(xx,(-1,1))
        yyr = np.reshape(yy,(-1,1))
        predictions = self.predict(np.concatenate((xxr,yyr),axis=-1))
        plt.scatter(xxr,yyr,c=predictions)

    def acc(self,data,labels):
        pred = self.predict(data)
        ret = np.equal(np.argmax(pred,axis=-1),np.argmax(labels,axis=-1))
        return np.mean(ret)
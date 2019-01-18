from .model import Model
import tensorflow as tf
import numpy as np
from .tools import kl,accuracy


class Lin():

    def __init__(self, layers, no_cls,name='lin'):
        self.name = name
        self.w = []
        self.b = []
        layers.append(no_cls)
        layers = [2]+layers
        with tf.variable_scope(self.name,reuse=tf.AUTO_REUSE):
            for i in range(len(layers)-1):
                self.w.append(tf.get_variable(f'w_{i}',[layers[i],layers[i+1]],
                            initializer=tf.initializers.random_normal()))
                self.b.append(tf.get_variable(f'b_{i}',[layers[i+1]],
                            initializer=tf.initializers.zeros()))

    def __call__(self,x):
        for i in range(len(self.w)):
            x = x@self.w[i]+self.b[i]
            if i!=len(self.w)-1:
                x = tf.tanh(x)
            else:
                x = tf.nn.softmax(x)
        return x

    @property
    def vars(self):
        ret = []
        ret.extend(self.w)
        ret.extend(self.b)
        return ret



class MLP(Model):

    def __init__(self,session,dist,name='mlp',layers=[100,100,100]):
        super().__init__(dist)
        self.s = session
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            self.model = Lin(layers,dist.no_cls,name='_lin')
            self.vars = self.model.vars
            self.data_ = tf.placeholder(tf.float32,[None,2])
            self.labels_ = tf.placeholder(tf.float32, [None,3])
            self.output = self.model(self.data_)
            self.accuracy = accuracy(self.output, self.labels_)
            self.loss = kl(self.labels_,self.output)
            self.optim = tf.train.MomentumOptimizer(1e-3,0.9)
            self.optim_step = self.optim.minimize(self.loss)
            self.vars.extend(self.optim.variables())
            self.reset = tf.initializers.variables(self.vars)

    def train(self, data, labels, val_data,val_labels,verbose=False):
        self.s.run(self.reset)
        n = len(data)
        best_loss = np.inf
        best_acc = 0
        for e in range(400):
            batch_size = max(256,len(data)//6)
            for it in range(n//batch_size+1):
                ind = np.random.choice(n,batch_size)
                batch_data = data[ind]
                batch_labels = labels[ind]
                _,loss,acc = self.s.run([self.optim_step, self.loss, 
                                            self.accuracy],
                                feed_dict={self.data_:batch_data,
                                            self.labels_:batch_labels})
                if verbose:
                    print(f'epoch[{e}] it[{it}] loss[{loss:.4f}] '
                            f'acc[{acc:.4f}]')
            loss,acc = self.s.run([self.loss,self.accuracy],
                                feed_dict={self.data_:val_data,
                                            self.labels_:val_labels})
            if loss<best_loss:
                best_loss = loss
                last_best = e
            if acc>best_acc:
                best_acc = acc
            if verbose:
                print(f'epoch[{e}] val/best_val loss[{loss:.4f}/{best_loss:.4f}] '
                    f'acc[{acc:.4f}/{best_acc:.4f}]')
            if e-last_best>6:
                break
        acc = self.s.run(self.accuracy, feed_dict={self.data_:data,
                                                self.labels_:labels})
        print(f'train_epochs[{e}] train_acc[{acc:.4f}] val_acc[{best_acc:.4f}]')



    def predict(self, data):
        return self.s.run(self.output, feed_dict={self.data_:data})
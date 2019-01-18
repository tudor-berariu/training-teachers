import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

WHITE = np.array([255,255,255,255])
COLOR_MARGIN = 100

def cdist(c1,c2):
  return np.linalg.norm(c1-c2)

def ceq(c1,c2):
  return cdist(c1,c2)<COLOR_MARGIN

def show(data,labels):
    plt.scatter(data[:,0],data[:,1],c=labels,edgecolor=[0,0,0],alpha=0.5)

def kl(d1,d2):
    # d1 = tf.distributions.Categorical(d1)
    # d2 = tf.distributions.Categorical(d2)
    # return tf.reduce_mean(tf.distributions.kl_divergence(d1,d2))
    return -tf.reduce_mean(tf.reduce_sum(d1*(tf.log(d2)),axis=-1))

def accuracy(x,y):
    ret = tf.equal(tf.argmax(x,axis=-1),tf.argmax(y,axis=-1))
    ret = tf.cast(ret, tf.float32)
    return tf.reduce_mean(ret)

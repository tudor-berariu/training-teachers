import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.distributions import Categorical
import scipy as sp



def covariance(x):
  mean = tf.reduce_mean(x,axis=0)
  x = x-mean
  var = tf.reduce_mean(tf.square(x),axis=0)
  return tf.transpose(x)@x/(var*100)


def save_img(data1, data2, path, n=100):
  if len(data1.shape)==2:
    l = int(np.sqrt(int(data1.shape[1])))
    data1 = np.reshape(data1, (-1,l,l,1))
    data2 = np.reshape(data2, (-1,l,l,1))
  else:
    data1=np.squeeze(data1,-1)
    data2=np.squeeze(data2,-1)
    l=int(data1.shape[1])
  data = np.concatenate((data1,data2),axis=-1)
  data = np.reshape(data[:n], [-1,2*l])
  sp.misc.toimage(data, cmin=0,cmax=1).save(path)

def kl(f, g, axis=-1):
  f_ = tf.nn.softmax(f)
  g_ = tf.nn.softmax(g)
  # return tf.reduce_sum(f_*(tf.log(f_)-tf.log(g_)),axis=axis)
  return tf.distributions.kl_divergence(Categorical(f_),Categorical(g_))

def simple_img(data, path, n=20):
  n=min(n,len(data))
  l=int(data.shape[1])
  data=np.squeeze(data,-1)
  data=np.reshape(data[:n],[-1,l])
  sp.misc.toimage(data, cmin=0,cmax=1).save(path)


class LinProj(object):

  def __init__(self, name, args):
    self.name=name
    self.sizes = [*args.in_size]+args.lin.layers+[args.out_size]
    self.w = []
    self.b = []
    self.args = args
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
      for i in range(len(self.sizes)-1):
        self.w.append(tf.get_variable(f'w_{i}',[self.sizes[i],self.sizes[i+1]],
          initializer=xavier_initializer()))
        self.b.append(tf.get_variable(f'b_{i}', [1,self.sizes[i+1]],
          initializer=tf.initializers.constant(0)))

  def __call__(self,x):
    with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
      for i,(w,b) in enumerate(zip(self.w,self.b)):
        x = tf.matmul(x,w)+b
        if i != len(self.w)-1:
          if self.args.activation == 'tanh':
            x = tf.tanh(x)
          elif self.args.activation == 'relu':
            x = tf.nn.relu(x) 
    return x

  @property
  def vars(self):
    ret = list(self.w)
    ret.extend(self.b)
    return ret


class ConvProj(object):

  def __init__(self,name, args):
    self.name=name
    self.w = []
    self.b = []
    self.args = args
    in_size=1
    self.layers = []
    self.layers.extend(args.conv.layers)
    self.layers.append([args.out_size])
    self.dropout = args.conv.dropout
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
      first = True
      for i,layer in enumerate(self.layers):
        if len(layer)==3:
          self.w.append(tf.get_variable(f'w_{i}',
                        [layer[0],layer[1],in_size,layer[2]],
                          initializer=xavier_initializer()))
        else:
          if len(layer)!=3 and first:
            first=False
            in_size = 7*7*32
          self.w.append(tf.get_variable(f'w_{i}',[in_size,layer[-1]],
                        initializer=xavier_initializer()))
        self.b.append(tf.get_variable(f'b_{i}', [1,layer[-1]],
                      initializer=tf.initializers.constant(0)))
        in_size = layer[-1]

  def __call__(self,x, training):
    first=True
    with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
      for i,(w,b) in enumerate(zip(self.w,self.b)):
        if len(w.shape)==4:
          x = tf.nn.conv2d(x,w,[1,1,1,1],'SAME')+b
          x = tf.nn.max_pool(x,[1,2,2,1],[1,2,2,1],'SAME')
        else:
          if first:
            first = False
            x = tf.layers.flatten(x)
          x = tf.matmul(x,w)+b
        if i != len(self.w)-1:
          x = tf.layers.dropout(x, self.dropout, training=training)
          if self.args.activation == 'tanh':
            x = tf.tanh(x)
          elif self.args.activation == 'relu':
            x = tf.nn.relu(x)
    return x 

  @property
  def vars(self):
    ret = list(self.w)
    ret.extend(self.b)
    return ret

def cosine(a,b,axis=-1):
  return tf.reduce_sum(a*b,axis=axis)/\
          (tf.norm(a,axis=axis)*tf.norm(b,axis=axis))
      

def acc(labels,pred):
  return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(labels,axis=-1),
                                  tf.argmax(pred,axis=-1)),tf.float32))    

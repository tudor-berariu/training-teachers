import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.distributions import Categorical
import scipy as sp
from tensorflow.keras.datasets import fashion_mnist,mnist,cifar10
import pickle



def rnd_ind(collection, size):
  try:
    collection=len(collection)
  except:
    pass
  if size>=collection:
    return list(range(collection))
  else:
    return np.random.choice(collection,size,replace=False)

def covariance(x):
  mean = tf.reduce_mean(x,axis=0)
  x = x-mean
  var = tf.reduce_mean(tf.square(x),axis=0)
  return tf.transpose(x)@x/(var*int(x.shape[0]))


def row_img(data):
  return np.squeeze(np.concatenate(np.split(data,len(data)),axis=2),0)


def save_img(data1, data2, path, n=100):
  n=min(len(data1),len(data2),n)
  data1 = data1[:n]
  data2 = data2[:n]
  data1 = row_img(data1)
  data2 = row_img(data2)
  data = np.concatenate((data1,data2),axis=0)
  if int(data.shape[-1])==1:
    data = np.squeeze(data,-1)
  sp.misc.toimage(data, cmin=0,cmax=1).save(path)

def kl(f, g, axis=-1):
  return tf.nn.softmax_cross_entropy_with_logits_v2(labels=f,logits=g)

def simple_img(data, path, n=20):
  n=min(n,len(data))
  l=int(data.shape[1])
  data=np.squeeze(data,-1)
  data=np.reshape(data[:n],[-1,l])
  sp.misc.toimage(data, cmin=0,cmax=1).save(path)


def get_dataset(args):
  if args.dataset == 'fashion_mnist':
      (train_data, train_labels), (test_data,test_labels) = \
          fashion_mnist.load_data()
      train_data = train_data.astype(np.float32)/255
      test_data = test_data.astype(np.float32)/255
      return (train_data[:,:,:,np.newaxis], train_labels), \
              (test_data[:,:,:,np.newaxis],test_labels)
  elif args.dataset == 'mnist':
      (train_data, train_labels), (test_data,test_labels) = \
          mnist.load_data()
      train_data = train_data.astype(np.float32)/255
      test_data = test_data.astype(np.float32)/255
      return (train_data[:,:,:,np.newaxis], train_labels), \
              (test_data[:,:,:,np.newaxis],test_labels)
  elif args.dataset == 'cifar10':
      (train_data, train_labels), (test_data,test_labels) = \
          cifar10.load_data()
      train_data = train_data.astype(np.float32)/255
      test_data = test_data.astype(np.float32)/255
      return (train_data, np.reshape(train_labels,[-1])), \
              (test_data, np.reshape(test_labels,[-1]))


def create_session(args):
    config = tf.ConfigProto()
    if args.use_gpu==1:
        config.gpu_options.allow_growth = True
    else:
        config = tf.ConfigProto(device_count = {'GPU':0})
    return tf.Session(config=config)



class Model(object):


  def __init__(self):
    raise NotImplementedError()

  @property
  def vars(self):
    ret = list(self.w)
    ret.extend(self.b)
    return ret

  def __call__(self,x):
    raise NotImplementedError()

  def load(self,params):
    ops = []
    for i in range(len(params)):
      if i<len(self.w):
        ops.append(tf.assign(self.w[i], params[i]))
      else:
        ops.append(tf.assign(self.b[i-len(self.w)], params[i]))
    return ops

  def train(self,s,train_data,train_labels,val_data,val_labels,args,save=None):
    with tf.variable_scope(self.name+'_train', reuse=tf.AUTO_REUSE):
      if args.optim == 'sgd':
        optim = tf.train.GradientDescentOptimizer(learning_rate=args.lr,
                                                  name='optimizer')
      indices = tf.placeholder(tf.int32,[None],name='ind')
      training = tf.placeholder_with_default(True,None,name='training')
      ref_data = tf.cond(training, lambda: train_data, lambda: val_data,
                          name='ref_data')
      ref_labels = tf.cond(training, lambda: train_labels, lambda: val_labels,
                          name='ref_labels')
      batch_data_ = tf.gather(ref_data, indices,name='batch_data')
      batch_labels_ = tf.gather(ref_labels, indices, name='batch_labels')
      batch_labels_onehot = tf.one_hot(batch_labels_, 10, dtype=tf.float32,
                                      name='batch_labels_onehot')
      output = self(batch_data_)
      objective = tf.reduce_mean(kl(batch_labels_onehot, output))
      optim_step = optim.minimize(objective,var_list = self.vars)
      accuracy = acc(output,batch_labels_onehot)
      train_size = int(s.run(tf.shape(train_data))[0])
      val_size = int(s.run(tf.shape(val_data))[0])
      best_acc = 0
      best_epoch = 0
      not_better = 0
      accs = []
      losses = []
      for e in range(args.epochs):
        iters = 1+train_size//args.batch_size
        for it in range(iters):
          ind = rnd_ind(train_size,args.batch_size)
          _,acc_,loss_ = s.run([optim_step,accuracy,objective],
                                feed_dict={indices:ind})
          losses.append(loss_)
          # if args.verbose==1:
          #   print(f'epoch[{e}] it[{it+1}/{iters}] '
          #         f'loss[{loss_:.12f}] acc[{acc_:.4f}]')
        iters = 1+val_size//args.batch_size
        t_acc = 0
        for it in range(iters):
          ind = rnd_ind(val_size,args.batch_size)
          t_acc += s.run(accuracy,feed_dict={indices:ind,training:False})/iters
        if args.verbose:
          print(f'epoch[{e}] val_acc[{t_acc:.4f}]')
        accs.append(t_acc)
        if t_acc > best_acc:
          best_acc = t_acc
          not_better = 0
          best_epoch = e
          if save is not None:
            with open(save+'_params.bin','wb') as stream:
              pickle.dump(s.run(self.vars), stream)
        else:
          not_better+=1
        if not_better>5:
          break
      d = {'train_loss': np.array(losses),'val_acc': np.array(accs)}
      with open(save+'_best.txt','w') as stream:
        print(f'best acc: {best_acc}', file=stream)
        print(f'best_epoch: {best_epoch}', file=stream)
      with open(save+'_stats.bin','wb') as stream:
        pickle.dump(d, stream)






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



class ConvProj(Model):

  def __init__(self,name, args,shape):
    self.name=name
    self.w = []
    self.b = []
    self.args = args
    in_size=shape[-1]
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
            s = (shape[1]//(2**i))
            if s==3:
              s=4
            in_size = s*s*self.layers[i-1][-1]
          self.w.append(tf.get_variable(f'w_{i}',[in_size,layer[-1]],
                        initializer=xavier_initializer()))
        self.b.append(tf.get_variable(f'b_{i}', [1,layer[-1]],
                      initializer=tf.initializers.constant(0)))
        in_size = layer[-1]

  def __call__(self,x, training=False):
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
          if self.dropout>0:
            x = tf.layers.dropout(x, self.dropout, training=training)
          if self.args.activation == 'tanh':
            x = tf.tanh(x)
          elif self.args.activation == 'relu':
            x = tf.nn.relu(x)
    return x 

  

def cosine(a,b,axis=-1):
  return tf.reduce_sum(a*b,axis=axis)/\
          (tf.norm(a,axis=axis)*tf.norm(b,axis=axis))
      

def acc(labels,pred,one_hot=True):
  if one_hot:
    return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(labels,axis=-1),
                                  tf.argmax(pred,axis=-1)),tf.float32))  
  else:
    return tf.reduce_mean(tf.cast(tf.equal(tf.cast(labels,tf.int64),
                                  tf.argmax(pred,axis=-1)),tf.float32))  

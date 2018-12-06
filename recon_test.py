import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
import numpy as np
import os
from types import SimpleNamespace
from liftoff.config import read_config
import pickle
from tensorflow.keras.datasets import fashion_mnist
import time
import scipy as sp

def kl(f, g, axis=-1):
    f_ = tf.nn.softmax(f)
    g_ = tf.nn.softmax(g)
    return tf.reduce_sum(f_*(tf.log(f_)-tf.log(g_)),axis=axis)
    # return tf.contrib.kl_distance(f_,g_)


def get_dataset(args):
  if args.dataset=='random':
    return tf.placeholder_with_default(
        np.random.normal(size=[args.ds_size,*args.in_size]).astype(np.float32),
        [args.ds_size,*args.in_size])
  elif args.dataset=='fashion_mnist':
    (train_images, train_labels), (test_images, test_labels) = \
      fashion_mnist.load_data()
    ret = train_images.copy()
    np.random.shuffle(ret)
    ret = ret[:args.ds_size]/255
    ret = ret.astype(np.float32)
    print(ret.shape)
    if args.proj == 'lin':
      ret = np.reshape(ret, [args.ds_size, -1])
      args.in_size = [int(ret.shape[1])]
    else:
      args.in_size = [28,28,1]
    ret = tf.constant(ret)
    ret = tf.expand_dims(ret,-1)
    return ret

def save_img(data1, data2, path, n=20):
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
    ret = self.w
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
            in_size = 7*7*16
          self.w.append(tf.get_variable(f'w_{i}',[in_size,layer[-1]],
                        initializer=xavier_initializer()))
        self.b.append(tf.get_variable(f'b_{i}', [1,layer[-1]],
                      initializer=tf.initializers.constant(0)))
        in_size = layer[-1]

  def __call__(self,x):
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
          if self.args.activation == 'tanh':
            x = tf.tanh(x)
          elif self.args.activation == 'relu':
            x = tf.nn.relu(x)
    return x 

  @property
  def vars(self):
    ret = self.w
    ret.extend(self.b)
    return ret


          

def run(args):
  tf.set_random_seed(time.time())
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  s = tf.Session(config=config)
  ds = get_dataset(args)
  x = tf.get_variable('rec', [args.ds_size,*args.in_size],
                      initializer=tf.initializers.random_normal())
  batch_mask = tf.placeholder(tf.float32,[args.n_proj])
  proj = []
  mse_output = 0
  kl_output = 0
  out_real, out_fake = [], []
  proj_vars = []
  for i in range(args.n_proj):
    if args.proj == 'lin':
      proj.append(LinProj(f'proj_{i}',args))
    elif args.proj == 'conv':
      proj.append(ConvProj(f'proj_{i}',args))
    proj_vars.extend(proj[-1].vars)
    out_real.append(tf.stop_gradient(proj[-1](ds)))
    out_fake.append(proj[-1](x))
    mse_output+=batch_mask[i]*tf.reduce_sum(\
                tf.square(out_fake[-1]-out_real[-1]),axis=1)
    kl_output+=batch_mask[i]*kl(out_real[-1], out_fake[-1], axis=1)

  mse_output = tf.reduce_mean(mse_output)/min(args.batch_size,args.n_proj)
  kl_output = tf.reduce_mean(kl_output)/min(args.batch_size,args.n_proj)
  recon_dist = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(\
    tf.layers.flatten(ds)-tf.layers.flatten(x)),axis=1)))

  if args.optim=='momentum':
    optim = tf.train.MomentumOptimizer(learning_rate=10, momentum=0.9)
  elif args.optim=='adam':
    optim = tf.train.AdamOptimizer(learning_rate=1e-2)
  if args.objective == 'mse':
    step = optim.minimize(mse_output,var_list=[x])
  else:
    step = optim.minimize(kl_output,var_list=[x])
   
  reset_proj = tf.initializers.variables(proj_vars)

  s.run(tf.global_variables_initializer())
  tf.summary.scalar('mse_output', mse_output)
  tf.summary.scalar('kl_output',kl_output)
  tf.summary.scalar('recon_dist', recon_dist)
  logs = tf.summary.merge_all()
  writer = tf.summary.FileWriter(args.out_dir, s.graph)
  for ep in range(args.epochs):
    if args.batch_size<args.n_proj:
      pos = np.random.permutation(args.n_proj)[:args.batch_size]
      batch = np.zeros(args.n_proj)
      batch[pos]=1
    else:
      batch=np.ones(args.n_proj)
    _, mse_output_, x_, ds_,recon_dist_,kl_output_, logs_ = \
      s.run([step,mse_output,x,ds,recon_dist,kl_output,logs],
            feed_dict={batch_mask: batch})
    if (ep+1)%args.reset_freq == 0:
      s.run(reset_proj)

    writer.add_summary(logs_,ep)

    if args.verbose == 1:
      print(f'ep[{ep}] mse_output[{mse_output_:.8f}] '
              f'kl_output[{kl_output_:.8f}] '
              f'recon_dist[{recon_dist_:.8f}]')

  if args.out_dir is None:
    args.out_dir = './junk'
  with open(os.path.join(args.out_dir,'summary.pkl'),'wb') as stream:
    pickle.dump({'recon_dist':recon_dist_, 'mse_output':mse_output_,
                  'kl_output':kl_output_},stream,pickle.HIGHEST_PROTOCOL)
  if args.dataset=='fashion_mnist':
    save_img(s.run(ds),x_,os.path.join(args.out_dir,'rec.png')) 

def main():
  args = read_config()
  run(args)

if __name__=='__main__':
  main()

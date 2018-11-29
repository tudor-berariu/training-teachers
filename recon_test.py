import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
import numpy as np
import os
from types import SimpleNamespace
from liftoff.config import read_config
import pickle

def kl(f, g):
    f_ = tf.nn.softmax(f)
    g_ = tf.nn.softmax(g)
    return tf.reduce_sum(f_*(tf.log(f_)-tf.log(g_)),axis=1)

class Proj(object):

  def __init__(self, name, args):
    self.name=name
    self.sizes = [args.in_size]+args.layers+[args.out_size]
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

def run(args):
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  s = tf.Session(config=config)
  if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)
  else:
    os.system(f'rm -r {args.out_dir}/*')
  ds = tf.placeholder_with_default(
        np.random.normal(size=[args.ds_size,args.in_size]).astype(np.float32),
        [args.ds_size,args.in_size])
  x = tf.get_variable('rec', [args.ds_size,args.in_size])
  proj = []
  mse_output = 0
  kl_output = 0
  out_real, out_fake = [], []
  proj_vars = []
  for i in range(args.n_proj):
    proj.append(Proj(f'proj_{i}',args))
    proj_vars.extend(proj[-1].vars)
    out_real.append(tf.stop_gradient(proj[-1](ds)))
    out_fake.append(proj[-1](x))
    mse_output+=tf.reduce_sum(tf.square(out_fake[-1]-out_real[-1]),axis=1)
    kl_output+=kl(out_fake[-1],out_real[-1])

  mse_output = tf.reduce_mean(mse_output)
  kl_output = tf.reduce_mean(kl_output)
  recon_dist = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(ds-x),axis=1)))

  optim = tf.train.MomentumOptimizer(learning_rate=1e-1, momentum=0.9)
  # optim = tf.train.AdamOptimizer(learning_rate=1e-1)
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
  d = {'mse_output':[], 'recon_dist':[], 'kl_output':[]}
  for ep in range(args.epochs):
    _, mse_output_, x_, ds_,recon_dist_,kl_output_, logs_ = \
      s.run([step,mse_output,x,ds,recon_dist,kl_output,logs])
    if (ep+1)%args.reset_freq == 0:
      s.run(reset_proj)
    d['mse_output'].append(mse_output_)
    d['recon_dist'].append(recon_dist_)
    d['kl_output'].append(kl_output_)
    writer.add_summary(logs_,ep)
    if args.verbose == 1:
      print(f'ep[{ep}] mse_output[{mse_output_:.8f}] '
              f'kl_output[{kl_output_:.8f}] '
              f'recon_dist[{recon_dist_:.8f}]')
  with open(os.path.join(args.out_dir,'summary.pkl'),'wb') as stream:
    pickle.dump(d,stream,pickle.HIGHEST_PROTOCOL)

def main():
  args = read_config()
  run(args)

if __name__=='__main__':
  main()

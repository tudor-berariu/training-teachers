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
from tensorflow.distributions import Categorical

def kl(f, g, axis=-1):
  f_ = tf.nn.softmax(f)
  g_ = tf.nn.softmax(g)
  # return tf.reduce_sum(f_*(tf.log(f_)-tf.log(g_)),axis=axis)
  return tf.distributions.kl_divergence(Categorical(f_),Categorical(g_))


def get_dataset(args):
  if args.dataset=='random':
    return tf.placeholder_with_default(
      np.random.normal(size=[args.ds_size,*args.in_size]).astype(np.float32),
      [args.ds_size,*args.in_size]), \
      tf.one_hot(tf.random.uniform(args.ds_size, 0,args.out_size,
                                    dtype=tf.int32),args.out_size,
                                    dtype=tf.int32)
  elif args.dataset=='fashion_mnist':
    (train_images, train_labels), (test_images, test_labels) = \
      fashion_mnist.load_data()
    ret = train_images.copy()
    perm = np.random.permutation(len(ret))[:args.ds_size]
    ret = ret[perm]/255
    ret = ret.astype(np.float32)
    if args.proj == 'lin':
      ret = np.reshape(ret, [args.ds_size, -1])
      args.in_size = [int(ret.shape[1])]
    else:
      args.in_size = [28,28,1]
    ret = tf.constant(ret)
    ret = tf.expand_dims(ret,-1)
    return ret, tf.one_hot(train_labels[perm],10,dtype=tf.int32)

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
    ret = list(self.w)
    ret.extend(self.b)
    return ret

def cosine(a,b,axis=-1):
  return tf.reduce_sum(a*b,axis=axis)/\
          (tf.norm(a,axis=axis)*tf.norm(b,axis=axis))
      

def acc(labels,pred):
  return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(labels,axis=-1),
                                  tf.argmax(pred,axis=-1)),tf.float32))    

def run(args):
  if args.reset_freq==1 and args.n_proj>args.batch_size:
    args.n_proj = args.batch_size
  if args.out_dir is None:
    args.out_dir = './junk'
  tf.set_random_seed(time.time())
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  s = tf.Session(config=config)
  ds, labels = get_dataset(args)
  x = tf.get_variable('rec', [args.ds_size,*args.in_size],
                      initializer=tf.initializers.random_normal())
  batch_mask = tf.placeholder_with_default(np.ones([args.n_proj],\
                                            dtype=np.float32),[args.n_proj])
  data_batch_mask = tf.placeholder_with_default(\
                    np.ones([args.ds_size], dtype=np.float32), 
                    [args.ds_size])
  proj = []
  mse_output = 0
  kl_output = 0
  grad_cos = 0
  grad_mse = 0
  out_real, out_fake = [], []
  proj_vars = []
  proj_acc_real = 0
  proj_acc_fake = 0
  total_loss_real = 0
  total_loss_fake = 0
  for i in range(args.n_proj):
    if args.proj == 'lin':
      proj.append(LinProj(f'proj_{i}',args))
    elif args.proj == 'conv':
      proj.append(ConvProj(f'proj_{i}',args))
    proj_vars.extend(proj[-1].vars)
    out_real.append(proj[-1](ds))
    out_fake.append(proj[-1](x))
    proj_acc_real += acc(labels,out_real[-1])
    proj_acc_fake += acc(labels,out_fake[-1])


    task_loss_fake = tf.nn.softmax_cross_entropy_with_logits(
                                    labels=labels,
                                    logits=out_fake[-1])
    task_loss_real = tf.nn.softmax_cross_entropy_with_logits(
                                    labels=labels,
                                    logits=out_real[-1])
    if args.align_grad==0:
      task_loss_fake = tf.reduce_mean(task_loss_fake)
      task_loss_real = tf.reduce_mean(task_loss_real)
    total_loss_fake += task_loss_fake/args.n_proj
    total_loss_real += task_loss_real/args.n_proj
    grad_fake = []
    grad_real = []
    if len(task_loss_real.shape)==0:
      task_loss_real = tf.reshape(task_loss_real, [1])
      task_loss_fake = tf.reshape(task_loss_fake, [1])
    
    for i in range(int(task_loss_real.shape[0])):
      grad_fake_i = tf.gradients(task_loss_fake[i], proj[-1].vars)
      grad_real_i = tf.gradients(task_loss_real[i], proj[-1].vars)
      grad_fake_i = [tf.reshape(v,[-1]) for v in grad_fake_i]
      grad_real_i = [tf.reshape(v,[-1]) for v in grad_real_i]
      grad_fake.extend(grad_fake_i)
      grad_real.extend(grad_real_i)
    grad_fake = tf.concat(grad_fake,0)
    grad_real = tf.concat(grad_real,0)
    grad_mse += batch_mask[i]*\
                tf.reduce_sum(tf.square(grad_fake-grad_real),
                              axis=-1)/args.batch_size
    grad_cos += batch_mask[i]*cosine(grad_real,grad_fake)/\
                int(task_loss_real.shape[0])
    mse_output+=batch_mask[i]*tf.reduce_sum(\
                tf.square(out_fake[-1]-out_real[-1]),axis=1)
    kl_output+=batch_mask[i]*kl(out_real[-1], out_fake[-1], axis=1)

  grad_cos = -tf.reduce_sum(grad_cos)/ \
              (min(args.batch_size,args.n_proj)*args.data_batch)
  mse_output = tf.reduce_sum(mse_output)/ \
                (min(args.batch_size,args.n_proj)*args.data_batch)
  kl_output = tf.reduce_sum(kl_output)/ \
                (min(args.batch_size,args.n_proj)*args.data_batch)
  recon_dist = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(\
    tf.layers.flatten(ds)-tf.layers.flatten(x)),axis=1)))
  proj_acc_real/=args.n_proj
  proj_acc_fake/=args.n_proj
  if args.optim=='momentum':
    optim = tf.train.MomentumOptimizer(learning_rate=10, momentum=0.9)
  elif args.optim=='adam':
    optim = tf.train.AdamOptimizer(learning_rate=1e-2)

  if args.train_proj>0:
    optim_proj = tf.train.AdamOptimizer(learning_rate=1e-3)
    proj_step = optim_proj.minimize(total_loss_real, var_list=proj_vars)
    reset_optim_proj = tf.initializers.variables(optim_proj.variables())

  if args.test_steps>0:
    optim_test = tf.train.AdamOptimizer(learning_rate=1e-3)
    test_step = optim_test.minimize(total_loss_fake, var_list=proj_vars)

  objective = None
  if args.objective == 'mse':
    objective = mse_output
  elif args.objective == 'kl':
    objective = kl_output
  elif args.objective == 'grad_cos':
    objective = grad_cos
  elif args.objective == 'grad_mse':
    objective = grad_mse
   

  reset_proj = tf.initializers.variables(proj_vars)
  # grads = optim.compute_gradients(objective, x)
  # print(grads)
  # grads = [(grads[0][0]*tf.tile(tf.reshape(data_batch_mask,
  #                                         [100,1,1,1]),[1,28,28,1]), 
  #           grads[0][1])]
  # step = optim.apply_gradients(grads) 
  objective-=args.real_divergence*recon_dist  
  step = optim.minimize(objective,var_list=[x])    
  s.run(tf.global_variables_initializer())
  s.run(tf.local_variables_initializer())
  tf.summary.scalar('mse_output', mse_output)
  tf.summary.scalar('kl_output',kl_output)
  tf.summary.scalar('grad_mse',grad_mse)
  tf.summary.scalar('grad_cos',grad_cos)
  tf.summary.scalar('recon_dist', recon_dist)
  tf.summary.scalar('proj_loss', total_loss_real)
  tf.summary.scalar('proj_acc_real', proj_acc_real)
  logs = tf.summary.merge_all()
  ds_img = tf.summary.image('ds_img', ds, max_outputs=10)
  recon_img = tf.summary.image('recon_img',x, max_outputs=10)
  img_sum = tf.summary.merge([ds_img, recon_img])
  writer = tf.summary.FileWriter(args.out_dir, s.graph)
  for i in range(args.train_proj):
    _, a = s.run([proj_step, proj_acc_real])
  for ep in range(args.epochs):
    if args.batch_size<args.n_proj:
      pos = np.random.permutation(args.n_proj)[:args.data_batch]
      batch = np.zeros(args.n_proj,dtype=np.float32)
      batch[pos]=1
    else:
      batch=np.ones(args.n_proj)
    pos = np.random.permutation(args.ds_size)\
                  .astype(np.int32)[:args.data_batch]
    data_batch = np.zeros(args.ds_size, dtype=np.float32)
    data_batch[pos]=1
    _, mse_output_, x_, ds_,recon_dist_,kl_output_, \
      grad_mse_,grad_cos_,logs_, img_sum_, proj_acc_real_ = \
      s.run([step,mse_output,x,ds,recon_dist,kl_output,\
              grad_mse,grad_cos,logs,img_sum,proj_acc_real],
            feed_dict={batch_mask: batch, data_batch_mask: data_batch})
    if (ep+1)%args.reset_freq == 0:
      if args.train_proj>0:
        s.run([reset_proj,reset_optim_proj])
      else:
        s.run(reset_proj)
      for i in range(args.train_proj):
        _, a = s.run([proj_step, proj_acc_real])

    writer.add_summary(logs_,ep)
    if ep%500==0:
      writer.add_summary(img_sum_, ep)

    if args.verbose == 1:
      print(f'ep[{ep}] mse_output[{mse_output_:.8f}] '
            f'kl_output[{kl_output_:.8f}] '
            f'grad_mse[{grad_mse_:.8f}] '
            f'grad_cos[{grad_cos_:.8f}] '
            f'recon_dist[{recon_dist_:.8f}] '
            f'proj_acc_real[{proj_acc_real_:.4f}]')

  test_acc_real = tf.summary.scalar('test_acc_real', proj_acc_real)
  test_acc_fake = tf.summary.scalar('test_acc_fake', proj_acc_fake)
  test_loss_real = tf.summary.scalar('test_loss_real', total_loss_real)
  test_loss_fake = tf.summary.scalar('test_loss_fake', total_loss_fake)
  test_sum = tf.summary.merge([test_acc_real, test_acc_fake,test_loss_real,\
                                test_loss_fake])

  s.run(reset_proj)
  best_acc_real = 0
  for i in range(args.test_steps):
    _,test_sum_,proj_acc_fake_,proj_acc_real_ = \
      s.run([test_step, test_sum,proj_acc_fake,proj_acc_real])
    if proj_acc_real_>best_acc_real:
      best_acc_real=proj_acc_real_
    writer.add_summary(test_sum_,i)

  with open(os.path.join(args.out_dir,'summary.pkl'),'wb') as stream:
    pickle.dump({'recon_dist':recon_dist_, 'mse_output':mse_output_,
                  'kl_output':kl_output_, 'grad_mse':grad_mse_,
                  'grad_cos':grad_cos_,'best_acc_real':best_acc_real},
                  stream,pickle.HIGHEST_PROTOCOL)
  if args.dataset=='fashion_mnist':
    save_img(ds_,x_,os.path.join(args.out_dir,'rec.png'))
  saver = tf.train.Saver()
  saver.save(s, args.out_dir) 

def main():
  args = read_config()
  run(args)

if __name__=='__main__':
  main()

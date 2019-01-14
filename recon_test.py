import tensorflow as tf
import numpy as np
import os
from types import SimpleNamespace
from liftoff.config import read_config
import pickle
import time
import scipy as sp
from tf_utils import *


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
    perm = np.random.permutation(len(ret))[:args.ds_size*2]
    ret = ret[perm]/255
    ret = ret.astype(np.float32)
    if args.proj == 'lin':
      ret = np.reshape(ret, [args.ds_size*2, -1])
      args.in_size = [int(ret.shape[1])]
    else:
      args.in_size = [28,28,1]
    ret = tf.constant(ret)
    ret = tf.expand_dims(ret,-1)
    return ret, tf.one_hot(train_labels[perm],10,dtype=tf.float32)


def run(args):
  np.set_printoptions(linewidth=200)
  if args.reset_freq==1 and args.n_proj>args.batch_size:
    args.n_proj = args.batch_size
  if args.out_dir is None:
    args.out_dir = './junk/'
  tf.set_random_seed(time.time())
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  s = tf.Session(config=config)
  ds, labels = get_dataset(args)
  ds_train = ds[:args.ds_size]
  labels_train = labels[:args.ds_size]
  ds_test = ds[args.ds_size:]
  labels_test = labels[args.ds_size:]
  memory = tf.get_variable('memory', [args.ds_size,*args.in_size],
                      initializer=tf.initializers.random_normal())
  fake_labels = tf.get_variable('fake_labels', labels_train.shape,
                      initializer=tf.initializers.random_normal())
  fake_labels_softmax = tf.nn.softmax(fake_labels)
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
  out_real, out_real_test, out_fake = [], [], []
  proj_vars = []
  proj_acc_real = 0
  proj_acc_real_test = 0
  proj_acc_fake = 0
  total_loss_real = 0
  total_loss_real_test = 0
  total_loss_fake = 0
  task_mse = 0
  for i in range(args.n_proj):
    if args.proj == 'lin':
      proj.append(LinProj(f'proj_{i}',args))
    elif args.proj == 'conv':
      proj.append(ConvProj(f'proj_{i}',args))
    proj_vars.extend(proj[-1].vars)
    out_real.append(proj[-1](ds_train))
    out_real_test.append(proj[-1](ds_test))
    out_fake.append(proj[-1](memory))
    proj_acc_real += acc(labels_train,out_real[-1])
    proj_acc_real_test += acc(labels_test, out_real_test[-1])
    proj_acc_fake += acc(fake_labels,out_fake[-1])


    # task_loss_fake = tf.nn.softmax_cross_entropy_with_logits(
    #                                 labels=labels,
    #                                 logits=out_fake[-1])
    # task_loss_real = tf.nn.softmax_cross_entropy_with_logits(
    #                                 labels=labels,
    #                                 logits=out_real[-1])
    task_loss_fake = kl(tf.nn.softmax(out_fake), fake_labels_softmax)
    task_loss_real = kl(tf.nn.softmax(out_real), labels_train)
    task_loss_real_test = kl(out_real_test, labels_test)
    if args.align_grad==0:
      task_loss_fake = tf.reduce_mean(task_loss_fake)
      task_loss_real = tf.reduce_mean(task_loss_real)
      task_loss_real_test = tf.reduce_mean(task_loss_real_test)
    total_loss_fake += task_loss_fake/args.n_proj
    total_loss_real += task_loss_real/args.n_proj
    total_loss_real_test += task_loss_real_test/args.n_proj
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
    task_mse += batch_mask[i]*tf.square(task_loss_real-task_loss_fake)/\
                args.batch_size
    mse_output+=batch_mask[i]*tf.reduce_sum(\
                tf.square(out_fake[-1]-out_real[-1]),axis=1)
    kl_output+=batch_mask[i]*kl(out_real[-1], out_fake[-1], axis=1)

  grad_cos = -tf.reduce_sum(grad_cos)/ \
              (min(args.batch_size,args.n_proj)*args.data_batch)
  mse_output = tf.reduce_sum(mse_output)/ \
                (min(args.batch_size,args.n_proj)*args.data_batch)
  kl_output = tf.reduce_sum(kl_output)/ \
                (min(args.batch_size,args.n_proj)*args.data_batch)
  task_mse = tf.squeeze(task_mse)
  recon_dist = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(\
    tf.layers.flatten(ds_train)-tf.layers.flatten(memory)),axis=1)))
  proj_acc_real/=args.n_proj
  proj_acc_fake/=args.n_proj
  proj_acc_real_test/=args.n_proj
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
  elif args.objective == 'task_mse':
    objective = task_mse
   

  reset_proj = tf.initializers.variables(proj_vars)
  # grads = optim.compute_gradients(objective, x)
  # print(grads)
  # grads = [(grads[0][0]*tf.tile(tf.reshape(data_batch_mask,
  #                                         [100,1,1,1]),[1,28,28,1]), 
  #           grads[0][1])]
  # step = optim.apply_gradients(grads) 
  objective-=args.real_divergence*recon_dist  
  step = optim.minimize(objective,var_list=[memory, fake_labels])    
  s.run(tf.global_variables_initializer())
  s.run(tf.local_variables_initializer())
  tf.summary.scalar('mse_output', mse_output)
  tf.summary.scalar('kl_output',kl_output)
  tf.summary.scalar('grad_mse',grad_mse)
  tf.summary.scalar('grad_cos',grad_cos)
  tf.summary.scalar('task_mse',task_mse)
  tf.summary.scalar('recon_dist', recon_dist)
  tf.summary.scalar('proj_loss', total_loss_real)
  tf.summary.scalar('proj_acc_real', proj_acc_real)
  tf.summary.tensor_summary('fake_labels', fake_labels_softmax)
  logs = tf.summary.merge_all()
  ds_img = tf.summary.image('ds_img', ds_train, max_outputs=10)
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
    _, mse_output_, memory_, ds_train_,recon_dist_,kl_output_, task_mse_, \
      fake_labels_,grad_mse_,grad_cos_,logs_, img_sum_, proj_acc_real_ = \
      s.run([step,mse_output,memory,ds_train,recon_dist,kl_output, task_mse, \
            fake_labels_softmax, grad_mse,grad_cos,logs,img_sum,proj_acc_real],
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
            f'task_mse[{task_mse_:.8f}] '
            f'recon_dist[{recon_dist_:.8f}] '
            f'proj_acc_real[{proj_acc_real_:.4f}]')
      print(fake_labels_[:10])

  test_acc_real = tf.summary.scalar('test_acc_real_test', proj_acc_real_test)
  test_acc_real = tf.summary.scalar('test_acc_real', proj_acc_real)
  test_acc_fake = tf.summary.scalar('test_acc_fake', proj_acc_fake)
  test_loss_real = tf.summary.scalar('test_loss_real_test', 
                                    total_loss_real_test)
  test_loss_real = tf.summary.scalar('test_loss_real', 
                                    total_loss_real)
  test_loss_fake = tf.summary.scalar('test_loss_fake', total_loss_fake)
  test_sum = tf.summary.merge([test_acc_real, test_acc_fake,test_loss_real,\
                                test_loss_fake])

  s.run(reset_proj)
  best_acc_real = 0
  for i in range(args.test_steps):
    _,test_sum_,proj_acc_fake_,proj_acc_real_test_ = \
      s.run([test_step, test_sum,proj_acc_fake,proj_acc_real_test])
    if proj_acc_real_test__>best_acc_real:
      best_acc_real=proj_acc_real_test_
    writer.add_summary(test_sum_,i)

  with open(os.path.join(args.out_dir,'summary.pkl'),'wb') as stream:
    pickle.dump({'recon_dist':recon_dist_, 'mse_output':mse_output_,
                  'kl_output':kl_output_, 'grad_mse':grad_mse_,
                  'task_mse':task_mse_,
                  'grad_cos':grad_cos_,'best_acc_real':best_acc_real},
                  stream,pickle.HIGHEST_PROTOCOL)
  if args.dataset=='fashion_mnist':
    save_img(ds_train_,memory_,os.path.join(args.out_dir,'rec.png'))
  saver = tf.train.Saver()
  saver.save(s, args.out_dir) 

def main():
  args = read_config()
  run(args)

if __name__=='__main__':
  main()

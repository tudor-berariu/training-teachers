from liftoff.config import read_config
import tensorflow as tf
import numpy as np
import pickle
from tf_utils import *
import os

def get_minimas(args):
    ret = []
    path = f'params/{args.dataset}/'
    for root,dirs,files in os.walk(path):
        for file in files:
            if 'params' in file:
                with open(os.path.join(path,file),'rb') as stream:
                    ret.append(pickle.load(stream))
    return np.array(ret)

def run(args):
    s = create_session(args)
    log_writer = tf.summary.FileWriter(args.out_dir, s.graph)
    np.set_printoptions(linewidth=10000)
    (train_data, train_labels), (test_data, test_labels) = get_dataset(args)
    minimas = get_minimas(args)
    val_ind = int(len(train_data)*0.8)
    val_data, val_labels = tf.constant(train_data[val_ind:]), \
                            tf.constant(train_labels[val_ind:],dtype=tf.float32)
        
    fake_data = tf.get_variable('fake_data', 
                            [args.memory_size, *train_data.shape[1:]],
                            initializer = tf.initializers.random_normal())
    fake_labels = tf.get_variable('fake_labels',
                            [args.memory_size, 10],
                            initializer = tf.initializers.random_normal())
    objective = 0
    models = []
    for i in range(args.batch_size):    
        m = ConvProj(f'conv_model_{i}',args,[None]+list(train_data.shape[1:]))
        output = m(fake_data)
        loss = tf.reduce_mean(kl(tf.nn.softmax(fake_labels),output))
        grads = tf.gradients(loss, m.vars)
        grads = tf.concat([tf.reshape(v,[-1]) for v in grads],0)
        objective += tf.reduce_sum(tf.square(grads))/args.batch_size
        models.append(m)

    optim = tf.train.AdamOptimizer(learning_rate=1e-1)
    optim_step = optim.minimize(objective, var_list = [fake_data,fake_labels])
    s.run(tf.global_variables_initializer())
    iters = 1+len(minimas)//args.batch_size
    val_accs = []
    init_model = tf.initializers.variables(models[0].vars)
    logs = tf.summary.scalar('objective',objective)
    fake_train = models[0].prepare_train(s,fake_data,tf.nn.softmax(fake_labels),
                                    val_data,val_labels,args)
    for e in range(args.epochs):
        for it in range(iters):
            ops = []
            ind = rnd_ind(len(minimas),args.batch_size)
            for i,j in enumerate(ind):
                ops.extend(models[i].load(minimas[j]))
            with tf.control_dependencies(ops):
                _, objective_, logs_ = s.run([optim_step, objective, logs])
                print(f'epoch[{e}] it[{it+1}/{iters}] loss[{objective_}]')
                log_writer.add_summary(logs_, e*iters+it)
        s.run(init_model)
        val_acc = models[0].train(s,*fake_train,args)
        val_accs.append(val_acc)
        print(f'epoch[{e}] val_acc[{val_acc}]')
        fake_data_, fake_labels_ = s.run([fake_data, tf.nn.softmax(fake_labels)])
        save_img(test_data,fake_data_,
                    os.path.join(args.out_dir,f'fake_data_{e}.png'),50)
        with open(os.path.join(args.out_dir,f'labels_{e}.txt'),'w') as stream:
            for i in range(50):
                print(fake_labels_[i],file=stream)
    
    with open(os.path.join(args.out_dir,'val_accs.txt'),'w') as stream:
        for a in val_accs:
            print(a, file=stream)


def main():
    run(read_config())

if __name__=='__main__':
    main()
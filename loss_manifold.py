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
    (train_data, train_labels), (test_data, test_labels) = get_dataset(args)
    minimas = get_minimas(args)
    val_ind = int(len(train_data)*0.8)
    val_data, val_labels = tf.constant(train_data[val_ind:]), \
                            tf.constant(train_labels[val_ind:])
    train_data, train_labels = tf.constant(train_data[:val_ind]), \
                                tf.constant(train_labels[:val_ind])
        
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
        objective += tf.reduce_mean(tf.square(grads))/args.batch_size
        models.append(m)

    optim = tf.train.AdamOptimizer()
    optim_step = optim.minimize(objective, var_list = [fake_data,fake_labels])
    s.run(tf.global_variables_initializer())
    for e in range(args.epochs):
        for it in range(1+len(minimas)//args.batch_size):
            ops = []
            ind = rnd_ind(len(minimas),args.batch_size)
            for i,j in enumerate(ind):
                ops.extend(models[i].load(minimas[j]))
            with tf.control_dependencies(ops):
                _, objective_ = s.run([optim_step,objective])
                print(objective_)

    save_img(test_data,fake_data_,
                    os.path.join(args.out_dir,'memory.png'),50)


def main():
    run(read_config())

if __name__=='__main__':
    main()
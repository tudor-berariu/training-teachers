from liftoff.config import read_config
import tensorflow as tf
import numpy as np
import pickle
from tf_utils import *



def run(args):
    s = create_session(args)
    (train_data, train_labels), (test_data, test_labels) = get_dataset(args)
    m = ConvProj('conv_model',args,[None]+list(train_data.shape[1:]))
    val_ind = int(len(train_data)*0.8)
    val_data, val_labels = tf.constant(train_data[val_ind:]), \
                            tf.constant(train_labels[val_ind:])
    train_data, train_labels = tf.constant(train_data[:val_ind]), \
                                tf.constant(train_labels[:val_ind])
    train_labels, val_labels = tf.one_hot(train_labels,10,dtype=tf.float32)
                                tf.one_hot(val_labels,10,dtype=tf.float32)

    reinit = tf.global_variables_initializer()
    graph_nodes = m.prepare_train(s,train_data,train_labels,
                                    val_data,val_labels,args)
    for i in range(1000):
        s.run(reinit)
        val = m.train(s,*graph_nodes,args,
                        save=f'params/{args.dataset}/{i+52}')
        print(f'it[{i}] val[{val}]')


def main():
    args = read_config()
    run(args)


if __name__=='__main__':
    main()
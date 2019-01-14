from liftoff.config import read_config
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from numpy.random import permutation
from tf_utils import ConvProj, kl, cosine, save_img, covariance
import tf_utils
import os

def get_dataset(args):
    if args.dataset == 'fashion_mnist':
        (train_data, train_labels), (test_data,test_labels) = \
            fashion_mnist.load_data()
        train_data = train_data.astype(np.float32)/255
        test_data = test_data.astype(np.float32)/255
        return (train_data[:,:,:,np.newaxis], train_labels), \
                (test_data[:,:,:,np.newaxis],test_labels)


def create_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def create_students(args):
    students = []
    students_params = []
    for i in range(args.n_proj):
        students.append(ConvProj(f'student_{i}',args))
        students_params.extend(students[-1].vars)
    return students, students_params





def run(args):
    if args.out_dir is None:
        args.out_dir = './junk/'
    s = create_session()
    logs_writer = tf.summary.FileWriter(args.out_dir, s.graph)
    (train_data, train_labels), (test_data, test_labels) = get_dataset(args)
    fake_data = tf.get_variable('fake_data', 
                            [args.memory_size, *train_data.shape[1:]],
                            initializer = tf.initializers.random_normal())
    fake_labels = tf.get_variable('fake_labels',
                            [args.memory_size, 10],
                            initializer = tf.initializers.random_normal())
    students, students_params = create_students(args)

    full_real_grad = []
    full_fake_grad = []
    training_fake = tf.placeholder_with_default(True,None)
    training_real = tf.placeholder_with_default(True,None)
    batch_data_ = tf.placeholder(tf.float32, [None,28,28,1],
                                'data_placeholder')
    batch_test_data_ = tf.placeholder(tf.float32,[None,28,28,1])
    batch_labels_ = tf.placeholder(tf.int32,[None],'labels_placeholder')
    batch_labels_onehot = tf.one_hot(batch_labels_, 10, dtype=tf.float32)
    batch_test_labels_ = tf.placeholder(tf.int32,[None])
    batch_test_labels_onehot = tf.one_hot(batch_test_labels_,
                                        10,dtype=tf.float32)
    memory_indices_ = tf.placeholder(tf.int32,[None],'indices_placeholder')
    batch_fake_data = tf.gather(fake_data, memory_indices_)
    batch_fake_labels = tf.nn.softmax(tf.gather(\
                            fake_labels, memory_indices_))
    full_real_grad, full_fake_grad = [], []
    real_acc, fake_acc, test_acc = 0, 0, 0
    total_real_loss, total_fake_loss, total_test_loss = 0, 0, 0
    test_conf = 0
    for student in students:
        real_output = student(batch_data_,training_fake)
        fake_output = student(batch_fake_data, training_real)
        test_output = student(batch_test_data_, False)

        real_loss = kl(batch_labels_onehot,tf.nn.softmax(real_output))
        fake_loss = kl(batch_fake_labels, tf.nn.softmax(fake_output))
        test_loss = kl(batch_test_labels_onehot, 
                        tf.nn.softmax(test_output))
        test_conf += tf.confussion_matrix(batch_test_labels_onehot,
                                            test_output)/args.n_proj
        real_loss = tf.reduce_mean(real_loss)
        fake_loss = tf.reduce_mean(fake_loss)
        test_loss = tf.reduce_mean(test_loss)

        total_real_loss += real_loss/args.n_proj
        total_fake_loss += fake_loss/args.n_proj
        total_test_loss += test_loss/args.n_proj

        real_acc+=tf_utils.acc(real_output, batch_labels_onehot)/args.n_proj
        fake_acc+=tf_utils.acc(fake_output, batch_fake_labels)/args.n_proj

        real_grad = tf.gradients(real_loss, student.vars)
        real_grad = [tf.reshape(v,[-1]) for v in real_grad]
        full_real_grad.extend(real_grad)
        fake_grad = tf.gradients(fake_loss, student.vars)
        fake_grad = [tf.reshape(v,[-1]) for v in fake_grad]
        full_fake_grad.extend(fake_grad)

    full_real_grad = tf.concat(full_real_grad,0)
    full_fake_grad = tf.concat(full_fake_grad,0)

    objective = -cosine(full_real_grad, full_fake_grad)/args.n_proj
    prof_optimizer = tf.train.AdamOptimizer(learning_rate=args.lr)
    prof_optim_step = prof_optimizer.minimize(objective, 
                                    var_list=[fake_data,fake_labels])
    train_optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
    train_optim_step = train_optimizer.minimize(total_real_loss,
                                    var_list=students_params)
    test_optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
    test_optim_step = test_optimizer.minimize(total_fake_loss, 
                                            var_list=students_params)
    reset_students = tf.initializers.variables(students_params)
    reset_train_optim = tf.initializers.variables(train_optimizer.variables())
    s.run(tf.global_variables_initializer())
    s.run(tf.local_variables_initializer())
    tf.summary.scalar('grad_cos', objective)
    tf.summary.scalar('train_real_acc', real_acc)
    tf.summary.scalar('train_fake_acc', fake_acc)
    tf.summary.scalar('train_real_loss', total_real_loss)
    tf.summary.scalar('train_fake_loss', total_fake_loss)
    logs = tf.summary.merge_all()
    for epoch in range(args.epochs):
        perm = permutation(len(train_data))
        no_iters = len(train_data)//args.batch_size
        # no_iters=1
        for it in range(no_iters):
            batch_indices = perm[it*args.batch_size:(it+1)*args.batch_size]
            batch_data = train_data[batch_indices]
            batch_labels = train_labels[batch_indices]
            if args.full_memory == 0 and args.batch_size<args.memory_size:
                memory_indices = np.random.choice(args.memory_size,
                                                    args.batch_size,
                                                    replace=False)
            else:
                memory_indices = list(range(args.memory_size))
            _,objective_,logs_ = s.run([prof_optim_step,objective,logs],
                            feed_dict={batch_data_:batch_data, 
                                        batch_labels_:batch_labels,
                                        memory_indices_:memory_indices})
            logs_writer.add_summary(logs_, epoch*no_iters+it)
            if args.verbose==1:
                print(f'epoch[{epoch+1}] it[{it+1}/{no_iters}] '
                        f'objective[{objective_}]')

        if (epoch+1)%args.reset_freq==0:
            s.run([reset_students,reset_train_optim])
            if args.train_proj > 0:
                train_steps = 0
                while train_steps<600:
                    train_steps+=1
                    batch_indices = np.random.choice(len(train_data),
                                                    args.batch_size,
                                                    replace=False)
                    batch_data = train_data[batch_indices]
                    batch_labels = train_labels[batch_indices]
                    _, acc = s.run([train_optim_step,real_acc],
                                feed_dict={batch_data_:batch_data,
                                            batch_labels_:batch_labels})
                    if acc>args.train_proj:
                        break
                print(f'finished training students steps[{train_steps}] '
                        f'acc[{acc:.4f}]')

    logs = tf.summary.merge([
                        tf.summary.scalar('test_memory_fake_acc', fake_acc),
                        tf.summary.scalar('test_memory_real_acc', real_acc),
                        tf.summary.scalar('test_memory_real_loss',
                                            total_real_loss),
                        tf.summary.scalar('test_memory_fake_loss',
                                            total_fake_loss)])

    s.run([reset_students,reset_train_optim])
    test_steps=max((args.test_epochs*args.memory_size//args.batch_size)+1,600)
    # test_steps=1
    for step in range(test_steps):
        if args.batch_size>args.memory_size:
            memory_indices = list(range(args.memory_size))
        else:
            memory_indices = np.random.choice(args.memory_size,args.batch_size,
                                            replace=False)
        # batch_indices = permutation(len(test_data))[:args.batch_size]
        batch_data=test_data
        batch_labels=test_labels
        _, logs_ = s.run([test_optim_step, logs],
                            feed_dict={batch_data_: batch_data,
                                        batch_labels_: batch_labels,
                                        memory_indices_:memory_indices,
                                        training_real: False})
        logs_writer.add_summary(logs_,step)

    logs = tf.summary.merge([
                        tf.summary.scalar('test_fake_acc', fake_acc),
                        tf.summary.scalar('test_train_acc', real_acc),
                        tf.summary.scalar('test_train_loss',
                                            total_real_loss),
                        tf.summary.scalar('test_fake_loss',
                                            total_fake_loss),
                        tf.summary.scalar('test_acc',test_acc),
                        tf.summary.scalar('test_loss',total_test_loss)])

    s.run([reset_students,reset_train_optim])
    perm = permutation(args.memory_size)
    for step in range(test_steps):
        if batch_size>args.memory_size:
            batch_indices = list(range(args.memory_size))
        else:
            batch_indices = np.random.choice(args.memory_size,args.batch_size,
                                            replace=False)
        batch_indices = perm[batch_indices]
        batch_data = train_data[batch_indices]
        batch_labels = train_labels[batch_indices]
        if args.batch_size>args.memory_size:
            memory_indices = list(range(args.memory_size))
        else:
            memory_indices = np.random.choice(args.memory_size,args.batch_size,
                                            replace=False)
        _, logs_, test_conf_ = s.run([train_optim_step,logs,test_conf],
                    feed_dict={batch_data_:batch_data,
                                batch_labels_:batch_labels,
                                memory_indices_: memory_indices,
                                training_fake: False,
                                batch_test_data_: test_data,
                                batch_test_labels_: test_labels})
        logs_writer.add_summary(logs_,step)
        with open(os.path.join(args.out_dir,f'conf_{step}.txt')) as f:
            print(np.array_str(test_conf_,max_line_with=10000,precision=3),
                file=f)


    fake_data_, fake_labels_,cov_ = s.run([fake_data, 
                                    tf.nn.softmax(fake_labels),
                                    covariance(fake_labels)])
    for line in cov_:
        s = ''
        for val in line:
            s+=f' {val:.4f}'
        print(line)
    print(fake_labels_[:30])
    tf_utils.simple_img(fake_data_,os.path.join(args.out_dir,'memory.png'),30)
            

def main():
    args = read_config()
    run(args)


if __name__=='__main__':
    main()
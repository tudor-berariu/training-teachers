from liftoff.config import read_config
import tensorflow as tf
import numpy as np
from numpy.random import permutation
import tf_utils
import os
from tf_utils import *


def create_students(args,shape):
    students = []
    students_params = []
    for i in range(args.n_proj):
        students.append(ConvProj(f'student_{i}',args,shape))
        students_params.extend(students[-1].vars)
    return students, students_params



def pfile(content,f,args):
    with open(os.path.join(args.out_dir,f),'w') as stream:
        print(content,file=stream)


def run(args):
    if args.out_dir is None:
        args.out_dir = './junk/'
    s = create_session(args)
    train_writer = tf.summary.FileWriter(args.out_dir, s.graph)
    (train_data, train_labels), (test_data, test_labels) = get_dataset(args)
    print(train_data.shape)
    fake_data = tf.get_variable('fake_data', 
                            [args.memory_size, *train_data.shape[1:]],
                            initializer = tf.initializers.random_normal())
    fake_labels = tf.get_variable('fake_labels',
                            [args.memory_size, 10],
                            initializer = tf.initializers.random_normal())
    train_shape = [None]
    train_shape.extend([int(x) for x in train_data.shape[1:]])
    students, students_params = create_students(args, train_shape)

    training_fake = tf.placeholder_with_default(True,None)
    training_real = tf.placeholder_with_default(True,None)
    batch_data_ = tf.placeholder(tf.float32, train_shape,
                                'data_placeholder')
    batch_labels_ = tf.placeholder(tf.int32,[None],'labels_placeholder')
    batch_labels_onehot = tf.one_hot(batch_labels_, 10, dtype=tf.float32)
    
    memory_indices_ = tf.placeholder(tf.int32,[None],'indices_placeholder')
    batch_fake_data = tf.gather(fake_data, memory_indices_)
    batch_fake_labels = tf.nn.softmax(tf.gather(\
                            fake_labels, memory_indices_))
    # full_real_grad, full_fake_grad = [], []
    real_acc, fake_acc = 0, 0
    total_real_loss, total_fake_loss = 0, 0
    real_conf = 0
    objectives = []
    for student in students:
        control_input = None
        if args.parallel==0 and len(objectives)>0:
            control_input=[objectives[-1],total_fake_loss,total_real_loss,
                            real_acc, fake_acc]
        with tf.control_dependencies(control_input):   
            real_output = student(batch_data_,training_real)
            fake_output = student(batch_fake_data, training_fake)

            real_loss = kl(batch_labels_onehot,tf.nn.softmax(real_output))
            fake_loss = kl(batch_fake_labels, tf.nn.softmax(fake_output))
            
            real_loss = tf.reduce_mean(real_loss)
            fake_loss = tf.reduce_mean(fake_loss)

            real_conf += tf.confusion_matrix(batch_labels_,
                                            tf.argmax(real_output,-1),
                                            num_classes=10)/args.n_proj

            total_real_loss += real_loss/args.n_proj
            total_fake_loss += fake_loss/args.n_proj

            real_acc+=tf_utils.acc(real_output, batch_labels_onehot)/args.n_proj
            fake_acc+=tf_utils.acc(fake_output, batch_fake_labels)/args.n_proj

            real_grad = tf.gradients(real_loss, student.vars)
            real_grad = [tf.reshape(v,[-1]) for v in real_grad]
            # full_real_grad.extend(real_grad)
            fake_grad = tf.gradients(fake_loss, student.vars)
            fake_grad = [tf.reshape(v,[-1]) for v in fake_grad]
            # full_fake_grad.extend(fake_grad)
            if args.objective=='grad_cos':
                objectives.append(-cosine(tf.concat(fake_grad,0),
                                            tf.concat(real_grad,0)))
            elif args.objective=='grad_mse':
                objectives.append(tf.reduce_mean(tf.square(
                    tf.concat(fake_grad,0)-tf.concat(real_grad,0))))

    # full_real_grad = tf.concat(full_real_grad,0)
    # full_fake_grad = tf.concat(full_fake_grad,0)

    # objective = -cosine(full_real_grad, full_fake_grad)/args.n_proj
    objective = 0
    for obj in objectives:
        objective+=obj/args.n_proj
    prof_optimizer = tf.train.AdamOptimizer(learning_rate=args.lr)
    prof_optim_step = prof_optimizer.minimize(objective, 
                                    var_list=[fake_data,fake_labels])
    real_optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
    real_optim_step = real_optimizer.minimize(total_real_loss,
                                    var_list=students_params)
    fake_optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
    fake_optim_step = fake_optimizer.minimize(total_fake_loss, 
                                            var_list=students_params)
    reset_students = tf.initializers.variables(students_params)
    reset_real_optim = tf.initializers.variables(real_optimizer.variables())
    s.run(tf.global_variables_initializer())
    s.run(tf.local_variables_initializer())
    tf.summary.scalar(args.objective, objective)
    tf.summary.scalar('train_real_acc', real_acc)
    tf.summary.scalar('train_fake_acc', fake_acc)
    tf.summary.scalar('train_real_loss', total_real_loss)
    tf.summary.scalar('train_fake_loss', total_fake_loss)
    logs = tf.summary.merge_all()

    for epoch in range(args.epochs):
        perm = permutation(len(train_data))
        no_iters = len(train_data)//args.batch_size
        for it in range(no_iters):
            batch_indices = perm[it*args.batch_size:(it+1)*args.batch_size]
            batch_data = train_data[batch_indices]
            batch_labels = train_labels[batch_indices]
            if args.full_memory == 0 and args.batch_size<args.memory_size:
                memory_indices = tf_utils.rnd_ind(args.memory_size,
                                                    args.batch_size)
            else:
                memory_indices = list(range(args.memory_size))
            _,objective_,logs_ = s.run([prof_optim_step,objective,logs],
                            feed_dict={batch_data_:batch_data, 
                                        batch_labels_:batch_labels,
                                        memory_indices_:memory_indices})
            train_writer.add_summary(logs_, epoch*no_iters+it)
            if args.verbose==1:
                print(f'epoch[{epoch+1}] it[{it+1}/{no_iters}] '
                        f'objective[{objective_}]')

        if (epoch+1)%args.reset_freq==0:
            s.run([reset_students,reset_real_optim])
            if args.train_proj > 0:
                train_steps = 0
                while train_steps<args.min_test_it:
                    train_steps+=1
                    batch_indices = tf_utils.rnd_ind(train_data,args.batch_size)
                    batch_data = train_data[batch_indices]
                    batch_labels = train_labels[batch_indices]
                    _, acc = s.run([real_optim_step,real_acc],
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

    s.run([reset_students,reset_real_optim])
    test_steps=max((args.test_epochs*args.memory_size//args.batch_size)+1,
                    args.min_test_it)
    for step in range(test_steps):
        memory_indices = tf_utils.rnd_ind(args.memory_size,args.batch_size)
        batch_indices = tf_utils.rnd_ind(test_data,args.batch_size)
        batch_data=test_data[batch_indices]
        batch_labels=test_labels[batch_indices]
        _, logs_,real_conf_ = s.run([fake_optim_step, logs, real_conf],
                            feed_dict={batch_data_: batch_data,
                                        batch_labels_: batch_labels,
                                        memory_indices_:memory_indices,
                                        training_real: False})
        train_writer.add_summary(logs_,step)
        real_conf_ /= np.sum(real_conf_,axis=1)
        with open(os.path.join(args.out_dir,f'conf_{step}.txt'),'w') as f:
            print(np.array_str(real_conf_,
                            max_line_width=10000,
                            precision=3),
                            file=f)
    logs_1 = tf.summary.merge([
                        tf.summary.scalar('test_fake_acc', fake_acc),
                        tf.summary.scalar('test_train_acc', real_acc),
                        tf.summary.scalar('test_train_loss',
                                            total_real_loss),
                        tf.summary.scalar('test_fake_loss',
                                            total_fake_loss)])

    logs_2 = tf.summary.merge([tf.summary.scalar('test_acc',real_acc),
                        tf.summary.scalar('test_loss',total_real_loss)])
                        

    s.run([reset_students,reset_real_optim])
    perm = permutation(args.memory_size)
    for step in range(test_steps):
        batch_indices = tf_utils.rnd_ind(args.memory_size,args.batch_size)
        batch_indices = perm[batch_indices]
        batch_data = train_data[batch_indices]
        batch_labels = train_labels[batch_indices]
        memory_indices = tf_utils.rnd_ind(args.memory_size,args.batch_size)


        batch_test_indices = tf_utils.rnd_ind(test_data,args.batch_size)
        batch_test_data = test_data[batch_test_indices]
        batch_test_labels = test_labels[batch_test_indices]
        _, logs_ = s.run([real_optim_step,logs_1],
                    feed_dict={batch_data_:batch_data,
                                batch_labels_:batch_labels,
                                memory_indices_: memory_indices,
                                training_fake: False})
        train_writer.add_summary(logs_,step)
        logs_ = s.run(logs_2,
                    feed_dict={batch_data_:batch_test_data,
                                batch_labels_:batch_test_labels,
                                memory_indices_: memory_indices,
                                training_real: False})
        train_writer.add_summary(logs_,step)


    fake_data_, fake_labels_,cov_ = s.run([fake_data, 
                                    tf.nn.softmax(fake_labels),
                                    covariance(fake_labels)])
    np.set_printoptions(linewidth=10000,suppress=True,precision=4)
    pfile(cov_,'cor_mat.txt',args)
    pfile(fake_labels_[:100],'fake_labels.txt',args)
    tf_utils.save_img(test_data,fake_data_,
                    os.path.join(args.out_dir,'memory.png'),50)
            

def main():
    args = read_config()
    run(args)


if __name__=='__main__':
    main()
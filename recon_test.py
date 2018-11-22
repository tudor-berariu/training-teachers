import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer

DS_SIZE = [1,512]
OUT_SIZE = 10
N_PROJ = 50
EPOCHS = 1000

class Proj(object):

  def __init__(self, name, in_size, out_size):
    self.name=name
    self.sizes = [in_size]+[256, 128]+[out_size]
    self.w = []
    self.b = []
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
      for i in range(len(self.sizes)-1):
        self.w.append(tf.get_variable(f'w_{i}',[self.sizes[i],self.sizes[i+1]],
          initializer=xavier_initializer()))
        self.b.append(tf.get_variable(f'b_{i}', [1,self.sizes[i+1]],
          initializer=tf.initializers.constant(0)))

  def __call__(self,x):
    with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
      for i in range(len(self.sizes)-1):
        x = tf.tanh((x@self.w[i])+self.b[i])
    return x


def main():
  s = tf.Session()
  ds = tf.random_normal(DS_SIZE)
  x = tf.get_variable('rec', DS_SIZE)
  proj = []
  loss = 0
  for i in range(N_PROJ):
    proj.append(Proj(f'proj_{i}',DS_SIZE[-1],OUT_SIZE))
    loss+=tf.reduce_mean(tf.reduce_sum(tf.square(proj[-1](ds)-proj[-1](x)),
          axis=1))

  optim = tf.train.AdamOptimizer()
  step = optim.minimize(loss,var_list=[x])

  s.run(tf.global_variables_initializer())
  for ep in range(EPOCHS):
    _, loss_ = s.run([step,loss])
    print(f'ep[{ep}] loss[{loss_:.6f}]')


if __name__=='__main__':
  main()

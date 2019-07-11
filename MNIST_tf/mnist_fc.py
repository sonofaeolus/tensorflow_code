import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

"""
Final acc is 0.91 and crash in the last step
"""

class Mnist_fc(object):

    def __init__(self, infile, batch_size, lr, train_steps):
        self.mnist_data = input_data.read_data_sets(infile, one_hot = True)
        self.x = tf.placeholder(tf.float32, shape = [None, 784])
        self.y = tf.placeholder(tf.float32, shape = [None, 10])
        self.pred_y, self.cost = self.network()
        self.batch_size = batch_size
        self.lr = lr
        self.train_steps = train_steps


    def network(self):

        w = tf.Variable(tf.truncated_normal([784, 10],mean=0.2,stddev=0.5))
        bias = tf.Variable(tf.truncated_normal([10],mean=0.2,stddev=0.5))
        pred_y = tf.nn.softmax(tf.matmul(self.x, w) + bias)
        cost = tf.reduce_mean(-tf.reduce_sum(self.y * tf.log(pred_y)))
        return pred_y, cost
    
    def train(self):

        opt = tf.train.GradientDescentOptimizer(self.lr).minimize(self.cost)
        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)
        for step in range(self.train_steps):
            images, labels = self.mnist_data.train.next_batch(self.batch_size)
            _, cost_value = sess.run([opt, self.cost], feed_dict = {self.x: images, self.y: labels})
            #print("cost is %.4f" % (cost_value))
            if step % 100 == 0:
                correct_num = tf.equal(tf.argmax(self.pred_y, 1), tf.argmax(self.y, 1))
                prec = tf.reduce_mean(tf.cast(correct_num, tf.float32))
                acc = sess.run(prec, feed_dict = {self.x: self.mnist_data.test.images, self.y: self.mnist_data.test.labels})
                print("after step %d, accuracy is %.4f" % (step, acc))

if __name__ == "__main__":
    mf = Mnist_fc("E:/gitcode/tensorflow_code/MNIST_tf/MNIST_data", 100, 0.01, 100000)
    mf.train()
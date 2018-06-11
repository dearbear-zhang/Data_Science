import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.framework import graph_util
mnist = input_data.read_data_sets('data/MNIST', one_hot=True)
# image_feed, labels_feed = data.next_batch(FLAGS.batch_size)
x = tf.placeholder("float", [None, 784], name='input/x_input')
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
# SoftMax 回归 Y = sofmax(XW + B)
y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder("float", [None, 10])
# 定义损失函数,使用交叉熵作为损失函数
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
# 使用梯度下降法 以0.01的学习速率 最小化交叉熵
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
# 随机1000次,每次抽取100个数据进行训练化参数(随机梯度下降法) W,b
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x:batch_xs, y_:batch_ys})

# 评估模型,测试集测试正确率
correct_prediction = tf.equal(tf.argmax(y, 1, output_type='int32', name='output'), tf.argmax(y_, 1, output_type='int32'))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print(sess.run(accuracy, feed_dict={x:mnist.test.images, y_:mnist.test.labels}))

# 保存训练好的模型
# 形参output_node_names用于指定输出的节点名称,output_node_names=['output']对应pre_num=tf.argmax(y,1,name="output"),
output_graph_def = graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names=['output'])
with tf.gfile.FastGFile('model/mnist.pb', mode='wb') as f:  # ’wb’中w代表写文件，b代表将数据以二进制方式写入文件。
    f.write(output_graph_def.SerializeToString())
sess.close()
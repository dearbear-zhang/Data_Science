import argparse

import os
import tempfile

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.framework import graph_util

FLAGS = None


class MnistPath:
    def __init__(self, inputDataPath, log_dir, appModePath, modePath, modeFileName):
        self.inputDataPath = inputDataPath
        self.log_dir = log_dir
        self.appModePath = appModePath
        self.modePath = modePath
        self.modeFileName = modeFileName


def initData():
    """
    初始化相关路径
    :return:
    """
    pathInfo = MnistPath('data/MNIST', 'temp/mnist_deep_logs', 'model_to_app/mnist_deep.pb', 'model/deep', 'model/deep/mnist_deep.ckpt')
    return pathInfo


def deepnn(x):
    """deepnn builds the graph for a deep net for classifying digits.

    Args:
      x: an input tensor with the dimensions (N_examples, 784), where 784 is the
      number of pixels in a standard MNIST image.

    Returns:
      A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
      equal to the logits of classifying the digit into one of 10 classes (the
      digits 0-9). keep_prob is a scalar placeholder for the probability of
      dropout.
    """
    # Reshape to use within a convolutional neural net.
    # Last dimension is for "features" - there is only one here, since images are
    # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, 28, 28, 1])

    # First convolutional layer - maps one grayscale image to 32 feature maps.
    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    # Pooling layer - downsamples by 2X.
    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv1)

    # Second convolutional layer -- maps 32 feature maps to 64.
    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    # Second pooling layer.
    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv2)

    # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
    # is down to 7x7x64 feature maps -- maps this to 1024 features.
    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout - controls the complexity of the model, prevents co-adaptation of
    # features.
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Map the 1024 features to 10 classes, one for each digit
    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    return y_conv, keep_prob


def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def start(isTrain):
    # Import data
    mnist = input_data.read_data_sets(FLAGS.inputDataPath)

    # Create the model
    x = tf.placeholder(tf.float32, [None, 784])

    # Define loss and optimizer
    y_ = tf.placeholder(tf.int64, [None])

    # Build the graph for the deep net
    y_conv, keep_prob = deepnn(x)

    with tf.name_scope('loss'):
        cross_entropy = tf.losses.sparse_softmax_cross_entropy(
            labels=y_, logits=y_conv)
    cross_entropy = tf.reduce_mean(cross_entropy)
    # 数据序列化
    tf.summary.scalar("cost", cross_entropy)

    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1, name="output"), y_)
        correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)
    tf.summary.scalar('accuracy', accuracy)

    merged_sunmary_op = tf.summary.merge_all()
    graph_location = FLAGS.log_dir + '/train'
    print('Saving graph to: %s' % graph_location)
    train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train')
    test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')
    train_writer.add_graph(tf.get_default_graph())
    test_writer.add_graph(tf.get_default_graph())

    # graph_location = tempfile.mkdtemp()
    # print('Saving graph to: %s' % graph_location)
    # train_writer = tf.summary.FileWriter(graph_location)
    # train_writer.add_graph(tf.get_default_graph())

    saver = tf.train.Saver(max_to_keep=1)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        if isTrain:
            # 开始模型训练
            for i in range(1000):
                batch = mnist.train.next_batch(50)
                if i % 100 == 0:
                    train_accuracy = accuracy.eval(feed_dict={
                        x: batch[0], y_: batch[1], keep_prob: 1.0})
                    summary = sess.run(merged_sunmary_op, feed_dict={
                        x: batch[0], y_: batch[1], keep_prob: 1.0})
                    test_writer.add_summary(summary, i)
                    print('step %d, training accuracy %g' % (i, train_accuracy))
                    saver.save(sess, FLAGS.modeFileName, global_step=i)
                train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
                summary = sess.run(merged_sunmary_op, feed_dict={
                    x: batch[0], y_: batch[1], keep_prob: 1.0})
                train_writer.add_summary(summary, i)
            # 模型训练后的测试数据模型评估
            print('test accuracy %g' % accuracy.eval(feed_dict={
                x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

            # 保存训练好的模型
            # 形参output_node_names用于指定输出的节点名称,output_node_names=['output']对应pre_num=tf.argmax(y,1,name="output"),
            output_graph_def = graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names=[
                'accuracy/output'])
            with tf.gfile.FastGFile(FLAGS.appModePath, mode='wb') as f:  # ’wb’中w代表写文件，b代表将数据以二进制方式写入文件。
                f.write(output_graph_def.SerializeToString())
        else:
            mode_file = tf.train.latest_checkpoint(FLAGS.modePath)
            saver.restore(sess, mode_file)
            # 模型读取后的测试数据模型评估
            print('test accuracy %g' % accuracy.eval(feed_dict={
                x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
        train_writer.close()
        test_writer.close()


def main():
    """
    训练初始化及调用模型训练
    """
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    if not tf.gfile.Exists(FLAGS.modePath):
        tf.gfile.MakeDirs(FLAGS.modePath)
    start(True)


if __name__ == '__main__':
    FLAGS = initData()
else:
    FLAGS = initData()
main()

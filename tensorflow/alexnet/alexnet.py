""" AlexNet for benchmark. 
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time
import sys
import numpy as np

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

IMAGE_SIZE = 224
NUM_CHANNELS = 3
NUM_LABELS = 1000
LEARNING_RATE = 0.1


# Flags for defining the tf.train.ClusterSpec
tf.app.flags.DEFINE_string("ps_hosts", "",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "",
                           "Comma-separated list of hostname:port pairs")

# Flags for defining the tf.train.Server
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_id", 0, "Index of task within the job")

tf.app.flags.DEFINE_integer('batch_size', 256,
                            """Batch size.""")

FLAGS = tf.app.flags.FLAGS


def synthetic_data(num_images):
    data = np.random.rand(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS) - 0.5
    labels = np.random.randint(0, 9, size=num_images)
    one_hot = np.zeros((labels.size, NUM_LABELS))
    one_hot[np.arange(labels.size),labels] = 1
    return data, one_hot


def print_activations(t):
  print(t.op.name, ' ', t.get_shape().as_list())


def get_graph():
    
    x = tf.placeholder(tf.float32, shape=[None, 
                                            IMAGE_SIZE,
                                            IMAGE_SIZE,
                                            NUM_CHANNELS])
    
    ### First layer ###
    
    with tf.name_scope('conv1') as scope:
        # Random parameters since this is only for benchmarking
        kernel = tf.Variable(tf.truncated_normal([11, 11, 3, 64], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                             trainable=True, name='biases')
        # First convolutional layer                     
        conv = tf.nn.conv2d(x, kernel, [1, 4, 4, 1], padding='SAME')
        bias = tf.nn.bias_add(conv, biases)
        # Relu
        relu = tf.nn.relu(bias, name=scope)
        radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
        # Response normalization
        lrn = tf.nn.local_response_normalization(relu,
                                                      depth_radius=radius,
                                                      alpha=alpha,
                                                      beta=beta,
                                                      bias=bias)
        # Max pool                                              
        pool1 = tf.nn.max_pool(lrn,
                             ksize=[1, 3, 3, 1],
                             strides=[1, 2, 2, 1],
                             padding='VALID',
                             name='pool1')
    
    
    
    ### Second layer ###
    
    with tf.name_scope('conv2') as scope:
        # Random parameters for benchmarking
        kernel = tf.Variable(tf.truncated_normal([5, 5, 64, 192], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        biases = tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32),
                             trainable=True, name='biases')
        # Second convolutional layer                     
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        bias = tf.nn.bias_add(conv, biases)
        # Relu
        relu = tf.nn.relu(bias, name=scope)
        # Response normalization
        radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
        lrn = tf.nn.local_response_normalization(relu,
                                                          depth_radius=radius,
                                                          alpha=alpha,
                                                          beta=beta,
                                                          bias=bias)
        # Max pool 
        pool2 = tf.nn.max_pool(lrn,
                             ksize=[1, 3, 3, 1],
                             strides=[1, 2, 2, 1],
                             padding='VALID',
                             name='pool2')
    
    
    
    ### Third layer ###
    with tf.name_scope('conv3') as scope:    
        # Random parameters for benchmarking
        kernel = tf.Variable(tf.truncated_normal([3, 3, 192, 384],
                                                 dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32),
                             trainable=True, name='biases')
        # Third convolutional layer
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        bias = tf.nn.bias_add(conv, biases)
        #Relu
        relu3 = tf.nn.relu(bias, name=scope)
    
    
    
    ### Fourth layer ###
    with tf.name_scope('conv4') as scope:
        # Random parameters for benchmarking
        kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 256],
                                                 dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable=True, name='biases')
        # Fourth convolutional layer
        conv = tf.nn.conv2d(relu3, kernel, [1, 1, 1, 1], padding='SAME')
        bias = tf.nn.bias_add(conv, biases)                                             
        # Relu
        relu4 = tf.nn.relu(bias, name=scope)
    
    
    
    ### Fifth layer ###
    with tf.name_scope('conv5') as scope:
        # Random parameters for benchmarking
        kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256],
                                                 dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable=True, name='biases')
        # Fifth convolutional layer
        conv = tf.nn.conv2d(relu4, kernel, [1, 1, 1, 1], padding='SAME')                                                                      
        bias = tf.nn.bias_add(conv, biases)
        # Relu
        relu = tf.nn.relu(bias, name='relu5')
        # Max pool
        # max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
        k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'

        maxpool5 = tf.nn.max_pool(relu, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)    

    
    
    ### Sixth layer. Fully connected. ###
    with tf.name_scope('fc6') as scope:
        weights = tf.Variable(tf.truncated_normal([9216, 4096], dtype=tf.float32, stddev=1e-1), name='fc6W')
        biases = tf.Variable(tf.constant(0.0, shape=[4096], dtype=tf.float32), trainable=True, name='fc6b')
        fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [-1, int(np.prod(maxpool5.get_shape()[1:]))]), weights, biases)
    
    
    ### Seventh layer. Fully connected ###
    with tf.name_scope('fc7') as scope:
        weights = tf.Variable(tf.truncated_normal([4096, 4096], dtype=tf.float32, stddev=1e-1), name='fc7W')
        biases = tf.Variable(tf.constant(0.0, shape=[4096], dtype=tf.float32), trainable=True, name='fc7b')
        fc7 = tf.nn.relu_layer(fc6, weights, biases)
    
        
    ### Eighth layer ###
    with tf.name_scope('fc8') as scope:
        weights = tf.Variable(tf.truncated_normal([4096, NUM_LABELS], dtype=tf.float32, stddev=1e-1), name='fc8W')
        biases = tf.Variable(tf.constant(0.0, shape=[NUM_LABELS], dtype=tf.float32), trainable=True, name='fc8b')
        fc8 = tf.nn.xw_plus_b(fc7, weights, biases)
    
    
    ### Probability. SoftMax ###
    prob = tf.nn.softmax(fc8)
    return prob, x




def main(_):
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")
    
    batch_size = FLAGS.batch_size

    # Create a cluster from the parameter server and worker hosts.
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    # Create and start a server for the local task.
    server = tf.train.Server(cluster,
                       job_name=FLAGS.job_name,
                       task_index=FLAGS.task_id)

    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":

        # Assigns ops to the local worker by default.
        with tf.device(tf.train.replica_device_setter(
            worker_device="/job:worker/task:%d" % FLAGS.task_id,
            cluster=cluster)):

            summary_op = tf.merge_all_summaries()
            
            y, x = get_graph()
            
            y_ = tf.placeholder(tf.float32, [None, NUM_LABELS])
            
            cross_entropy = tf.reduce_mean( -tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]) )
            
            global_step = tf.Variable(0)
            
            gradient_descent_opt = tf.train.GradientDescentOptimizer(LEARNING_RATE)
            
            num_workers = len(worker_hosts)
            sync_rep_opt = tf.train.SyncReplicasOptimizer(gradient_descent_opt, replicas_to_aggregate=num_workers,
                    replica_id=FLAGS.task_id, total_num_replicas=num_workers)
            
            train_op = sync_rep_opt.minimize(cross_entropy, global_step=global_step)
            
            init_token_op = sync_rep_opt.get_init_tokens_op()
            chief_queue_runner = sync_rep_opt.get_chief_queue_runner()
            
            #saver = tf.train.Saver()
            summary_op = tf.merge_all_summaries()

            init_op = tf.initialize_all_variables()
            saver = tf.train.Saver()
        
        is_chief=(FLAGS.task_id == 0)
            
        # Create a "supervisor", which oversees the training process.
        sv = tf.train.Supervisor(is_chief=(FLAGS.task_id == 0),
                                 init_op=init_op,
                                 summary_op=summary_op,
                                 saver=saver,
                                 global_step=global_step)

        # The supervisor takes care of session initialization, restoring from
        # a checkpoint, and closing when done or an error occurs.
        with sv.managed_session(server.target) as sess:
            
            if is_chief:
                sv.start_queue_runners(sess, [chief_queue_runner])
                sess.run(init_token_op)
            
            num_steps_burn_in = 10
            total_duration = 0
            total_duration_squared = 0
            
            step = 0
            while step <= 2000:
                sys.stdout.flush()
                batch_xs, batch_ys = synthetic_data(batch_size)
                train_feed = {x: batch_xs, y_: batch_ys}
                
                start_time = time.time()
                
                _, step = sess.run([train_op, global_step], feed_dict=train_feed)
                
                duration = time.time() - start_time

                examples_per_sec = batch_size / float(duration)
                format_str = ('Worker %d: %s: step %d, loss = NA'
                              '(%.1f examples/sec; %.3f  sec/batch)')

                if step > num_steps_burn_in:
                    print(format_str %
                            (FLAGS.task_id, datetime.now(), step,
                             examples_per_sec, duration))
                    sys.stdout.flush()
                else:
                    print('Not considering step %d (%.1f samples/sec)' %
                                    (step, examples_per_sec))
                    sys.stdout.flush()
                
                    
        sv.stop()

                                
if __name__ == "__main__":
    tf.app.run()

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

import resnet_model

IMAGE_SIZE = 224
NUM_CHANNELS = 3
NUM_LABELS = 1000
LEARNING_RATE = 0.5



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
tf.app.flags.DEFINE_integer('num_batches', 100,
                            """Number of batches to run.""")

FLAGS = tf.app.flags.FLAGS


def synthetic_data(batch_size):
    images = np.random.rand(batch_size, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS) - 0.5
    images = tf.cast(images, tf.float32)
    labels = np.random.randint(0, 9, size=batch_size)
    one_hot = np.zeros((labels.size, NUM_LABELS))
    one_hot[np.arange(labels.size),labels] = 1
    one_hot = tf.cast(one_hot, tf.float32)
    return images, one_hot


def main(_):
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")

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

            hps = resnet_model.HParams(batch_size=FLAGS.batch_size,
                                        num_classes=NUM_LABELS,
                                        min_lrn_rate=0.0001,
                                        lrn_rate=0.1,
                                        num_residual_units=5,
                                        use_bottleneck=True,
                                        weight_decay_rate=0.0002,
                                        relu_leakiness=0.1,
                                        optimizer='mom')

            mode = 'train'
            
            images, labels = synthetic_data(hps.batch_size)

            model = resnet_model.ResNet(hps, images, labels, mode)
            model.build_graph()
    
            cross_entropy = model.cost
            
            global_step = tf.Variable(0)
            
            gradient_descent_opt = tf.train.GradientDescentOptimizer(LEARNING_RATE)
            
            num_workers = len(worker_hosts)
            sync_rep_opt = tf.train.SyncReplicasOptimizer(gradient_descent_opt, replicas_to_aggregate=num_workers,
                    replica_id=FLAGS.task_id, total_num_replicas=num_workers)
            
            train_op = sync_rep_opt.minimize(cross_entropy, global_step=global_step)
            
            init_token_op = sync_rep_opt.get_init_tokens_op()
            chief_queue_runner = sync_rep_opt.get_chief_queue_runner()
            
            init_op = tf.initialize_all_variables()
        
        is_chief=(FLAGS.task_id == 0)
            
        # Create a "supervisor", which oversees the training process.
        sv = tf.train.Supervisor(is_chief=(FLAGS.task_id == 0),
                                 init_op=init_op,
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
            lrn_rate = 0.1
            
            while step <= 2000:
                
                start_time = time.time()
                
                _, step = sess.run([train_op, global_step], feed_dict={model.lrn_rate: lrn_rate})
                
                duration = time.time() - start_time
                
                examples_per_sec = hps.batch_size / float(duration)
                format_str = ('Worker %d: %s: step %d, loss = NA'
                              '(%.4f examples/sec; %.3f  sec/batch)')

                if step > num_steps_burn_in:
                    print(format_str %
                            (FLAGS.task_id, datetime.now(), step,
                             examples_per_sec, duration))
                    sys.stdout.flush()
                else:
                    print('Not considering burn-in step %d (%.4f samples/sec; %.3f  sec/batch)' %
                                    (step, examples_per_sec, duration))
                    sys.stdout.flush()
                    
        sv.stop()
                                
if __name__ == "__main__":
    tf.app.run()

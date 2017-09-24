#! /usr/bin/env python
# based on ideas in https://github.com/dennybritz/cnn-text-classification-tf

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import preprocessing
from model import CharCNN
from model_w2v import WordCNN

# Parameters
# ==================================================

# Model Hyperparameters
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 128, "Batch Size (default: 128)")
tf.flags.DEFINE_integer("num_epochs", 50, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 1000, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 5000, "Save model after this many steps (default: 100)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


# Data Preparation
# ==================================================

# Load data
print("Loading data...")
#x, y = preprocessing.load_data()
x, y = preprocessing.load_data_w2v()
# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]
# Split train/test set
n_dev_samples = 15000

x_train, x_dev = x_shuffled[:-n_dev_samples], x_shuffled[-n_dev_samples:]
y_train, y_dev = y_shuffled[:-n_dev_samples], y_shuffled[-n_dev_samples:]
print("Train/Test split: {:d}/{:d}".format(len(y_train), len(y_dev)))


# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        #cnn = CharCNN()
        cnn = WordCNN()
        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        '''
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.merge_summary(grad_summaries)
        '''
        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        '''
        loss_summary = tf.scalar_summary("loss", cnn.loss)
        acc_summary = tf.scalar_summary("accuracy", cnn.accuracy)
  
        
        # Train Summaries
        train_summary_op = tf.merge_summary([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.merge_summary([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph)
        '''
        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables())

        # Initialize all variables
        sess.run(tf.initialize_all_variables())

        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, loss, accuracy = sess.run(
                [train_op, global_step, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            if step % 100 == 0:
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            #train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            dev_size = len(x_batch)
            max_batch_size = 1000
            num_batches = int(dev_size/max_batch_size)
            acc = []
            losses = []
            p = []
            r = []
            print("Number of batches in test set is " + str(num_batches))
            for i in range(num_batches):
                tp = 0
                tn = 0
                fp = 0
                fn = 0         
                #for char-level cnn
                '''
                x_batch_dev, y_batch_dev = preprocessing.get_batched_one_hot(
                    x_batch, y_batch, i * max_batch_size, (i + 1) * max_batch_size)
                '''
                #for word vector cnn
                start =  i * max_batch_size
                end = min((i + 1) * max_batch_size, len(x_batch))
                x_batch_dev = x_batch[start:end]
                x_batch_dev = np.reshape(x_batch_dev, [len(x_batch_dev), 128, 64, 1])
                y_batch_dev = y_batch[start:end]

                fd = {
                  cnn.input_x: x_batch_dev,
                  cnn.input_y: y_batch_dev,
                  cnn.dropout_keep_prob: 1.0
                }
                
                step, loss, accuracy, predicted, ground_truth = sess.run(
                    [global_step, cnn.loss, cnn.accuracy, cnn.pred, cnn.real],
                    fd)
                    
                #metrics of class "[1, 0]"
                for j in range(1000):
                    if predicted[j] == 0 and ground_truth[j] == 0:
                        tp += 1
                    elif not predicted[j] ==0 and not ground_truth[j] == 0:
                        tn += 1
                    elif predicted[j] == 0 and not ground_truth[j] == 0:
                        fp += 1
                    else:
                        fn += 1
                        
                precision = 0
                recall = 0
                if (fp+fn+tp+tn) < 1000:
                    print("An error occured, total =",fp+fn+tp+tn)
                else:
                    pass
                    #precision = tp/(tp+fp)
                    #recall = tp/(tp+fn)

                acc.append(accuracy)
                losses.append(loss)
                p.append(precision)
                r.append(recall)
                time_str = datetime.datetime.now().isoformat()
                #print("TP:%d, TN:%d, FP:%d, FN:%d" % (tp, tn, fp, fn))
                print("batch " + str(i + 1) + " in test >>" +
                      " {}: loss {:g}, acc {:g}, precision {:g}, recall {:g}\n".format(time_str, loss, accuracy, precision, recall))
                      
                '''
                if writer:
                    writer.add_summary(summaries, step)
                '''   
            print("\nMean accuracy=" + str(sum(acc)/len(acc)))
            #print("Mean loss=" + str(sum(losses)/len(losses)))
            print("\nMean precision=" + str(sum(p)/len(p)))
            print("\nMean recall=" + str(sum(r)/len(r)))


        # Generate batches in one-hot-encoding format
        batches = preprocessing.batch_iter_w2v(x_train, y_train, FLAGS.batch_size, FLAGS.num_epochs)
        # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                dev_step(x_dev, y_dev)
                print("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))

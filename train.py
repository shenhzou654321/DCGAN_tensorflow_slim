#!/usr/bin/env python
# encoding: utf8
#
#       @author       : tcund@126.com
#       @file         : train.py
#       @date         : 2018/10/07 09:49
import tensorflow as tf
import dcgan
import data_provider

flags = tf.app.flags
flags.DEFINE_string("data_dir", "", "Directory for storing input data")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_integer("input_size", 64, "The size of image. The width is same to the height [64]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("sample_size", 64, "The size of sample z [64]")
flags.DEFINE_integer("gen_sample_dim", 100, "dim [100]")
flags.DEFINE_integer("layer_ratio", 16, "The convolution layer number. 2 ** L")
flags.DEFINE_integer("disc_hidden_dim", 64, "")
flags.DEFINE_integer("gen_hidden_dim", 64, "")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
FLAGS = flags.FLAGS

def main(_):
    sess = tf.Session()
    dcgan_obj = dcgan.DCGAN(
            sess,
            FLAGS.checkpoint_dir,
            FLAGS.gen_sample_dim,
            FLAGS.input_size,
            FLAGS.layer_ratio,
            FLAGS.disc_hidden_dim,
            FLAGS.gen_hidden_dim,
            FLAGS.learning_rate)
    dcgan_obj.build_network()
    dcgan_obj.init_model()
    dp = data_provider.DataProvider(FLAGS.data_dir, FLAGS.input_size)
    dp.load()
    for batch in dp.batchs(10000, FLAGS.batch_size):
        dcgan_obj.train(batch, FLAGS.sample_size)
    dcgan_obj.save_model()

if __name__ == '__main__':
    tf.app.run()

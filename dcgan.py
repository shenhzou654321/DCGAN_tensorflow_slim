#!/usr/bin/env python
# encoding: utf8
#
#       @author       : zhangduo@baidu.com
#       @file         : dcgan.py
#       @date         : 2018/10/06 10:27
import tensorflow as tf
import math
import numpy as np
slim = tf.contrib.slim

class DCGAN(object):
    def __init__(self, sess, ckpt_path, gen_sample_dim, img_size, ratio, df_dim, gf_dim, learning_rate = 2e-4):
        self.sess = sess
        self.ckpt_path = ckpt_path
        self.gen_sample_dim = gen_sample_dim
        self.img_size = img_size
        self.ratio = ratio
        self.df_dim = df_dim
        self.gf_dim = gf_dim
        self.learning_rate = learning_rate

    def discriminator(self, img_input, reuse, trainable):
        assert img_input.shape[1] == self.img_size
        with tf.variable_scope("discriminator", reuse=reuse):
          with slim.arg_scope([slim.conv2d, slim.fully_connected], trainable = trainable):
            cur_ratio = 1
            layer = img_input
            while cur_ratio < self.ratio:
                layer = slim.conv2d(layer, self.df_dim * cur_ratio, 5, 2, "SAME")
                cur_ratio *= 2
            flatten_layer = slim.flatten(layer)
            logit = slim.fully_connected(flatten_layer, 1, activation_fn = None, normalizer_fn = None)
            prob = tf.nn.sigmoid(logit)
            return prob, logit

    def generator(self, z):
        def conv_out_size(img_size, ratio):
            cur_size = img_size
            cur_ratio = 1
            while cur_ratio < ratio:
                cur_size = int(math.ceil(float(cur_size) / float(2)))
                cur_ratio *= 2
            return cur_size

        with tf.variable_scope("generator"):
            cur_ratio = int(self.ratio) / 2
            out_size = conv_out_size(self.img_size, self.ratio)
            logit = slim.fully_connected(z, self.gf_dim * cur_ratio * out_size * out_size, activation_fn = tf.nn.relu)
            layer = tf.reshape(logit, [-1, out_size, out_size, self.gf_dim * cur_ratio])
            while cur_ratio > 1:
                cur_ratio /= 2
                layer = slim.conv2d_transpose(layer, self.df_dim * cur_ratio, 5, 2, "SAME", activation_fn=tf.nn.relu)
            out_img = slim.conv2d_transpose(layer, 3, 5, 2, "SAME", activation_fn = tf.nn.tanh)
            assert out_img.shape[1] == self.img_size
            return out_img

    def build_network(self):
        self.img_input = tf.placeholder(tf.float32, [None, self.img_size, self.img_size, 3])
        self.z = tf.placeholder(tf.float32, [None, self.gen_sample_dim])
        self.is_training = tf.placeholder(tf.bool, ())
        self.d_step = tf.Variable(0, name="global_step", trainable=False)
        self.g_step = tf.Variable(0, name="global_step", trainable=False)
        batch_norm_params = {
            'decay': 0.9,
            'epsilon': 1e-5,
            'scale': True,
            'updates_collections': None,
            'is_training': self.is_training,
        }
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose, slim.fully_connected],
                weights_regularizer=slim.l2_regularizer(1e-4),
                weights_initializer=slim.variance_scaling_initializer(),
                activation_fn=tf.nn.leaky_relu,
                normalizer_fn=slim.batch_norm,
                normalizer_params=batch_norm_params):
            self.real_prob, self.real_logit = self.discriminator(self.img_input, False, True)
            #self.real_logit = tf.Print(self.real_logit, [tf.shape(self.real_logit), tf.shape(self.real_prob)], summarize=100)
            real_prob_sum = tf.summary.histogram("real_prob", self.real_prob)
            self.gen_img = self.generator(self.z)
            self.fake_prob, self.fake_logit = self.discriminator(self.gen_img, True, True)
            #self.fake_logit = tf.Print(self.fake_logit, [tf.shape(self.z), tf.shape(self.gen_img)], summarize=100)
            fake_prob_sum = tf.summary.histogram("fake_prob", self.fake_prob)
            self.d_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.real_logit, labels=tf.ones_like(self.real_logit)))
            self.d_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fake_logit, labels=tf.zeros_like(self.fake_logit)))
            d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_real_loss)
            d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_fake_loss)
            self.d_loss = self.d_real_loss + self.d_fake_loss
            #self.d_loss = self.d_fake_loss
            d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
            self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fake_logit, labels=tf.ones_like(self.fake_logit)))
            g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)

            t_vars = tf.trainable_variables()

            d_vars = [var for var in t_vars if 'discriminator' in var.name]
            g_vars = [var for var in t_vars if 'generator' in var.name]

            d_optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5)
            g_optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5)
            self.d_train_step = slim.learning.create_train_op(self.d_loss, d_optimizer, global_step=self.d_step, variables_to_train = d_vars)
            self.g_train_step = slim.learning.create_train_op(self.g_loss, g_optimizer, global_step=self.g_step, variables_to_train = g_vars)

            self.d_sum = tf.summary.merge([d_loss_real_sum, d_loss_fake_sum, d_loss_sum, real_prob_sum, fake_prob_sum])
            self.g_sum = tf.summary.merge([g_loss_sum])

    def init_model(self):
        self.saver = tf.train.Saver()
        state = tf.train.get_checkpoint_state(self.ckpt_path)
        if state is None:
            self.sess.run( tf.global_variables_initializer())
        else:
            self.saver.restore(self.sess, tf.train.latest_checkpoint(self.ckpt_path))
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

    def save_model(self):
        self.saver.save(self.sess, self.ckpt_path)

    def train(self, imgs, sample_len):
        model_vars = tf.global_variables()
        sample_z = np.random.uniform(-1, 1, size=(sample_len, self.gen_sample_dim)).astype(np.float32)
        ds, _, d_sum = self.sess.run([self.d_step, self.d_train_step, self.d_sum], feed_dict = {self.img_input: imgs, self.z: sample_z, self.is_training: True})
        self.writer.add_summary(d_sum, ds)
        #"""
        gs, _, g_sum = self.sess.run([self.g_step, self.g_train_step, self.g_sum], feed_dict = {self.img_input: np.zeros([1, self.img_size, self.img_size, 3]), self.z: sample_z, self.is_training: True})
        self.writer.add_summary(g_sum, gs)
        gs, _, g_sum = self.sess.run([self.g_step, self.g_train_step, self.g_sum], feed_dict = {self.img_input: np.zeros([1, self.img_size, self.img_size, 3]), self.z: sample_z, self.is_training: True})
        self.writer.add_summary(g_sum, gs)
        loss_d_real, loss_d_fake, loss_g = self.sess.run([self.d_real_loss, self.d_fake_loss, self.g_loss], feed_dict = {self.img_input: imgs, self.z: sample_z, self.is_training: True})

        print("ds:%d gs:%d -> d_real_loss: %.6f, d_fake_loss: %.6f g_loss: %.6f" % (ds, gs, loss_d_real, loss_d_fake, loss_g))
        #"""

    def predict(self, imgs):
        prob, = self.sess.run([self.real_prob], feed_dict = {self.img_input: imgs})
        return prob

if __name__ == '__main__':
    pass

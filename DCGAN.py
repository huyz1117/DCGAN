# -*- coding: utf-8 -*-
# @Author: huyz1117
# @Date:   2019-01-23 21:36:23
# @Last Modified by:   huyz1117
# @Last Modified time: 2019-01-24 16:45:11
# Reference: Unsupervised Representation Learning With Deep Convolutional Generative Adversarial Networks

import os
import time
import math
import numpy as np
import tensorflow as tf

from ops import *
from utils import *

def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))

class DCGAN():

	def __init__(self, sess, args):
		self.sess = sess
		self.c_dim = args.c_dim
		self.batch_size = args.batch_size
		self.image_size = args.image_size
		self.z_dim = args.z_dim
		self.learning_rate = args.learning_rate
		self.log_dir = args.log_dir
		self.checkpoint_dir = args.checkpoint_dir
		self.dataset = args.dataset
		self.epoch = args.epoch
		self.print_freq = args.print_freq
		self.sample_num = args.sample_num
		self.test_num = args.test_num
		self.sample_dir = args.sample_dir
		self.result_dir = args.result_dir
		self.gf_dim = 64
		self.df_dim = 64
        
		self.output_height = args.output_height
		self.output_width = args.output_width
		self.model_name = 'DCGAN'
        
		self.d_bn1 = batch_norm(name='d_bn1')
		self.d_bn2 = batch_norm(name='d_bn2')
		self.d_bn3 = batch_norm(name='d_bn3')

		self.g_bn0 = batch_norm(name='g_bn0')
		self.g_bn1 = batch_norm(name='g_bn1')
		self.g_bn2 = batch_norm(name='g_bn2')
		self.g_bn3 = batch_norm(name='g_bn3')

		self.data_X = load_data(self.dataset)
		#print(self.data_X)
		self.num_batches = len(self.data_X) // self.batch_size

	def generator(self, z, is_training=True, reuse=False, scope='generator'):
		with tf.variable_scope(scope, reuse=reuse):

			s_h, s_w = self.output_height, self.output_width
			s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
			s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
			s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
			s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)


			self.z_ = fully_connected(z, output_units=64 * 8 * s_h16 * s_w16, scope='project')
			h0 = tf.reshape(self.z_, [-1, s_h16, s_w16, self.gf_dim * 8])
			h0 = tf.nn.relu(h0)

			self.h0 = tf.reshape(
				self.z_, [-1, s_h16, s_w16, self.gf_dim * 8])
			h0 = tf.nn.relu(self.g_bn0(self.h0))

			self.h1, self.h1_w, self.h1_b = deconv2d(
				h0, [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h1', with_w=True)
			h1 = tf.nn.relu(self.g_bn1(self.h1))

			h2, self.h2_w, self.h2_b = deconv2d(
				h1, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h2', with_w=True)
			h2 = tf.nn.relu(self.g_bn2(h2))

			h3, self.h3_w, self.h3_b = deconv2d(
				h2, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h3', with_w=True)
			h3 = tf.nn.relu(self.g_bn3(h3))

			h4, self.h4_w, self.h4_b = deconv2d(
				h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4', with_w=True)

			return tf.nn.tanh(h4)

	def discriminator(self, x, is_training=True, reuse=False, scope='discriminator'):
		with tf.variable_scope(scope, reuse=reuse):

			x = conv2d(x, output_channels=64, scope='d_conv0')
			x = lrelu(x)

			x = conv2d(x, output_channels=64*2, scope='d_conv1')
			#x = batch_norm(x, name='d_bn1')
			#x = lrelu(x)
			x = tf.nn.relu(self.d_bn1(x))

			x = conv2d(x, output_channels=64*4, scope='d_conv2')
			#x = batch_norm(x, name='d_bn2')
			#x = lrelu(x)
			x = tf.nn.relu(self.d_bn2(x))


			x = conv2d(x, output_channels=64*8, scope='d_conv3')
			#x = batch_norm(x, name='d_bn3')
			#x = lrelu(x)
			x = tf.nn.relu(self.d_bn3(x))


			x = flatten(x)
			x = fully_connected(x, 1, scope='fc')


			return x

	def load_checkpoint(self, checkpoint_dir):
		print('[*] Reading checkpoint ...')

		ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			self.saver.restore(self.sess, ckpt.model_checkpoint_path)

			ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
			counter = int(ckpt_name.split('-')[-1])
			print('[*] Success to load {}'.format(ckpt_name))

			return True, counter
		else:
			print('[*] Failed to find a checkpoint!')
			return False, 0

	def save(self, checkpoint_dir, step):
    	#checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

		if not os.path.exists(checkpoint_dir):
		    os.makedirs(checkpoint_dir)

		self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name+'.model'), global_step=step)

	def visualize_results(self, epoch):
		tot_num_samples = min(self.sample_num, self.batch_size)
		image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

		""" random condition, random noise """

		z_sample = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))

		samples = self.sess.run(self.fake_images, feed_dict={self.z: z_sample})

		#sample_dir = os.path.join(self.sample_dir, self.model_dir)
		check_folder(self.sample_dir)

		save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
		            self.sample_dir + '/' + self.model_name + '_epoch%03d' % epoch + '_test_all_classes.png')

	def test(self):
		tf.global_variables_initializer().run()

		self.saver = tf.train.Saver()
		could_load, checkpoint_counter = self.load_checkpoint(self.checkpoint_dir)
		#result_dir = os.path.join(self.result_dir, self.model_dir)
		check_folder(self.result_dir)

		if could_load:
		    print(" [*] Load SUCCESS")
		else:
		    print(" [!] Load failed...")

		tot_num_samples = min(self.sample_num, self.batch_size)
		image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

		""" random condition, random noise """

		for i in range(self.test_num) :
		    z_sample = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))

		    samples = self.sess.run(self.fake_images, feed_dict={self.z: z_sample})

		    save_images(samples[:image_frame_dim * image_frame_dim, :, :, :],
		                [image_frame_dim, image_frame_dim],
		                self.result_dir + '/' + self.model_name + '_test_all_classes_{}.png'.format(i))


	def build_model(self):
		''' Graph input '''
		# ireal mages
		self.real_images = tf.placeholder(tf.float32, shape=[self.batch_size, self.image_size, self.image_size, self.c_dim], name='real_images')
		# noise vector
		self.z = tf.placeholder(tf.float32, shape=[self.batch_size, self.z_dim], name='noise_z')

		''' Loss function '''
		# output of D for real images
		D_real_logits = self.discriminator(self.real_images, is_training=True, reuse=False)

		# output of D for fake images
		G = self.generator(self.z, is_training=True, reuse=False)
		D_fake_logits = self.discriminator(G, is_training=True, reuse=True)

		# get loss for discriminator
		self.d_loss = discriminator_loss(real=D_real_logits, fake=D_fake_logits)

		# get loss for generator loss
		self.g_loss = generator_loss(fake=D_fake_logits)

		# summary
		self.d_summ = tf.summary.scalar('d_loss', self.d_loss)
		self.g_summ = tf.summary.scalar('g_loss', self.g_loss)
		#self.merge = tf.summary.merge_all()

		''' training '''
		D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
		G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')

		# optimizer
		self.D_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.d_loss, var_list=D_vars)
		self.G_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.g_loss, var_list=G_vars)

		''' testing'''
		# for test
		self.fake_images = self.generator(self.z, is_training=False, reuse=True)

	def train(self):
		tf.global_variables_initializer().run()

		# saver to save model
		self.saver = tf.train.Saver()



		# summary writer
		
		self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

		# graph input for visualizing the training result
		self.sample_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim])

		# restore checkpoint if it exists
		could_load, checkpoint_counter = self.load_checkpoint(self.checkpoint_dir)
		if could_load:
			start_epoch = int(checkpoint_counter / self.num_batches)
			start_batch_id = checkpoint_counter - start_epoch * self.num_batches
			counter = checkpoint_counter

			print('[*] Load success ^_^')
		else:
			start_epoch = 0
			start_batch_id = 0
			counter = 1

			print('[!] Load failed ...')

		start_time = time.time()
		for epoch in range(start_epoch, self.epoch):
			for idx in range(start_batch_id, self.num_batches):
				batch_images = self.data_X[idx*self.batch_size: (idx+1)*self.batch_size]
				batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim])

				train_fedd_dict = {self.real_images: batch_images, self.z: batch_z}

				# update D network
				_, summ, d_val_loss = self.sess.run([self.D_optimizer, self.d_summ, self.d_loss],
													feed_dict=train_fedd_dict)
				self.writer.add_summary(summ, counter)
				# update G network
				# run generator 5 to ensure that d_loss do not go to zero, different from the paper
				for _ in range(5):
					_, summ, g_val_loss = self.sess.run([self.G_optimizer, self.g_summ, self.g_loss],
													feed_dict=train_fedd_dict)
					self.writer.add_summary(summ, counter)

				# training status
				counter += 1
				print('[%d][%2d/%2d][%4d/%4d] time: %.4f d_loss: %.8f g_loss: %.8f'\
					%(counter-1, epoch+1, self.epoch, idx, self.num_batches, time.time()-start_time, d_val_loss, g_val_loss))

                # save training results for every 300 steps
				if np.mod(counter, self.print_freq) == 0:
					samples = self.sess.run(self.fake_images, feed_dict={self.z: self.sample_z})
					tot_num_samples = min(self.sample_num, self.batch_size)
					manifold_h = int(np.floor(np.sqrt(tot_num_samples)))
					manifold_w = int(np.floor(np.sqrt(tot_num_samples)))

                    #sample_dir = os.path.join(self.sample_dir, self.model_dir)
					check_folder(self.sample_dir)

					save_images(samples[:manifold_h * manifold_w, :, :, :],
					            [manifold_h, manifold_w],
					            './' + sample_dir + '/' + self.model_name + '_train_{:02d}_{:04d}.png'.format(epoch, idx))
			start_batch_id = 0

		self.save(self.checkpoint_dir, counter-1)



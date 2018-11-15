from operation import *
from skimage import io,color,img_as_float32
from PIL import Image
from random import shuffle

import tensorflow as tf
import skimage
import glob
import numpy as np
import matplotlib.pyplot as plt

batch_size = 2
input_folder = "inputs/"
train_iter = 2000

H = 256
W = 256

X = tf.placeholder(tf.float32, [None, H, W, 1], "input")
Y = tf.placeholder(tf.float32, [None, H, W, 2], "output")
is_training = tf.placeholder(tf.bool, name = "is_training")

# conv1
net1_1 = tf.nn.relu(conv("conv1_1", X, 64))
net1_2 = tf.nn.relu(conv("conv1_2", net1_1, 64, batch_norm = True, is_training = is_training))

# conv2
net2_d = net1_2[:, ::2, ::2, :]
net2_1 = tf.nn.relu(conv("conv2_1", net2_d, 128))
net2_2 = tf.nn.relu(conv("conv2_2", net2_1, 128, batch_norm = True, is_training = is_training))

# conv3
net3_d = net2_2[:, ::2, ::2, :]
net3_1 = tf.nn.relu(conv("conv3_1", net3_d, 256))
net3_2 = tf.nn.relu(conv("conv3_2", net3_1, 256))
net3_3 = tf.nn.relu(conv("conv3_3", net3_2, 256, batch_norm = True, is_training = is_training))

# conv4
net4_d = net3_3[:, ::2, ::2, :]
net4_1 = tf.nn.relu(conv("conv4_1", net4_d, 512))
net4_2 = tf.nn.relu(conv("conv4_2", net4_1, 512))
net4_3 = tf.nn.relu(conv("conv4_3", net4_2, 512, batch_norm = True, is_training = is_training))

# conv5
net5_1 = tf.nn.relu(conv("conv5_1", net4_3, 512, 2))
net5_2 = tf.nn.relu(conv("conv5_2", net5_1, 512, 2))
net5_3 = tf.nn.relu(conv("conv5_3", net5_2, 512, 2, batch_norm = True, is_training = is_training))

# conv6
net6_1 = tf.nn.relu(conv("conv6_1", net5_3, 512, 2))
net6_2 = tf.nn.relu(conv("conv6_2", net6_1, 512, 2))
net6_3 = tf.nn.relu(conv("conv6_3", net6_2, 512, 2, batch_norm = True, is_training = is_training))

# conv7
net7_1 = tf.nn.relu(conv("conv7_1", net6_3, 512))
net7_2 = tf.nn.relu(conv("conv7_2", net7_1, 512))
net7_3 = tf.nn.relu(conv("conv7_3", net7_2, 512, batch_norm = True, is_training = is_training))

# conv8
net8_u = deconv("conv8_u", net7_3, 256)
net8_c = tf.concat([net8_u, net3_3], 3)
net8_1 = tf.nn.relu(conv("conv8_1", net8_c, 256))
net8_2 = tf.nn.relu(conv("conv8_2", net8_1, 256))
net8_3 = tf.nn.relu(conv("conv8_3", net8_2, 256, batch_norm = True, is_training = is_training))

# conv9
net9_u = deconv("conv9_u", net8_3, 128)
net9_c = tf.concat([net9_u, net2_2], 3)
net9_1 = tf.nn.relu(conv("conv9_1", net9_c, 128))
net9_2 = tf.nn.relu(conv("conv9_2", net9_1, 128, batch_norm = True, is_training = is_training))

# conv10
net10_u = deconv("conv10_u", net9_2, 128)
net10_c = tf.concat([net10_u, net1_2], 3)
net10_1 = tf.nn.relu(conv("conv10_1", net10_c, 128))
net10_2 = tf.nn.relu(conv("conv10_2", net10_1, 128))

net_result = tf.nn.tanh(conv("conv_result", net10_2, 2, filter_size = 1)) * 110

loss = tf.reduce_mean(tf.losses.huber_loss(Y, net_result))

# actual training
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
with tf.Session(config=config) as sess:
	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	with tf.control_dependencies(update_ops):
		train = tf.train.AdamOptimizer().minimize(loss)
	sess.run(tf.global_variables_initializer())

	file_names = glob.glob(input_folder + "*.png")
	inputs = []
	outputs = []

	for file_name in file_names:
		rgb = io.imread(file_name)
		lab = np.reshape(img_as_float32(color.rgb2lab(rgb)), [H,W,3])
		inputs.append(lab[:,:,:1])
		outputs.append(lab[:,:,1:3])

	test_pair = {"input": [inputs[0]], "output": [outputs[0]]}

	for epoch in range(train_iter):
		pairs = list(zip(inputs, outputs))
		shuffle(pairs)
		inputs, outputs = zip(*pairs)

		for i in range(0, len(pairs), batch_size):
			_ = sess.run([train], feed_dict = {X:inputs[i:i + batch_size], Y:outputs[i:i + batch_size], is_training:True})
		
		if epoch % 50 == 0:
			_loss, _result = sess.run([loss, net_result], feed_dict = {X:test_pair["input"], Y:test_pair["output"], is_training:True})
			print("Epoch #" + str(epoch) + " loss: " + str(_loss))
			final_lab = np.concatenate((test_pair["input"], _result), axis = 3)
			final_lab = np.reshape(final_lab,[H,W,3])
			final_rgb = color.lab2rgb(final_lab)
			skimage.io.imsave("./images/{0:05d}.png".format(epoch), final_rgb)
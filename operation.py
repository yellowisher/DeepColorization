import tensorflow as tf

def conv(name, input, num_filters, dilations = 1, filter_size = 3, batch_norm = False, is_training = False):
	#is_training = tf.placeholder("") get is_training by tf.placeholder?
	with tf.variable_scope(name):
		weight = tf.get_variable("weight", [filter_size, filter_size, input.get_shape().as_list()[-1], num_filters], initializer = tf.contrib.layers.xavier_initializer())
		net = tf.nn.conv2d(input, weight, [1, 1, 1, 1], "SAME", dilations = [1, dilations, dilations, 1], name = name)
		
		if batch_norm:
			net = tf.contrib.layers.batch_norm(net, is_training = is_training)
		else:
			bias = tf.get_variable("bias", [num_filters], initializer = tf.contrib.layers.xavier_initializer())
			net += bias
		return net

def deconv(name, input, num_filters, strides = 2, filter_size = 4):
	with tf.variable_scope(name):
		shape_list = input.get_shape().as_list()
		weight = tf.get_variable("weight", [filter_size, filter_size, num_filters, shape_list[-1]], initializer = tf.contrib.layers.xavier_initializer())
		batch_size = tf.shape(input)[0]
		out_shape = tf.stack([batch_size, shape_list[1] * strides, shape_list[1] * strides, num_filters])
		net = tf.nn.conv2d_transpose(input, weight, out_shape, [1, strides, strides, 1])

		bias = tf.get_variable("bias", [num_filters], initializer = tf.contrib.layers.xavier_initializer())
		net += bias
		return net

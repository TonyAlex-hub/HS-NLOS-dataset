from vgg import vgg16
import segnet as NN
import tensorflow as tf
import numpy as np
from skimage import io
from skimage.transform import resize
from matplotlib import pyplot as plt
from imagenet_classes import class_names
#from scipy.misc import imread, imresize
import cv2 as cv
flags = tf.app.flags
flags.DEFINE_string("input", "/data1/cmy/data/road_aerial/road_1500_1500_128_128_121_really/0s.tif", "Path to input image ['laska.png']")
flags.DEFINE_string("output", "/data1/cmy/heatmap/roda_128_0.jpg", "Path to input image ['laska_save.png']")
flags.DEFINE_string("layer_name", "logits", "Layer till which to backpropagate ['pool5']")

FLAGS = flags.FLAGS


def load_image(img_path):
	print("Loading image")
	img = cv.imread(img_path)
	img = cv.resize(img, (128, 128))
	# Converting shape from [224,224,3] tp [1,224,224,3]
	x = np.expand_dims(img, axis=0)
	# Converting RGB to BGR for VGG
	x = x[:,:,:,::-1]
	return x, img


def grad_cam(x, nn, sess, layer_name):
	print("Setting gradients to 1 for target class and rest to 0")
	# Conv layer tensor [?,7,7,512]
	conv_layer = nn.layers[layer_name]
	# [1000]-D tensor with target class index set to 1 and rest as 0
	# one_hot = tf.sparse_to_dense(predicted_class, [nb_classes], 1.0)
	# signal = tf.multiply(vgg.layers['fc3'], one_hot)
	# loss = tf.reduce_mean(signal)

	# grads = tf.gradients(loss, conv_layer)[0]
	# Normalizing the gradients
	#norm_grads = tf.div(grads, tf.sqrt(tf.reduce_mean(tf.square(grads))) + tf.constant(1e-5))

	output = sess.run([conv_layer], feed_dict={nn.input_tensor: x})
	output = output[0]           # [7,7,512]
	#grads_val = grads_val[0]	 # [7,7,512]

	#weights = np.mean(grads_val, axis = (0, 1)) 			# [512]
	cam = np.ones(output.shape[0 : 2], dtype = np.float32)	# [7,7]
	cam =output
	# Taking a weighted average
	# for i, w in enumerate(weights):
	#     cam += 1* output[:, :, i]#本来是w*output[:, :, i]

	# Passing through ReLU
	cam = np.maximum(cam, 0)
	cam = cam / np.max(cam)
	cam = resize(cam, (128,128))

	# Converting grayscale to 3-D
	cam3 = np.expand_dims(cam, axis=2)
	cam3 = np.tile(cam3,[1,1,3])

	return cam3


def main(_):
	x, img = load_image(FLAGS.input)

	sess = tf.Session()

	print("\nLoading Vgg")
	imgs = tf.placeholder(tf.float32, [None, 128, 128, 3], name='images')
	nn = NN.SegNet()
	nn.build(imgs)
	sess = tf.Session()
	saver, sess = nn.load_model(sess,'/data1/cmy/save/rich_road_segnet_1e-7', '/data1/cmy/save/rich_road_segnet_1e-7')
	variable_inited = tf.global_variables()
	sess.run(tf.variables_initializer(variable_inited))
	saver, sess = nn.load_model(sess, '/data1/cmy/save/rich_road_segnet_1e-7', '/data1/cmy/save/rich_road_segnet_1e-7')
	print("\nFeedforwarding")
	sess.run(nn.output, feed_dict={imgs: x})[0]
	logits=sess.run(nn.logits, feed_dict={imgs: x})[0]
	print('\nTop 5 classes are')
	layer_name = FLAGS.layer_name

	#cam = tf.reduce_sum(cam, 2)
	cam = np.ones(logits.shape[0: 2], dtype=np.float32)  # [7,7]
	# Taking a weighted average
	for i in range(0,1):
		cam +=  logits[:, :, i]
	cam=np.asarray(cam)
	cam = np.maximum(cam, 0)
	cam = cam / np.max(cam)
	cam = resize(cam, (128, 128))

	cam3 = np.expand_dims(cam, axis=2)
	cam3 = np.tile(cam3, [1, 1, 3])
	img = img.astype(float)
	img /= img.max()

	# Superimposing the visualization with the image.
	new_img = img+3*cam3
	new_img /= new_img.max()
	io.imshow(cam3)
	plt.show()
	io.imsave(FLAGS.output, new_img)

if __name__ == '__main__':
	tf.app.run()


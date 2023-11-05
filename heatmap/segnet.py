
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
import layers_object
import gpu_train_segnet
args = gpu_train_segnet.Create_args()
#class_num
class SegNet:
    def __init__(self, fcn_net_npy_path=None, trainable=True):

        self.bayes = True
        self.prob = None
        self.is_training_pl = tf.cast(True, tf.bool)
        self.vgg_param_dict=None
        self.use_vgg=tf.cast(False, tf.bool)
        self.convlayers()
        self.input_tensor=()
        if fcn_net_npy_path is not None:
            self.data_dict = np.load(fcn_net_npy_path, encoding='latin1').item()
            # print('data dict is like',self.data_dict)
        else:
            self.data_dict = None
        self.var_dict = {}
        self.trainable = trainable

    def convlayers(self):
        self.parameters = []
        self.layers = {}
    def build(self, input_tensor):
        #input_tensor=self.input_tensor
        # first box of convolution layer,each part we do convolution two times, so we have conv1_1, and conv1_2
        self.conv1_1 = layers_object.conv_layer(input_tensor, "conv1_1", [3, 3, 3, 64])
        self.conv1_2 = layers_object.conv_layer(self.conv1_1, "conv1_2", [3, 3, 64, 64])
        self.pool1, self.pool1_index, self.shape_1 = layers_object.max_pool(self.conv1_2, 'pool1')

        # Second box of convolution layer(4)
        self.conv2_1 = layers_object.conv_layer(self.pool1, "conv2_1", [3, 3, 64, 128])
        self.conv2_2 = layers_object.conv_layer(self.conv2_1, "conv2_2", [3, 3, 128, 128])
        self.pool2, self.pool2_index, self.shape_2 = layers_object.max_pool(self.conv2_2, 'pool2')

        # Third box of convolution layer(7)
        self.conv3_1 = layers_object.conv_layer(self.pool2, "conv3_1", [3, 3, 128, 256])
        self.conv3_2 = layers_object.conv_layer(self.conv3_1, "conv3_2", [3, 3, 256, 256])
        self.conv3_3 = layers_object.conv_layer(self.conv3_2, "conv3_3", [3, 3, 256, 256])
        self.pool3, self.pool3_index, self.shape_3 = layers_object.max_pool(self.conv3_3, 'pool3')

        # Fourth box of convolution layer(10)
        if self.bayes:
            self.dropout1 = tf.layers.dropout(self.pool3, rate=0.5, name="dropout1")
            self.conv4_1 = layers_object.conv_layer(self.dropout1, "conv4_1", [3, 3, 256, 512])
        else:
            self.conv4_1 = layers_object.conv_layer(self.pool3, "conv4_1", [3, 3, 256, 512])
        self.conv4_2 = layers_object.conv_layer(self.conv4_1, "conv4_2", [3, 3, 512, 512])
        self.conv4_3 = layers_object.conv_layer(self.conv4_2, "conv4_3", [3, 3, 512, 512])
        self.pool4, self.pool4_index, self.shape_4 = layers_object.max_pool(self.conv4_3, 'pool4')

        #Fifth box of convolution layers(13)
        if self.bayes:
            self.dropout2 = tf.layers.dropout(self.pool4, rate=0.5, name="dropout2")
            self.conv5_1 = layers_object.conv_layer(self.dropout2, "conv5_1", [3, 3, 512, 512])
        else:
            self.conv5_1 = layers_object.conv_layer(self.pool4, "conv5_1", [3, 3, 512, 512])
        self.conv5_2 = layers_object.conv_layer(self.conv5_1, "conv5_2", [3, 3, 512, 512], )
        self.conv5_3 = layers_object.conv_layer(self.conv5_2, "conv5_3", [3, 3, 512, 512])
        self.pool5, self.pool5_index, self.shape_5 = layers_object.max_pool(self.conv5_3, 'pool5')

        # ---------------------So Now the encoder process has been Finished--------------------------------------#
        # ------------------Then Let's start Decoder Process-----------------------------------------------------#

        # First box of deconvolution layers(3)
        if self.bayes:
            self.dropout3 = tf.layers.dropout(self.pool5, rate=0.5, name="dropout3")
            self.deconv5_1 = layers_object.up_sampling(self.dropout3, self.pool5_index, self.shape_5, args.batchsize,
                                         name="unpool_5")
        else:
            self.deconv5_1 = layers_object.up_sampling(self.pool5, self.pool5_index, self.shape_5,  args.batchsize,
                                         name="unpool_5")
        self.deconv5_2 = layers_object.conv_layer(self.deconv5_1, "deconv5_2", [3, 3, 512, 512], self.is_training_pl)
        self.deconv5_3 = layers_object.conv_layer(self.deconv5_2, "deconv5_3", [3, 3, 512, 512], self.is_training_pl)
        self.deconv5_4 = layers_object.conv_layer(self.deconv5_3, "deconv5_4", [3, 3, 512, 512], self.is_training_pl)
        #Second box of deconvolution layers(6)
        if self.bayes:
            self.dropout4 = tf.layers.dropout(self.deconv5_4, rate=0.5,name="dropout4")
            self.deconv4_1 = layers_object.up_sampling(self.dropout4, self.pool4_index, self.shape_4, args.batchsize,
                                         name="unpool_4")
        else:
            self.deconv4_1 = layers_object.up_sampling(self.deconv5_4, self.pool4_index, self.shape_4, args.batchsize,
                                         name="unpool_4")
        self.deconv4_2 = layers_object.conv_layer(self.deconv4_1, "deconv4_2", [3, 3, 512, 512], self.is_training_pl)
        self.deconv4_3 = layers_object.conv_layer(self.deconv4_2, "deconv4_3", [3, 3, 512, 512], self.is_training_pl)
        self.deconv4_4 = layers_object.conv_layer(self.deconv4_3, "deconv4_4", [3, 3, 512, 256], self.is_training_pl)
        #Third box of deconvolution layers(9)
        if self.bayes:
            self.dropout5 = tf.layers.dropout(self.deconv4_4, rate=0.5
                                              , name="dropout5")
            self.deconv3_1 = layers_object.up_sampling(self.dropout5, self.pool3_index, self.shape_3, args.batchsize,
                                         name="unpool_3")
        else:
            self.deconv3_1 = layers_object.up_sampling(self.deconv4_4, self.pool3_index, self.shape_3, args.batchsize,
                                         name="unpool_3")
        self.deconv3_2 = layers_object.conv_layer(self.deconv3_1, "deconv3_2", [3, 3, 256, 256], self.is_training_pl)
        self.deconv3_3 = layers_object.conv_layer(self.deconv3_2, "deconv3_3", [3, 3, 256, 256], self.is_training_pl)
        self.deconv3_4 = layers_object.conv_layer(self.deconv3_3, "deconv3_4", [3, 3, 256, 128], self.is_training_pl)
        #Fourth box of deconvolution layers(11)
        if self.bayes:
            self.dropout6 = tf.layers.dropout(self.deconv3_4, rate=0.5,
                                               name="dropout6")
            self.deconv2_1 = layers_object.up_sampling(self.dropout6, self.pool2_index, self.shape_2,args.batchsize,
                                         name="unpool_2")
        else:
            self.deconv2_1 = layers_object.up_sampling(self.deconv3_4, self.pool2_index, self.shape_2, args.batchsize,
                                         name="unpool_2")
        self.deconv2_2 = layers_object.conv_layer(self.deconv2_1, "deconv2_2", [3, 3, 128, 128], self.is_training_pl)
        self.deconv2_3 = layers_object.conv_layer(self.deconv2_2, "deconv2_3", [3, 3, 128, 64], self.is_training_pl)
        # Fifth box of deconvolution layers(13)
        self.deconv1_1 = layers_object.up_sampling(self.deconv2_3, self.pool1_index, self.shape_1, args.batchsize,
                                     name="unpool_1")
        self.deconv1_2 = layers_object.conv_layer(self.deconv1_1, "deconv1_2", [3, 3, 64, 64], self.is_training_pl)
        self.deconv1_3 = layers_object.conv_layer(self.deconv1_2, "deconv1_3", [3, 3, 64, 64], self.is_training_pl)

        with tf.variable_scope('conv_classifier') as scope:
            self.kernel = layers_object.variable_with_weight_decay('weights', initializer=layers_object.initialization(1,64),
                                                     shape=[1, 1, 64, args.class_num], wd=False)
            self.conv = tf.nn.conv2d(self.deconv1_3, self.kernel, [1, 1, 1, 1], padding='SAME')
            self.biases = layers_object.variable_with_weight_decay('biases', tf.constant_initializer(0.0),
                                                     shape=[args.class_num], wd=False)
            self.logits = tf.nn.bias_add(self.conv, self.biases, name=scope.name)
            self.layers['logits'] = self.logits
            self.output=tf.nn.softmax(self.logits)
            tf.summary.histogram('self.output_', self.output)
    def save_npy(self, sess, npy_path):
        saver = tf.train.Saver()
        # saver.save(sess,npy_path,write_meta_graph=False)
        saver.save(sess, npy_path)
        print("file saved")
        return npy_path

    def load_model(self, sess, meta, checkpoint):
        #saver=tf.train.import_meta_graph(meta)
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir=checkpoint))
        return saver, sess


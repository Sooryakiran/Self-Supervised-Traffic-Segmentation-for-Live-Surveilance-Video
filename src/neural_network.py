import tensorflow as tf
import numpy as np

class ImageSegmenter:
    def __init__(self, learning_rate = 1e-3, image_size = [500, 500, 3], batch_size = 2):
        """
        The class constructor

        """
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.image_size = image_size
        self.__graph = tf.Graph()
        with self.__graph.as_default():

            """
            Define the placeholders for training

            """
            self.train_x_placeholder = tf.placeholder(shape = (self.batch_size, image_size[0], image_size[1], image_size[2]), dtype = tf.float32)
            self.train_y_placeholder = tf.placeholder(shape = (self.batch_size, image_size[0], image_size[1], 1), dtype = tf.float32)

            """
            Resize appropriately to fit our network

            """
            train_x_batch = tf.image.resize_images(images = self.train_x_placeholder, size = [512, 512])
            train_y_batch = tf.image.resize_images(images = self.train_y_placeholder, size = [512, 512])
            self.unet(train_x_batch)
            """
            The forward pass

            """
            train_pred = self.unet(train_x_batch)

            """
            Compute the training loss

            """
            dice_loss_ = tf.reduce_mean(self.dice_loss(y_true = train_y_batch, y_pred = train_pred))

            """
            Compute the regularisation

            """
            vars   = tf.trainable_variables()
            regularisation_loss = tf.add_n([ tf.nn.l2_loss(v) for v in vars ])

            """
            Define the training operation

            """
            self.loss = dice_loss_ + 0.01*regularisation_loss
            optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
            self.trainop = optimizer.minimize(self.loss)

            """
            Placeholders and graphs for single frame inference

            """
            self.input_single_placeholder = tf.placeholder(shape = (1, image_size[0], image_size[1], image_size[2]), dtype = tf.float32)
            input_single_resized = tf.image.resize_images(images = self.input_single_placeholder, size = [512, 512])
            self.output_single = self.unet(input_single_resized)

            self.saver = tf.train.Saver()
            self.global_init = tf.global_variables_initializer()
            self.local_init = tf.local_variables_initializer()
        self.sess = tf.Session(graph = self.__graph)

    def initialize(self):
        self.sess.run(self.global_init)
        self.sess.run(self.local_init)
        return 1

    def predict(self, image):
        feed_dict = {self.input_single_placeholder : np.expand_dims(image, 0)}
        outputs = self.sess.run(self.output_single, feed_dict = feed_dict)
        return outputs

    def close(self):
        self.sess.close()
        print("Session Closed")

    def fit(self, train_x, train_y):
        train_y = np.expand_dims(train_y, -1)

        feed_dict = {self.train_x_placeholder : train_x,
                     self.train_y_placeholder : train_y}
        loss_out, _ = self.sess.run([self.loss, self.trainop], feed_dict = feed_dict)
        return loss_out

    def save(self, path = "model/model.ckpt"):
        self.saver.save(self.sess, path)

    def unet(self, inputs):

        """
        Full implimentation of Unet

        """
        conv1 = tf.layers.conv2d(inputs=inputs, filters=16, kernel_size=3, strides=1,
                                 padding="same", activation=tf.nn.relu,
                                 kernel_initializer=tf.glorot_normal_initializer(), name = 'c1', reuse = tf.AUTO_REUSE)
        conv1 = tf.layers.dropout(inputs=conv1, rate=0.1)
        conv1 = tf.layers.conv2d(inputs=conv1, filters=16, kernel_size=3, strides=1,
                                 padding="same", activation=tf.nn.relu,
                                 kernel_initializer=tf.glorot_normal_initializer(), name = 'c2', reuse = tf.AUTO_REUSE)
        p1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=2, strides=2, padding="valid")

        conv2 = tf.layers.conv2d(inputs=p1, filters=32, kernel_size=3, strides=1,
                                 padding="same", activation=tf.nn.relu,
                                 kernel_initializer=tf.glorot_normal_initializer(), name = 'c3', reuse = tf.AUTO_REUSE)
        conv2 = tf.layers.dropout(inputs=conv2, rate=0.1)
        conv2 = tf.layers.conv2d(inputs=conv2, filters=32, kernel_size=3, strides=1,
                                 padding="same", activation=tf.nn.relu,
                                 kernel_initializer=tf.glorot_normal_initializer(), name = 'c4', reuse = tf.AUTO_REUSE)
        p2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=2, strides=2, padding="valid")

        conv3 = tf.layers.conv2d(inputs=p2, filters=64, kernel_size=3, strides=1,
                                 padding="same", activation=tf.nn.relu,
                                 kernel_initializer=tf.glorot_normal_initializer(), name = 'c5', reuse = tf.AUTO_REUSE)
        conv3 = tf.layers.dropout(inputs=conv3, rate=0.1)
        conv3 = tf.layers.conv2d(inputs=conv3, filters=64, kernel_size=3, strides=1,
                                 padding="same", activation=tf.nn.relu,
                                 kernel_initializer=tf.glorot_normal_initializer(), name = 'c6', reuse = tf.AUTO_REUSE)
        p3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=2, strides=2, padding="valid")

        conv4 = tf.layers.conv2d(inputs=p3, filters=128, kernel_size=3, strides=1,
                                 padding="same", activation=tf.nn.relu,
                                 kernel_initializer=tf.glorot_normal_initializer(), name = 'c7', reuse = tf.AUTO_REUSE)
        conv4 = tf.layers.dropout(inputs=conv4, rate=0.1)
        conv4 = tf.layers.conv2d(inputs=conv4, filters=128, kernel_size=3, strides=1,
                                 padding="same", activation=tf.nn.relu,
                                 kernel_initializer=tf.glorot_normal_initializer(), name = 'c8', reuse = tf.AUTO_REUSE)
        p4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=2, strides=2, padding="valid")

        conv5 = tf.layers.conv2d(inputs=p4, filters=256, kernel_size=3, strides=1,
                                 padding="same", activation=tf.nn.relu,
                                 kernel_initializer=tf.glorot_normal_initializer(), name = 'c9', reuse = tf.AUTO_REUSE)
        conv5 = tf.layers.dropout(inputs=conv5, rate=0.2)
        conv5 = tf.layers.conv2d(inputs=conv5, filters=256, kernel_size=3, strides=1,
                                 padding="same", activation=tf.nn.relu,
                                 kernel_initializer=tf.glorot_normal_initializer(), name = 'c10', reuse = tf.AUTO_REUSE)

        up1 = tf.layers.conv2d_transpose(inputs=conv5, filters=128, kernel_size=2, strides=2,
                                         padding="same", kernel_initializer=tf.glorot_normal_initializer())
        up1 = tf.concat([up1, conv4], axis=3)
        conv6 = tf.layers.conv2d(inputs=up1, filters=128, kernel_size=3, strides=1,
                                 padding="same", activation=tf.nn.relu,
                                 kernel_initializer=tf.glorot_normal_initializer(), name = 'c11', reuse = tf.AUTO_REUSE)
        conv6 = tf.layers.dropout(inputs=conv6, rate=0.2)
        conv6 = tf.layers.conv2d(inputs=conv6, filters=128, kernel_size=3, strides=1,
                                 padding="same", activation=tf.nn.relu,
                                 kernel_initializer=tf.glorot_normal_initializer(), name = 'c12', reuse = tf.AUTO_REUSE)

        up2 = tf.layers.conv2d_transpose(inputs=conv6, filters=64, kernel_size=2, strides=2,
                                         padding="same", kernel_initializer=tf.glorot_normal_initializer())
        up2 = tf.concat([up2, conv3], axis=3)
        conv7 = tf.layers.conv2d(inputs=up2, filters=64, kernel_size=3, strides=1,
                                 padding="same", activation=tf.nn.relu,
                                 kernel_initializer=tf.glorot_normal_initializer(), name = 'c13', reuse = tf.AUTO_REUSE)
        conv7 = tf.layers.dropout(inputs=conv7, rate=0.2)
        conv7 = tf.layers.conv2d(inputs=conv7, filters=64, kernel_size=3, strides=1,
                                 padding="same", activation=tf.nn.relu,
                                 kernel_initializer=tf.glorot_normal_initializer(), name = 'c14', reuse = tf.AUTO_REUSE)

        up3 = tf.layers.conv2d_transpose(inputs=conv7, filters=32, kernel_size=2, strides=2,
                                         padding="same", kernel_initializer=tf.glorot_normal_initializer())
        up3 = tf.concat([up3, conv2], axis=3)
        conv8 = tf.layers.conv2d(inputs=up3, filters=32, kernel_size=3, strides=1,
                                 padding="same", activation=tf.nn.relu,
                                 kernel_initializer=tf.glorot_normal_initializer(), name = 'c15', reuse = tf.AUTO_REUSE)
        conv8 = tf.layers.dropout(inputs=conv8, rate=0.2)
        conv8 = tf.layers.conv2d(inputs=conv8, filters=32, kernel_size=3, strides=1,
                                 padding="same", activation=tf.nn.relu,
                                 kernel_initializer=tf.glorot_normal_initializer(), name = 'c16', reuse = tf.AUTO_REUSE)

        up4 = tf.layers.conv2d_transpose(inputs=conv8, filters=16, kernel_size=2, strides=2,
                                         padding="same", kernel_initializer=tf.glorot_normal_initializer())
        up4 = tf.concat([up4, conv1], axis=3)
        conv9 = tf.layers.conv2d(inputs=up4, filters=16, kernel_size=3, strides=1,
                                 padding="same", activation=tf.nn.relu,
                                 kernel_initializer=tf.glorot_normal_initializer(), name = 'c17', reuse = tf.AUTO_REUSE)
        conv9 = tf.layers.dropout(inputs=conv9, rate=0.1)
        conv9 = tf.layers.conv2d(inputs=conv9, filters=16, kernel_size=3, strides=1,
                                 padding="same", activation=tf.nn.relu,
                                 kernel_initializer=tf.glorot_normal_initializer(), name = 'c18', reuse = tf.AUTO_REUSE)

        conv10 = tf.layers.conv2d(inputs=conv9, filters=1, kernel_size=1, strides=1,
                                  kernel_initializer=tf.glorot_normal_initializer(), name = 'c19', reuse = tf.AUTO_REUSE)

        logits = tf.nn.sigmoid(conv10)
        return logits

    def model(self, inputs):

        """
        The network architecture

        Define your own sooper cool network here!

        """
        conv_1 = tf.layers.batch_normalization(tf.layers.max_pooling2d(tf.layers.conv2d(inputs, filters = 8, kernel_size = [7, 7], dilation_rate = (3, 3), activation = tf.nn.relu, name = 'l1c1', reuse = tf.AUTO_REUSE), pool_size = [3, 3], strides = 2))
        conv_2 = tf.layers.batch_normalization(tf.layers.max_pooling2d(tf.layers.conv2d(conv_1, filters = 16, kernel_size = [3, 3], dilation_rate = (3, 3), activation = tf.nn.relu, name = 'l1c2', reuse = tf.AUTO_REUSE), pool_size = [2, 2], strides = 1))
        conv_3 = tf.layers.batch_normalization(tf.layers.max_pooling2d(tf.layers.conv2d(conv_2, filters = 32, kernel_size = [5, 5], dilation_rate = (3, 3), activation = tf.nn.relu, name = 'l1c3', reuse = tf.AUTO_REUSE), pool_size = [2, 2], strides = 2))
        conv_4 = tf.layers.batch_normalization(tf.layers.max_pooling2d(tf.layers.conv2d(conv_3, filters = 64, kernel_size = [3, 3], dilation_rate = (3, 3), activation = tf.nn.relu, name = 'l1c4', reuse = tf.AUTO_REUSE), pool_size = [2, 2], strides = 1))
        conv_5 = tf.layers.batch_normalization(tf.layers.max_pooling2d(tf.layers.conv2d(conv_4, filters = 128, kernel_size = [3, 3], dilation_rate = (3, 3), activation = tf.nn.relu, name = 'l1c5', reuse = tf.AUTO_REUSE), pool_size = [1, 1], strides = 1))

        deconv_1 = tf.layers.batch_normalization(tf.layers.conv2d_transpose(conv_5, filters = 32, kernel_size = [3, 3], strides = (2, 2), activation = tf.nn.relu, name = 'l1d1', reuse = tf.AUTO_REUSE))

        deconv_2 = tf.layers.conv2d_transpose(deconv_1, filters = 8, kernel_size = [3, 3], strides = (2, 2), activation = tf.nn.relu, name = 'l1d2', reuse = tf.AUTO_REUSE)
        # deconv_2_2 = tf.layers.conv2d_transpose(conv_2, filters = 16, kernel_size = [3, 3], strides = (2, 2), activation = tf.nn.relu, name = 'l1d21', reuse = tf.AUTO_REUSE)

        semi1 = tf.image.resize_images(images = deconv_2, size = [512, 512])
        # semi2 = tf.image.resize_images(images = deconv_2_2, size = [300, 300])

        # semi = tf.concat([semi1, semi2], axis = -1)
        out = tf.layers.conv2d(semi1, filters = 1, kernel_size = [3, 3], activation = tf.nn.sigmoid, padding = "SAME", name = 'outputs', reuse = tf.AUTO_REUSE)

        return out


    def dice_loss(self, y_true, y_pred):
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred, [-1])
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        return 1. - (2. * intersection + 1.) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + 1.)

if __name__ == "__main__":
    print("This is not an executable file")
    print("Please run main.py")

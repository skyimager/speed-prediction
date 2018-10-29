import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import math

from keras.models import Sequential, Model, load_model
from keras.layers import Input, Flatten, Lambda, Dense, Conv2D, Conv1D, Cropping2D, MaxPooling2D, BatchNormalization, Concatenate, Dropout, Reshape, Activation
from keras.layers import Conv3D, ZeroPadding3D, MaxPooling3D
from keras.layers import SimpleRNN, TimeDistributed, LSTM
from keras.optimizers import SGD
from keras.backend import tf as ktf
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, Callback
from keras.utils import Sequence
from keras import regularizers, optimizers
import time
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle, resample
from moviepy import *
from moviepy.editor import *
from imgaug import augmenters as iaa



class SampleSequence(Sequence):

    def __init__(self, x, y, batch_size=2, augment=True):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.augment = augment
        sometime = lambda aug: iaa.Sometimes(0.3, aug)
        self.sequence = iaa.Sequential([sometime(iaa.Dropout((0.0, 0.1))),
                                   sometime(iaa.CoarseDropout((0.0, 0.1), size_percent=(0.01, 0.02), per_channel=0.2))],
                                  random_order=True)


    def __len__(self):
        return int(math.ceil(len(self.x) / float(self.batch_size)))

    def read_image(self, f, bright_factor, shadow_flag, shadow_left, shadow_right, flip_flag, rotate_flag, crop_level = None, crop_dir = None):
        im = cv2.imread(f)

        h, w = im.shape[0], im.shape[1]

        if crop_level is not None:
            # im = im[:320, 160-crop_level*10:w-crop_level*10, :]
            if rotate_flag < 0.6:
                angle = 10 if rotate_flag < 0.3 else -10
                M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
                im = cv2.warpAffine(im, M, (w, h))
            im = im[60:300, 170+crop_level*10: 490+crop_level*10, :]
            if crop_dir < 0.5:
                im = cv2.flip(im, 1)
        else:
            im = im[60:300, 160:480, :]

        im = cv2.resize(im, (112, 112))

        if self.augment:
            im = self.sequence.augment_image(im)

        im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        # Brightness
        im = im.astype(np.float64)
        if self.augment:
            im[:, :, 2] = im[:, :, 2] * (0.5 + bright_factor * 0.5)
            im[:, :, 2][im[:, :, 2] > 255] = 255
            # Shadow
            if shadow_flag < 0.25:
                dx, dy, _ = im.shape
                mask = np.full((dx, dy), False)
                x, y = np.mgrid[0:dx, 0:dy]
                x1, y1 = 0, shadow_left * dy
                x2, y2 = dx, shadow_right * dy
                mask[(x - x1) * (y2 - y1) >= (y - y1) * (x2 - x1)] = True
                if shadow_flag < 0.125:
                    mask = ~mask
                im[:, :, 2][mask] *= 0.5
            if flip_flag < 0.25:
                im = cv2.flip(im, 1)
        im = cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_HSV2RGB)
        return im

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx+1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx+1) * self.batch_size]

        xs = []
        crop_dir = np.random.random()
        for x in batch_x:
            xseq = []
            # for x0, x1 in zip(x, x[1:]):
            #     bright_factor = np.random.random()
            #     shadow_flag = np.random.random()
            #     shadow_left = np.random.random()
            #     shadow_right = np.random.random()
            #     im1 = self.read_image(x0, bright_factor, shadow_flag, shadow_left, shadow_right)
            #     im2 = self.read_image(x1, bright_factor, shadow_flag, shadow_left, shadow_right)
            #     # im1 = cv2.resize(im1, (112, 112))
            #     # im2 = cv2.resize(im2, (112, 112))
            #     im = np.concatenate((im1, im2), axis=2)
            #     xseq.append(im)
            rotate_flag = np.random.random()
            rotate_angle = 10 #+ np.random.randint(0, 15)
            for i, x0 in enumerate(x):
                flip_flag = np.random.random()
                bright_factor = np.random.random()
                shadow_flag = np.random.random()
                shadow_left = np.random.random()
                shadow_right = np.random.random()
                crop_level = None
                if x0.startswith(':flip:'):
                    x0 = x0.partition(":flip:")[2]
                    flip_flag = 0.0
                if x0.startswith(':noflip:'):
                    x0 = x0.partition(":noflip:")[2]
                    flip_flag = 1.0
                if x0.startswith(':crop:'):
                    x0 = x0.partition(":crop:")[2]
                    crop_level = len(x)-(i+1)
                im = self.read_image(x0, bright_factor, shadow_flag, shadow_left, shadow_right, flip_flag, rotate_flag, crop_level, crop_dir)
                im = im - [104.00699, 116.66877, 122.67892]
                # im = (im - 127)/255.0
                xseq.append(im)

            xs.append(xseq)
        xs = np.array(xs)
        if self.y is not None and len(self.y) > 0:
            # ys = []
            # for y in batch_y:
            #     ys.append(y[:-1].reshape((-1,1)))
            # ys = np.array(ys)
            # ys = np.mean(batch_y, axis=1)
            ys = batch_y
            return xs, ys
        else:
            return xs

    def on_epoch_end(self):
        # TODO Is this invoked for prediction?
        self.x, self.y = shuffle(self.x, self.y)


class Dataset():

    @classmethod
    def load(cls, kind, filetype='png', n_sequences=5, strided_sequences=True):
        nfeats = len(glob.glob("data/%s/*.%s" % (kind, filetype)))
        files = ["data/%s/%s.%s" % (kind, i+1, filetype) for i in range(0, nfeats)]
        patch_len = (int(math.ceil(float(len(files)) / n_sequences)) * n_sequences - len(files))

        if strided_sequences:
            features = np.array([files[i: i+n_sequences] for i in range(len(files)-(n_sequences-1))])
        else:
            features = [files[i: i+n_sequences] for i in range(0, len(files), n_sequences)]
            for _ in range(patch_len):
                features[-1].append(files[-1])
            features = np.array(features)

        if glob.glob("data/%s.txt" % kind):
            ys = np.array(np.loadtxt("data/%s.txt" % kind))
            if strided_sequences:
                labels = np.array([ys[i: i+n_sequences] for i in range(len(ys)-(n_sequences-1))])
            else:
                labels = [ys[i: i+n_sequences] for i in range(0, len(ys), n_sequences)]
                for i in range(patch_len):
                    labels[-1] = np.append(labels[-1], 0)
                labels = np.array(labels)
        else:
            labels = np.array([])

        return cls(features, labels)

    def __init__(self, f, l):
        self.features = f
        self.labels = l

    def distribution_index(self, nbins, pass_thru_low=False):
        _, bins = np.histogram(self.labels[:,0], bins=nbins)
        if pass_thru_low:
            final_bins = np.append([0, 1.5], bins[1:-1])
        else:
            final_bins = bins[:-1]
        bin_index = np.digitize(self.labels[:,0], final_bins)
        return final_bins, bin_index

    def stats(self, nbins=10, pass_thru_low=False):
        bins, bin_index = self.distribution_index(nbins, pass_thru_low=pass_thru_low)
        counts = np.bincount(bin_index)
        return bins, counts[1:]

    def resample(self, nsamples=10, nbins=10, pass_thru_low=False):
        bins, bin_index = self.distribution_index(nbins, pass_thru_low=pass_thru_low)
        partitioned = [np.argwhere(bin_index == i + 1).reshape(-1) for i in range(len(bins))]
        resampled = [s for p in partitioned for s in resample(p, n_samples=min(nsamples, len(p)), replace=False)]
        return Dataset(self.features[resampled], self.labels[resampled])

    def stationary_sequences(self, naug):
        nfeat, nseq = len(self.features), len(self.features[0])
        features = [np.repeat(self.features[np.random.randint(0, nfeat), 0], nseq) for _ in range(naug)]
        labels = np.zeros((naug, nseq))
        return Dataset(np.array(features), np.array(labels))

    def scaled_sequences(self, mn, mx, scale=2):
        nseq = len(self.features[0])
        cond = np.logical_and(self.labels[:,0]>=mn, self.labels[:,0]<mx)
        feats = []
        labs = []
        for i in np.argwhere(cond).reshape((-1)):
            sel = np.arange(i, i+scale*nseq, scale)
            f = self.features[sel, 0]
            l = np.sum([self.labels[sel+off, 0] for off in range(scale)], 0)
            feats.append(f)
            labs.append(l)
        return Dataset(np.array(feats), np.array(labs))

    def slowed_sequences(self, from_idx, to_idx, n, scale=2):
        fs, ls = self.features[from_idx:to_idx], self.labels[from_idx:to_idx]
        fs, ls = resample(fs, ls, n_samples=n, replace=False)
        nonflipped = np.vectorize(lambda f: ":noflip:" + f)(fs)
        flipped = np.vectorize(lambda f: ":flip:" + f)(fs)
        # TODO Works only for scale=2
        fs = np.dstack((nonflipped, flipped)).reshape(n * scale, -1)
        ls = np.repeat(ls, scale, axis=1).reshape((n*scale, -1))/scale
        return Dataset(fs, ls)

    def turn_sequences(self, mn, mx, n):
        cond = np.logical_and(self.labels[:,0]>=mn, self.labels[:,0]<mx)
        sel = resample(np.argwhere(cond), n_samples=n, replace=False).flatten()
        print(sel.shape)
        # sel = np.random.random_integers(0, len(self.features)-1, n)
        prefix = np.vectorize(lambda f: ":crop:" + f)
        fs = np.vectorize(lambda f: prefix(f))(self.features[sel])
        ls = self.labels[sel]
        return Dataset(fs, ls)

    def without_sequences(self, from_idx, to_idx):
        fs = np.append(self.features[0:from_idx], self.features[to_idx:None], axis=0)
        ls = np.append(self.labels[0:from_idx], self.labels[to_idx:None], axis=0)
        return Dataset(fs, ls)

    def append(self, d):
        features = np.append(self.features, d.features, axis=0)
        labels = np.append(self.labels, d.labels, axis=0)
        return Dataset(features, labels)

    def clip(self, frm, to):
        return Dataset(self.features[frm:to], self.labels[frm:to])

    def video(self, path, label_path, fps=20):
        def make_frame(t):
            i = int(t*20)
            im = cv2.imread(self.features[i])
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            if i < len(self.labels):
                cv2.putText(im, "%06d - %.2f" % (i, self.labels[i]), (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)
            return im

        self.features = self.features.reshape((-1,))
        self.labels = np.loadtxt(label_path)
        v = VideoClip(make_frame, duration=(len(self.features)/fps))
        v.write_videofile(path, fps=fps)

    def video_generator(self, path, fps=20):
        gen = self.generator(augment=False)

        def make_frame(t):
            i = int(t*20)
            f, l = gen[i]
            im = cv2.imread(self.features[i])
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            if i < len(self.labels):
                cv2.putText(im, "%06d - %.2f" % (i, self.labels[i]), (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)
            return im

        # v = VideoClip(make_frame, duration=(len(self.features)/fps))
        # v.write_videofile(path, fps=fps)

    def generator(self, batch_size=4, augment=True):
        return SampleSequence(self.features, self.labels, batch_size, augment)

    def train_test_split(self, test_ratio=0.1, nbins=10):
        _, bin_index = self.distribution_index(nbins)
        X, Xt, y, yt = train_test_split(self.features, self.labels, test_size=test_ratio, shuffle=True, stratify=bin_index)
        return Dataset(X, y), Dataset(Xt, yt)


class Prediction():

    @classmethod
    def load(cls, dataset, f):
        return cls(dataset, np.loadtxt(f))

    def __init__(self, dataset, pred):
        if dataset.labels.size > 0:
            y = np.concatenate([np.array(i) for i in dataset.labels])
            self.labels = y.reshape((-1))
        else:
            self.labels = np.array([])
        self.predictions = pred.reshape((-1))

    def err(self, st=0, end=None):
        return np.mean(np.square(self.labels[st:end] - self.predictions[st:end]))

    def plot(self, st=0, end=None):
        pred = self.predictions[st:end]
        plt.plot(pred, "r")
        if self.labels.size >  0:
            lab = self.labels[st:end]
            plt.plot(lab, "b")
            plt.show()
            plt.axis('equal')
            plt.plot(lab, pred, ".")
            plt.show()
            err = pred - lab
            over = err > 0
            print(np.mean(err[over]), np.mean(err[~over]))
            plt.plot(lab[over], err[over], "g.")
            plt.plot(lab[~over], err[~over], "r.")

        plt.show()

    def save(self, f):
        np.savetxt(f, self.predictions)


class Network():

    def __init__(self):
        # self.model, self.layers = self.init_FlowNet_model()
        # self.model, self.layers = self.init_DeepVO_model()
        # self.model, self.layers = self.init_RNN_model()
        self.model, self.layers = self.init_C3D_model()
        #self.tensorboard = TensorBoard(log_dir='./logs/{}'.format(time.time()), batch_size=32) #write_images=True, histogram_freq=1)
        # self.checkpoint = ModelCheckpoint('DeepVO-corr-final.hdf5', 'val_loss', save_best_only=True)
        self.checkpoint = ModelCheckpoint('final-stage11.{epoch:02d}-{val_loss:.2f}.hdf5', 'val_loss', save_best_only=True)
        self.stopping = EarlyStopping(patience=3)

    def init_DeepVO_model(self):
        alexnet = self.get_alexnet()
        alexnet_model = self.get_DeepVO_model(alexnet, alexnet)
        layers = alexnet_model.layers + alexnet.layers
        return alexnet_model, dict([(l.name, l) for l in layers])

    def init_FlowNet_model(self):
        model = self.get_FlowNet_model()
        return model, dict([(l.name, l) for l in model.layers])

    def init_RNN_model(self):
        model = self.get_RNN_model()
        return model, dict([(l.name, l) for l in model.layers])

    def init_C3D_model(self):
        model = self.get_C3D_model()
        return model, dict([(l.name, l) for l in model.layers])

    @classmethod
    def load(self, path):
        network = Network()
        network.model.load_weights(path)
        return network

    def train(self, dataset, batch_size=1, epochs=20, callbacks = []):
        print('training now')
        trainset, valset = dataset.train_test_split()
        trains, vals = trainset.generator(batch_size), valset.generator(batch_size, augment=False)
        training_steps, validation_steps = len(trains), len(vals)
        print("Training for ", training_steps, ", validating for ", validation_steps)

        self.model.fit_generator(
            generator=trains, epochs=epochs, steps_per_epoch=training_steps, use_multiprocessing=False,
            validation_data=vals, validation_steps=validation_steps,
            callbacks=callbacks)

    def predict(self, dataset):
        pred = self.model.predict_generator(dataset.generator(augment=False))
        return Prediction(dataset, pred)

    def layer_output(self, input):
        input_layer = 'input_1'
        conv_layers = ['conv2d_%d' % i for i in range(1, 6)]
        regression_layers = ['dense_1', 'dense_2']
        conv = []
        reg = []
        for conv_layer in conv_layers:
            functor = K.function([self.layers[input_layer].input], [self.layers[conv_layer].output])
            output = functor([input])
            im = output[0][0]
            im = np.moveaxis(im, -1, 0)
            n, h, w = im.shape
            im = im.reshape(-1, 8, h, w)
            im = np.array([np.hstack(i) for i in im])
            im = np.vstack(im)
            conv.append(im)
        for regression_layer in regression_layers:
            functor = K.function([self.layers[input_layer].input], [self.layers[regression_layer].output])
            output = functor([input])
            reg.append(output[0][0])
        return np.array(conv), np.array(reg)

    def layer_video(self, dataset, path, nframes=100):
        def make_frame(t):
            i = int(t*20)
            # im = cv2.imread(dataset.features[i][0])
            # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = mpimg.imread(dataset.features[i][0])
            im = np.array([im])
            image_out, func_out = self.layer_output(im)
            ret = image_out[0]
            ret = ret*255.0/np.max(ret)
            ret = cv2.cvtColor(ret, cv2.COLOR_GRAY2RGB)
            return ret
        v = VideoClip(make_frame, duration=nframes/20.0)
        v.write_videofile(path, fps=20)

    def get_alexnet(self):
        image_input = Input(shape=(480, 640, 3), name='input_1')

        x = image_input

        # x = Cropping2D(cropping=((200,130), (50,50)))(x)
        # x = Cropping2D(cropping=((150,125), (0,0)))(x)
        x = Lambda(lambda image: ktf.image.resize_images(image, (227, 227)))(x)
        x = Lambda(lambda image: (image  - 127.)/255.)(x)

        x = Conv2D(48, (11, 11), strides=4, padding='same', activation='relu', name='conv2d_1')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((3, 3), strides=2, padding='same')(x)

        x = Conv2D(128, (5, 5), strides=1, padding='same', activation='relu', name='conv2d_2')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((3, 3), strides=2, padding='same')(x)

        x = Conv2D(192, (3, 3), strides=1, padding='same', activation='relu', name='conv2d_3')(x)
        x = Conv2D(192, (3, 3), strides=1, padding='same', activation='relu', name='conv2d_4')(x)
        x = Conv2D(128, (3, 3), strides=1, padding='same', activation='relu', name='conv2d_5')(x)
        x = MaxPooling2D((3, 3), strides=2, padding='same')(x)

        x = Flatten()(x)
        x = Dense(2048, activation='relu', name='dense_1')(x)
        x = Dropout(0.5)(x)
        x = Dense(2048, activation='relu', name='dense_2')(x)
        x = Dropout(0.5)(x)
        output = x

        model = Model(image_input, output)
        return model


    def get_DeepVO_model(self, curr_state, next_state):
        input = Input(shape=(1, 480, 640, 6), name='input')
        x = Reshape((480, 640, 6))(input)

        # curr_image = Input(shape=(480, 640, 3))
        # next_image = Input(shape=(480, 640, 3))
        curr_image = Lambda(lambda x: x[:,:,:,0:3], output_shape=(480, 640, 3))(x)
        next_image = Lambda(lambda x: x[:,:,:,3:6], output_shape=(480, 640, 3))(x)

        curr_image_feats = curr_state(curr_image)
        next_image_feats = next_state(next_image)

        # x = Concatenate()([curr_image_feats, next_image_feats])

        x1 = Lambda(lambda x: K.expand_dims(x))(curr_image_feats)
        x2 = Lambda(lambda x: K.expand_dims(x))(next_image_feats)
        x = Concatenate()([x1, x2])
        x = Dense(3, name='fc_comb', activation='relu')(x)
        x = Flatten()(x)

        x = Dense(8192, name='fc1', activation='relu')(x)
        x = Dense(1024, name='fc2', activation='relu')(x)
        out = Dense(1, name='fc3')(x)

        out = Reshape((1, 1))(out)

        model = Model(input, out)
        model.compile(loss='mse', optimizer='adam')

        return model


    def get_TimeDistributed_DeepVO_model(self, curr_state, next_state):
        input = Input(shape=(None, 480, 640, 6), name='input')
        x = input
        # x = Reshape((480, 640, 6))(input)

        # curr_image = Input(shape=(480, 640, 3))
        # next_image = Input(shape=(480, 640, 3))
        curr_image = TimeDistributed(Lambda(lambda x: x[:,:,:,0:3], output_shape=(480, 640, 3)))(x)
        next_image = TimeDistributed(Lambda(lambda x: x[:,:,:,3:6], output_shape=(480, 640, 3)))(x)

        curr_image_feats = TimeDistributed(curr_state)(curr_image)
        next_image_feats = TimeDistributed(next_state)(next_image)

        # x = Concatenate()([curr_image_feats, next_image_feats])
        x1 = TimeDistributed(Lambda(lambda x: K.expand_dims(x)))(curr_image_feats)
        x2 = TimeDistributed(Lambda(lambda x: K.expand_dims(x)))(next_image_feats)
        x = Concatenate()([x1, x2])
        x = TimeDistributed(Dense(3, name='fc_comb', activation='relu'))(x)
        x = TimeDistributed(Flatten())(x)

        # x = TimeDistributed(Dense(8192, name='fc1', activation='relu'))(x)
        # x = TimeDistributed(Dense(1024, name='fc2', activation='relu'))(x)
        x = SimpleRNN(2000, return_sequences=True, activation='relu', name='rnn1')(x)
        x = SimpleRNN(1000, return_sequences=True, activation='relu', name='rnn2')(x)

        out = TimeDistributed(Dense(1, name='fc3'))(x)

        # out = Reshape((1, 1))(out)

        model = Model(input, out)
        model.compile(loss='mse', optimizer='adam')

        return model

    def get_FlowNet_model(self):
        # To accommodate time-series dataset
        input = Input(shape=(1, 480, 640, 6), name='input')
        x = Reshape((480, 640, 6))(input)

        # curr_resized = Lambda(lambda image: ktf.image.resize_images(image, (120, 160)))(curr_image)
        # next_resized = Lambda(lambda image: ktf.image.resize_images(image, (120, 160)))(next_image)

        x = Lambda(lambda image: (image - 127.)/255.)(x)

        ### Reduced FlowNet
        x = Conv2D(32, (7, 7), strides=2, padding='same', activation='relu', name='conv1')(x)
        x = BatchNormalization()(x)
        x = Conv2D(64, (5, 5), strides=2, padding='same', activation='relu', name='conv2')(x)
        x = BatchNormalization()(x)
        x = Conv2D(128, (5, 5), strides=2, padding='same', activation='relu', name='conv3')(x)
        x = Conv2D(128, (3, 3), strides=1, padding='same', activation='relu', name='conv3_1')(x)
        x = BatchNormalization()(x)
        x = Conv2D(256, (3, 3), strides=2, padding='same', activation='relu', name='conv4')(x)
        x = Conv2D(256, (3, 3), strides=1, padding='same', activation='relu', name='conv4_1')(x)
        x = BatchNormalization()(x)
        x = Conv2D(256, (3, 3), strides=2, padding='same', activation='relu', name='conv5')(x)
        x = Conv2D(256, (3, 3), strides=1, padding='same', activation='relu', name='conv5_1')(x)
        x = BatchNormalization()(x)
        x = Conv2D(512, (3, 3), strides=2, padding='same', activation='relu', name='conv6')(x)

        x = Flatten()(x)
        x = Dense(2048, name='fc1', activation='relu')(x)
        x = Dense(1024, name='fc2', activation='relu')(x)
        out = Dense(1, name='fc3')(x)

        # To accommodate time-series dataset
        out = Reshape((1, 1))(out)


        model = Model(input, out)
        model.compile(loss='mse', optimizer='adam')

        return model

    def get_simple_RNN_model(self, n_sequences=10):

        cnn = Sequential()
        cnn.add(TimeDistributed(Lambda(lambda image: (image - 127.) / 255.), input_shape=(None, 480, 640, 6)))
        cnn.add(TimeDistributed(Conv2D(32, (7, 7), strides=2, padding='same', activation='relu', name='conv1')))
        cnn.add(BatchNormalization())
        cnn.add(TimeDistributed(Conv2D(64, (5, 5), strides=2, padding='same', activation='relu', name='conv2')))
        cnn.add(BatchNormalization())
        cnn.add(TimeDistributed(Conv2D(128, (5, 5), strides=2, padding='same', activation='relu', name='conv3')))
        cnn.add(TimeDistributed(Conv2D(128, (3, 3), strides=1, padding='same', activation='relu', name='conv3_1')))
        cnn.add(BatchNormalization())
        cnn.add(TimeDistributed(Conv2D(256, (3, 3), strides=2, padding='same', activation='relu', name='conv4')))
        cnn.add(TimeDistributed(Conv2D(256, (3, 3), strides=1, padding='same', activation='relu', name='conv4_1')))
        cnn.add(BatchNormalization())
        cnn.add(TimeDistributed(Conv2D(256, (3, 3), strides=2, padding='same', activation='relu', name='conv5')))
        cnn.add(TimeDistributed(Conv2D(256, (3, 3), strides=1, padding='same', activation='relu', name='conv5_1')))
        cnn.add(BatchNormalization())
        cnn.add(TimeDistributed(Conv2D(512, (3, 3), strides=2, padding='same', activation='relu', name='conv6')))
        cnn.add(TimeDistributed(Flatten()))

        cnn.add(SimpleRNN(1000, return_sequences=True, activation='relu'))
        cnn.add(SimpleRNN(1000, return_sequences=True, activation='relu'))

        # cnn.add(TimeDistributed(Dense(1024, name='fc1', activation='relu')))
        # cnn.add(TimeDistributed(Dense(512, name='fc2', activation='relu')))
        cnn.add(TimeDistributed(Dense(1, name="output")))

        cnn.compile(loss='mse', optimizer='adam')
        return cnn


    def get_RNN_model(self):

        cnn = Sequential()
        cnn.add(TimeDistributed(Lambda(lambda image: (image - 127.)/255.), input_shape=(None, 480, 640, 6)))

        cnn.add(TimeDistributed(Conv2D(32, (7, 7), strides=2, padding='same', activation=None), name='conv1'))
        cnn.add(TimeDistributed(BatchNormalization(), name='bn1'))
        cnn.add(TimeDistributed(Activation('relu')))
        # cnn.add(Dropout(0.5))

        cnn.add(TimeDistributed(Conv2D(64, (5, 5), strides=2, padding='same', activation=None), name='conv2'))
        cnn.add(TimeDistributed(BatchNormalization(), name='bn2'))
        cnn.add(TimeDistributed(Activation('relu')))
        # cnn.add(Dropout(0.5))

        cnn.add(TimeDistributed(Conv2D(128, (5, 5), strides=2, padding='same', activation=None), name='conv3'))
        cnn.add(TimeDistributed(BatchNormalization(), name='bn3'))
        cnn.add(TimeDistributed(Activation('relu')))
        cnn.add(TimeDistributed(Conv2D(128, (3, 3), strides=1, padding='same', activation=None), name='conv3_1'))
        cnn.add(TimeDistributed(BatchNormalization(), name='bn3_1'))
        cnn.add(TimeDistributed(Activation('relu')))
        # cnn.add(Dropout(0.5))

        cnn.add(TimeDistributed(Conv2D(256, (3, 3), strides=2, padding='same', activation=None), name='conv4'))
        cnn.add(TimeDistributed(BatchNormalization(), name='bn4'))
        cnn.add(TimeDistributed(Activation('relu')))
        cnn.add(TimeDistributed(Conv2D(256, (3, 3), strides=1, padding='same', activation=None), name='conv4_1'))
        cnn.add(TimeDistributed(BatchNormalization(), name='bn4_1'))
        cnn.add(TimeDistributed(Activation('relu')))
        # cnn.add(Dropout(0.5))

        cnn.add(TimeDistributed(Conv2D(256, (3, 3), strides=2, padding='same', activation=None), name='conv5'))
        cnn.add(TimeDistributed(BatchNormalization(), name='bn5'))
        cnn.add(TimeDistributed(Activation('relu')))
        cnn.add(TimeDistributed(Conv2D(256, (3, 3), strides=1, padding='same', activation=None), name='conv5_1'))
        cnn.add(TimeDistributed(BatchNormalization(), name='bn5_1'))
        cnn.add(TimeDistributed(Activation('relu')))
        # cnn.add(Dropout(0.5))

        cnn.add(TimeDistributed(Conv2D(512, (3, 3), strides=2, padding='same', activation=None), name='conv6'))
        cnn.add(TimeDistributed(BatchNormalization(), name='bn6'))
        cnn.add(TimeDistributed(Activation('relu')))
        # cnn.add(Dropout(0.5))

        cnn.add(TimeDistributed(Flatten()))
        cnn.add(SimpleRNN(1000, return_sequences=True, activation='relu', use_bias=False, name='rnn1'))
        cnn.add(SimpleRNN(1000, return_sequences=True, activation='relu', use_bias=False, name='rnn2'))
        # cnn.add(Dropout(0.5))
        cnn.add(TimeDistributed(Dense(1, use_bias=False), name="output"))

        # adam = optimizers.Adam(lr=0.1)
        cnn.compile(loss='mse', optimizer='adam')
        return cnn

    def get_C3D_model(self):
        c3d = Sequential()
        # c3d.add(Lambda(lambda image: (image - 127.) / 255., input_shape=(16, 112, 112, 3)))

        c3d.add(Conv3D(64, (3, 3, 3), activation='relu', padding='same', name='conv1', input_shape=(16, 112, 112, 3)))
        c3d.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), padding='valid', name='pool1'))

        c3d.add(Conv3D(128, (3, 3, 3), activation='relu', padding='same', name='conv2'))
        c3d.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool2'))

        c3d.add(Conv3D(256, (3, 3, 3), activation='relu', padding='same', name='conv3a'))
        c3d.add(Conv3D(256, (3, 3, 3), activation='relu', padding='same', name='conv3b'))
        c3d.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool3'))

        c3d.add(Conv3D(512, (3, 3, 3), activation='relu', padding='same', name='conv4a'))
        c3d.add(Conv3D(512, (3, 3, 3), activation='relu', padding='same', name='conv4b'))
        c3d.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool4'))

        c3d.add(Conv3D(512, (3, 3, 3), activation='relu', padding='same', name='conv5a'))
        c3d.add(Conv3D(512, (3, 3, 3), activation='relu', padding='same', name='conv5b'))
        c3d.add(ZeroPadding3D(padding=((0, 0), (0, 1), (0, 1)), name='zeropad'))
        c3d.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool5'))
        c3d.add(Flatten())

        c3d.add(Dense(4096, activation='relu', name='fc6'))
        c3d.add(Dropout(0.5))
        c3d.add(Dense(4096, activation='relu', name='fc7'))

        c3d.add(Dropout(0.5))
        c3d.add(Dense(16, name='out'))

        sgd = SGD(lr=1e-5, decay=0.0005, momentum=0.9)

        def custom_loss(y_true, y_pred):
            loss = ktf.squared_difference(y_true, y_pred)
            loss = ktf.reduce_mean(loss)

            return loss
        c3d.compile(loss=custom_loss, optimizer=sgd)

        return c3d



def export_video(mode='train'):
    cap = cv2.VideoCapture("data/%(mode)s.mp4" % locals())
    img_id = 0
    length = int(cap.get(cv2.cv.CV_CAP_PROP_FPS))
    print(length)
    ret = cap.isOpened()
    while(ret):
        ret, frame = cap.read()
        img_id += 1
        filepath = "data/%(mode)s/%(img_id)s.png" % locals()
        cv2.imwrite(filepath, frame)
        print("Wrote ", filepath)
    cap.release()




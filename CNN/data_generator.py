"""
DataGenerator supplies data to fit method of keras models.

Copyright (c) 2023 Global Health Labs, Inc
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
"""
import tensorflow.keras as keras
import numpy as np
import cv2
import random

class DataGenerator(keras.utils.Sequence):
    """
    Generates data for Keras
    """

    def __init__(self,
                 files,
                 labels,
                 output_shape,
                 augmentation = False,
                 batch_size=64,
                 n_classes=2,
                 shuffle=False,
                 balanced=True,
                 mask=None):
        """
        Constructor.
        """
        self.files = files
        self.labels = labels
        self.shuffle = shuffle
        self.mask = mask
        self.augmentation = augmentation
        self.output_shape = output_shape
        self.n_channels = output_shape[2]
        self.batch_size = batch_size
        self.n_classes = n_classes
        # convert mean, std to float32 between [0, 1]
        self.indices = None
        self.positive_indices = [ind for ind, el in enumerate(self.labels) if el]
        self.negative_indices = [ind for ind, el in enumerate(self.labels) if not el]
        if balanced:
            # since we are upsampling the class with fewer examples, n_images will be twice the number of images in the
            # dominant class
            self.n_images = max(len(self.positive_indices), len(self.negative_indices)) * 2
            # initialize
            self.on_epoch_end()
        else:
            self.n_images = len(self.positive_indices)+len(self.negative_indices)
            self.indices = np.array([i for i in range(self.n_images)])

    def __len__(self):
        """
        Denotes the number of batches per epoch.
        :return: number of batches per epoch
        """
        return int(np.floor(self.n_images / self.batch_size))

    def __getitem__(self, index):
        """
        Generates one batch of data.
        X : (batch_size, n_rows, n_cols, n_channels)
        :param index: index of batch
        :return: X, y images and labels
        """
        # generate indices of the batch
        batch_indices = self.indices[index * self.batch_size:min((index + 1) * self.batch_size, self.n_images)]

        # get batch of files and labels
        batch_files = [self.files[k] for k in batch_indices]
        batch_labels = [self.labels[k] for k in batch_indices]

        # generate data
        X, y = self.generate_data(batch_indices, batch_files, batch_labels)

        # truncate (for inference use case in last partial batch)
        if self.mask is None:
            if X.shape[0] > batch_indices.shape[0]:
                X = X[:batch_indices.shape[0], ]
                y = y[:batch_indices.shape[0]]
        else:
            if X[0].shape[0] > batch_indices.shape[0]:
                for ind in range(len(X)):
                    X[ind] = X[ind][:batch_indices.shape[0], ]
                y = y[:batch_indices.shape[0]]

        # return to caller
        return X, y

    def get_len(self):
        n_batches = int(np.floor(self.n_images / self.batch_size))
        if self.n_images == n_batches * self.batch_size:
            return n_batches
        else:
            return n_batches + 1

    def generate_data(self, batch_indices, batch_files, batch_labels):
        """
        Generates data containing batch_size samples
        :param batch_indices: indices for batch (not used here)
        :param batch_files: filenames for batch
        :param batch_labels: labels for  batch
        :return: X, z
        """
        # initialization
        X = np.empty((self.batch_size, *self.output_shape), dtype=np.float32)
        y = np.empty((self.batch_size,), dtype=np.uint8)

        # generate data
        pairs = zip(batch_files, batch_labels)

        # read each image in batch
        for i, pair in enumerate(pairs):
            img = cv2.imread(pair[0], cv2.IMREAD_UNCHANGED)
            # TODO: explain the magic number 2
            if len(img.shape) > 2:
                img = img[:, :, 0]
            img = cv2.resize(img, (self.output_shape[1], self.output_shape[0]))
            X[i,] = self.augment_data(img,self.augmentation)
            y[i] = pair[1]

        # convert y to one-hot form
        z = keras.utils.to_categorical(y, num_classes=self.n_classes).astype(np.uint8)

        if self.mask is not None:
            return [X, self.mask], z

        return X, z

    def augment_data(self, image,augmentation):
        # pick a random augmentation
        if augmentation:
            aug = [random.choice([True, False]), random.choice([True, False]), random.choice([True, False]),
                random.choice([True, False]), random.choice([True, False])]
            aug_types = [0, 1, 2, 3, 4]
            random.shuffle(aug_types)
            for ind, aug_type in enumerate(aug_types):
                if aug[ind]:
                    if aug_type == 0:
                        image = np.fliplr(image)  # flip left right
                    elif aug_type == 1:
                        gamma = random.choice([0.7, 0.8, 0.9, 1.1, 1.2, 1.3])
                        t = np.power(np.linspace(0, 1, 256), gamma)
                        # put through gamma lookup table, output is float32 between [0, 1]
                        image = cv2.LUT(image, t)
                    elif aug_type == 2:
                        kernel_size = random.choice([2, 3, 4, 5])
                        image = cv2.blur(image, (kernel_size, kernel_size))
                    elif aug_type == 3:
                        delta = 10
                        pct_pix = 5
                        sz = image.shape
                        pix = np.random.randint(0, np.product(sz[:2]), [np.int(np.product(sz[:2]) * pct_pix * 0.01), ])
                        image = image.flatten()
                        image[pix] = image[pix] + delta
                        image = image.reshape(sz)
                        image[image < 0] = 0
                        image[image > 255] = 255
                    else:
                        image = image
                else:
                    pass
        image = (image - np.mean(image)) / np.std(image)
        image = np.expand_dims(image, axis=2)

        return image

    def on_epoch_end(self):
        """
        Updates indices after each epoch.
        """
        # to create balance, deal separately with positive and negative samples
        # if there are fewer of one class, append a random sampling of indices
        # then interweave positive and negative indices (after shuffling each independently if requested)
        # so batches will be balanced
        # deal with positive indices
        positive_indices = self.positive_indices.copy()
        negative_indices = self.negative_indices.copy()
        if len(positive_indices) > len(negative_indices):
            negative_indices.extend(np.random.choice(negative_indices, (len(positive_indices) - len(negative_indices),)).tolist())
        elif len(negative_indices) > len(positive_indices):
            positive_indices.extend(np.random.choice(positive_indices, (len(negative_indices) - len(positive_indices),)).tolist())

        if self.shuffle:
            np.random.shuffle(positive_indices)
            np.random.shuffle(negative_indices)

        indicesList = []
        for ind in range(len(positive_indices)):
            indicesList.extend([positive_indices[ind], negative_indices[ind]])
        self.indices = np.array(indicesList)
        return


class DataGeneratorMemory(DataGenerator):
    """
    Same as DataGenerator, but saves all images to memory on instantiation
    """

    def __init__(self,
                 files,
                 labels,
                 output_shape,
                 augmentation = False,
                 batch_size=64,
                 n_classes=2,
                 shuffle=False,
                 balanced=True,
                 mask=None):
        # call super constructor to initiate attributes
        super().__init__(
            files,
            labels,
            output_shape,
            augmentation = augmentation,
            batch_size=batch_size,
            n_classes=n_classes,
            shuffle=shuffle,
            balanced=balanced,
            mask=mask)

        # read all images into memory
        self.images = list()
        for i in range(len(self.files)):
            img = cv2.imread(self.files[i], cv2.IMREAD_UNCHANGED)
            if len(img.shape) > 2:
                img = img[:, :, 0]
            img = cv2.resize(img, (self.output_shape[1], self.output_shape[0]))
            self.images.append(img)

    def generate_data(self, batch_indices, batch_files, batch_labels):
        """
        Generates data containing batch_size samples
        :param batch_indices: indices of batch
        :param batch_files: files of batch (not used here)
        :param batch_labels: labels of batch
        :return: X, z
        """
        # initialization
        X = np.empty((self.batch_size, *self.output_shape), dtype=np.float32)
        y = np.empty((self.batch_size,), dtype=np.uint8)

        # generate data
        pairs = zip(batch_indices, batch_labels)

        # read each image in batch
        for i, pair in enumerate(pairs):
            img = self.images[pair[0]]
            X[i,] = self.augment_data(img,self.augmentation)
            y[i] = pair[1]

        # convert y to one-hot form
        z = keras.utils.to_categorical(y, num_classes=self.n_classes).astype(np.uint8)

        if self.mask is not None:
            return [X, self.mask], z

        return X, z

"""
Train a binary classifier.
Automatically optimizes hyper parameters with customized grid search
Architecture Type is VGG style.
Best with Python3 and TensorFlow 1.15
Run this command from the base directory :
$python3 --train_pos_dir <path_to_positive_train_data> --train_neg_dir <path_to_negative_train_data> --val_pos_dir <path_to_positive_validation_data> --val_neg_dir <path_to_negative_validation_data>

Copyright (c) 2023 Global Health Labs, Inc
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
"""
import os
import time
import sys
import glob
import random
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import argparse
from data_generator import DataGeneratorMemory
from tensorflow.keras.preprocessing.image import ImageDataGenerator

parser = argparse.ArgumentParser()
parser.add_argument('--cuda_number',type=str, default='0', help='0 |1| 0,1')
parser.add_argument('--channel_multiplier',type=int, default=3 )
parser.add_argument('--lr', type=float, default=0.0005)
parser.add_argument('--drop_out', type=float, default=0.25)
parser.add_argument('--bz', type=int, default=8, help='batch size')
parser.add_argument('--img_size', type=int, default=256)
parser.add_argument('--epoch',type=int, default=100, help='maximum epoch')
parser.add_argument('--trial', type=int, default=12)
parser.add_argument('--train_pos_dir', type=str, required=True)
parser.add_argument('--train_neg_dir', type=str, required=True)
parser.add_argument('--val_pos_dir', type=str, required=True)
parser.add_argument('--val_neg_dir', type=str, required=True)
parser.add_argument('--save_dir', type=str, default="logs/")

args = parser.parse_args()

def parameter_unique_dirname(params):
    """
    Function to return unique dir name for the current hyperparameter setup.
    :param params: dictionary of the parameters that were chosen to join the hyper parameter serach
    :return: string of file name
    """
    dir_name = ''
    hy_paras = params.keys()
    hy_values = params.values()

    for param, value in zip(hy_paras, hy_values):
        dir_name = dir_name + str(param) + str(value)
    return dir_name

# Specify which GPU number to use.
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_number

# laod the positive and negative class in the training data set
training_images_1 = glob.glob(args.train_pos_dir + '/*.jpg') # positive images
training_images_2 = glob.glob(args.train_neg_dir + '/*.jpg') # negative images

training_images = training_images_1+training_images_2

# laod the positive and negative class in the validation data set
validation_images_1 = glob.glob(args.val_pos_dir + '/*.jpg')+glob.glob(args.val_pos_dir + '/**/*.jpg')
validation_images_2 = glob.glob(args.val_neg_dir + '/*.jpg')+glob.glob(args.val_neg_dir + '/**/*.jpg')
validation_images=validation_images_1+validation_images_2


# Get image lists
random.shuffle(training_images)
training_labels = [1 if args.train_pos_dir in image else 0 for image in training_images]

random.shuffle(validation_images)
validation_labels = [1 if args.val_pos_dir in image else 0 for image in validation_images]

# check for unacceptable data imbalances
n_train_positive = sum(training_labels)
n_train_negative = len(training_labels) - n_train_positive
print('Number of positive training samples: ' + str(n_train_positive))
print('Number of negative training samples: ' + str(n_train_negative))
n_val_positive = sum(validation_labels)
n_val_negative = len(validation_labels) - n_val_positive
print('Number of positive validation samples: ' + str(n_val_positive))
print('Number of negative validation samples: ' + str(n_val_negative))

IMG_SHAPE = (args.img_size, args.img_size, 1)
batch_size = args.bz

# All the images are being stored in the memory.
# Images are being mean subtracted and divided by std.
# Data augmentation is performed on the fly.
train_datagen = DataGeneratorMemory(training_images,
                                    training_labels,
                                    IMG_SHAPE,
                                    augmentation=True,
                                    batch_size=batch_size,
                                    n_classes=2,
                                    balanced=True,
                                    shuffle=True)

validation_datagen = DataGeneratorMemory(validation_images,
                                         validation_labels,
                                         IMG_SHAPE,
                                         augmentation=False,
                                         batch_size=batch_size,
                                         n_classes=2,
                                         shuffle=True,
                                         balanced=False
                                         )

for i in range(args.trial):

    # a random search over these 3 hyper parameters
    learning_rate = random.sample([0.005,0.003,0.001,0.0005,0.00005],1)[0]
    channel_multiplier = random.sample([3,5,8],1)[0]
    dropout = random.sample([0.25,0.35,0.45],1)[0]

    # uncomment if we want to use fixed value
    # learning_rate = args.learning_rate
    # channel_multiplier = args.channel_multiplier
    # dropout = args.dropout

    log_dir = args.save_dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Setup where the logs will be saved.
    saved_model_filepath = "saved-model-{epoch:02d}-{val_acc:.4f}.hdf5"
    params = {'channel_multiplier':channel_multiplier,'lr':learning_rate,'dropout':dropout}
    log_dir = os.path.join(log_dir, parameter_unique_dirname(params))

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True,
                                                          write_images=False)
    checkpoint_callback = keras.callbacks.ModelCheckpoint(os.path.join(log_dir, saved_model_filepath),
                                                            monitor='val_loss', verbose=0, save_best_only=False,
                                                            save_weights_only=False, mode='auto', period=1)



    model = tf.keras.Sequential([
        keras.layers.SeparableConv2D(4 * params['channel_multiplier'], input_shape=IMG_SHAPE,
                                        kernel_size=(3, 3), activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.SeparableConv2D(4 * params['channel_multiplier'], (3, 3), activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(),
        keras.layers.Dropout(params['dropout']),

        keras.layers.SeparableConv2D(8 * params['channel_multiplier'], (3, 3), activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.SeparableConv2D(8 * params['channel_multiplier'], (3, 3), activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(),
        keras.layers.Dropout(params['dropout']),

        keras.layers.SeparableConv2D(16 * params['channel_multiplier'], (3, 3), activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.SeparableConv2D(16 * params['channel_multiplier'], (3, 3), activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(),
        keras.layers.Dropout(params['dropout']),

        keras.layers.SeparableConv2D(32 * params['channel_multiplier'], (3, 3), activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.SeparableConv2D(32 * params['channel_multiplier'], (3, 3), activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(),
        keras.layers.Dropout(params['dropout']),

        keras.layers.SeparableConv2D(64 * params['channel_multiplier'], (3, 3), activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(),
        keras.layers.Dropout(params['dropout']),

        keras.layers.SeparableConv2D(64 * params['channel_multiplier'], (3, 3), activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(),
        keras.layers.Dropout(params['dropout']),

        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dropout(params['dropout']),
        keras.layers.Dense(2, activation='softmax')
    ])

    model.summary()

    steps_per_epoch = train_datagen.n_images // batch_size
    validation_steps = validation_datagen.n_images // batch_size

    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=params['lr']), loss='binary_crossentropy',
                    metrics=['accuracy'])
    checkpoint_callback.set_model(model)

    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)


    model.fit_generator(train_datagen,
                        steps_per_epoch=steps_per_epoch,
                        epochs=args.epoch,
                        workers=4,
                        validation_data=validation_datagen,
                        validation_steps=validation_steps,
                        callbacks=[tensorboard_callback, checkpoint_callback,callback])

    [val_loss, val_acc] = model.evaluate_generator(validation_datagen)
    [train_loss, train_acc] = model.evaluate_generator(train_datagen)

    print("Setup value for trial: "+str(i), params)
    print("Validation accuracy for trial i :", val_acc)

    sys.stdout.flush()
    K.clear_session()

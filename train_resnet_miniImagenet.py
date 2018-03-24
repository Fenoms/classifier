from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import argparse
import os
import sys

import tensorflow as tf 
import resnetmodel
import pickle
from termcolor import colored


_PATH = os.getcwd()

parser = argparse.ArgumentParser(prog = 'miniImagenet_resnet_v2')

parser.add_argument(
    '--resnet_size', type=int, default=18, choices=[18, 34, 50, 101], help='choose which model to use.')

parser.add_argument(
    '--data_format', type=str, default='channels_last', help='the data_format of data.')


_OMNIGLOT_DATA_DIR = _PATH + '/omniglot_data'

_MINIIMAGENET_DATA_DIR = _PATH + '/miniImagenet'
#total training epochs
_TRAIN_EPOCHS = 100
#the number of training epochs between validation
_EPOCHS_PER_EVAL = 1
#batch size
_BATCH_SIZE = 64
    
_DEFAULT_IMAGE_SIZE = 84 # or 224

_NUM_CHANNELS = 3

_MOMENTUM = 0.9

_WEIGHT_DECAY = 1e-4

"""miniImagenet: 64 for directly train the classifier on training data,
                 100 for few-shot training approch on training(64) and validation(16) and test(20) data
   omniglot:     1623 for few-shot training approch on training(1200) and test(423) data
"""
_TOTAL_NUMBER_CLASSES = 64 

"""
    first_step: directly train the classifier on training data
    second_step: train with hard attention, see if the accuracy increases
    third_step: add comparison module for few-shot learning
"""
_TRAIN_FILENAME = 'train.tfrecords'  

_VAL_FILENAME = 'eval.tfrecords'


_NUM_IMAGES = {
    'train': 33005,
    'validation': 5120
}
 
def load_omniglot_data(data_dir, is_training):
    """
    loads training date or validation data
    """

    omniglot_data = np.load(data_dir + '/data.npy')
    omniglot_data = np.reshape(omniglot_data, newshape = (1622, 20, 28, 28, 1))
    omniglot_label = np.arange(1622)
    omniglot_label = tf.onehot(omniglot_label, 1622)
    if is_training:
        tra_data = omniglot_data[:1200]
        tra_label = omniglot_label[:1200]
        return tra_data, tra_label
    else:
        val_data = omniglot_data[1200:1411]
        val_label = omniglot_label[1200:1411]
        return val_data, val_label


def load_miniImagenet_data(data_dir, is_training):
    """load traing data and label from pickled file,
       data_dict_pickle is name of the pickled file
    """
    pickle_in = open("data_dict_pickle", "rb")
    data = pickle.load(pickle_in)
    tr_data = data["training_data"]
    tr_label = data["training_label"]

    assert tr_data.shape[0] == tr_label.shape[0]
    return tr_data, tr_label


def input_fn(is_training, path, batch_size, num_epoch = 1):

    if is_training:
        filename = _TRAIN_FILENAME
    else:
        filename = _VAL_FILENAME

    def record_parser(record):
        keys_to_features = {
            'image_raw': tf.FixedLenFeature((), tf.string),
            'label': tf.FixedLenFeature([], tf.int64)
        }

        parsed = tf.parse_single_example(record, keys_to_features)
        image = tf.decode_raw(parsed['image_raw'], tf.float32)
        image = tf.reshape(image, [_DEFAULT_IMAGE_SIZE, _DEFAULT_IMAGE_SIZE, _NUM_CHANNELS])
        #may improve the training preformance 
        #image = tf.subtract(image, 0.5)
        #image = tf.multiply(image, 2.0)
        if is_training:

            image = tf.random_crop(image, size=[int(_DEFAULT_IMAGE_SIZE*0.75), int(_DEFAULT_IMAGE_SIZE*0.75), _NUM_CHANNELS])
            image = tf.image.resize_images(image, [_DEFAULT_IMAGE_SIZE, _DEFAULT_IMAGE_SIZE])
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_flip_up_down(image)
            image = tf.image.random_hue(image, max_delta=0.05)
            image = tf.image.random_contrast(image, lower=0.3, upper=1.0)
            image = tf.image.random_brightness(image, max_delta=0.2)
            image = tf.image.random_saturation(image, lower=0.0, upper=2.0)
            # Limit the image pixels between [0, 255] in case of overflow.
            image = tf.minimum(image, 255.0)
            image = tf.maximum(image, 0.0)
            #image = tf.image.resize_image_with_crop_pad(image, target_height = _DEFAULT_IMAGE_SIZE
            #                                                   target_width = _DEFAULT_IMAGE_SIZE)
        label = tf.cast(parsed['label'], tf.int32)
        return image, tf.one_hot(label, _TOTAL_NUMBER_CLASSES)

    dataset = tf.data.TFRecordDataset(filename)
    dataset = dataset.shuffle(buffer_size=_NUM_IMAGES['train'])
    dataset = dataset.map(record_parser)
    dataset = dataset.repeat(num_epoch)
    dataset = dataset.batch(_BATCH_SIZE)
    iterator = dataset.make_one_shot_iterator()
    images, labels = iterator.get_next()
    return images, labels


def resnet_model_fn(features, labels, mode, params):
    """our model_fn for ResNet to be used with our Estimator"""
    #TODO why is 6
    tf.summary.image('images', features, max_outputs = 3)

    network = resnetmodel.miniImagenet_resnet_v2(
        params['resnet_size'], _TOTAL_NUMBER_CLASSES, params['data_format'])


    logits = network(inputs = features, is_training = (mode == tf.estimator.ModeKeys.TRAIN))


    predictions = {
        'classes': tf.argmax(logits, axis = 1),
        'probabilities': tf.nn.softmax(logits, name = 'softmax_tensor')
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode = mode, predictions = predictions)


    #calculate loss, which includes sofrmax cross entropy and l2 regularization
    cross_entropy = tf.losses.softmax_cross_entropy(logits = logits, onehot_labels = labels)

    #create a tensor named cross_entropy for logging purposes
    tf.identity(cross_entropy, name='cross_entropy')
    tf.summary.scalar('cross_entropy', cross_entropy)

    # Add weight decay to the loss. We exclude the batch norm variables because
    # doing so leads to a small improvement in accuracy.
    loss = cross_entropy + _WEIGHT_DECAY * tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'batch_normalization' not in v.name])

    if mode == tf.estimator.ModeKeys.TRAIN:
        """
        scale the learning rate during train phase depends on the epochs
        """
        initial_learning_rate = 0.1
        batches_per_epoch = _NUM_IMAGES['train'] / _BATCH_SIZE
        global_step = tf.train.get_or_create_global_step()

        #multiply the learning rate by 0.1 at epoch 30, 60, 80, 90
        boundaries = [int(batches_per_epoch * epoch) for epoch in [30, 60, 80, 90]]
        values = [initial_learning_rate*decay for decay in [1, 0.1, 0.01, 1e-3, 1e-4]]
        learning_rate = tf.train.piecewise_constant(
            tf.cast(global_step, tf.int32), boundaries, values)

        #create a tensor named learning_rate for logging information
        tf.identity(learning_rate, name = 'learning_rate')
        tf.summary.scalar('learning_rate', learning_rate)

        optimizer = tf.train.MomentumOptimizer(
            learning_rate = learning_rate,
            momentum = _MOMENTUM)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, global_step)
    else:
        train_op = None


    #define the accuracy
    accuracy = tf.metrics.accuracy(tf.argmax(labels, axis=1), predictions['classes'])

    metrics = {'accuracy': accuracy}
    
    #create a tensor named train_accuracy for logging purpose
    tf.identity(accuracy[1], name = 'train_accuracy')
    tf.summary.scalar('train_accuracy', accuracy[1])

    return tf.estimator.EstimatorSpec(
        mode = mode,
        predictions = predictions,
        loss = loss,
        train_op = train_op,
        eval_metric_ops = metrics)




def main(unused_argv):
    #using the winograd non-fused algorithms provides a small performance boost
    # winograd's minimal filtering algorithm makes convolution fast over small filter and batch_size(1~64)
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

    #set up the runconfig to only save checkpoints once per training cycle
    run_config = tf.estimator.RunConfig().replace(save_checkpoints_steps = 1000) 
    resnet_classifier = tf.estimator.Estimator(
        model_fn = resnet_model_fn, model_dir = '/tmp/exp_2', config = run_config,
        params = {
            'resnet_size': FLAGS.resnet_size,
            'data_format': FLAGS.data_format
        })

    for _ in range(_TRAIN_EPOCHS // _EPOCHS_PER_EVAL):
        tensor_to_log = {
            'learning_rate': 'learning_rate',
            'cross_entropy': 'cross_entropy',
            'train_accuracy': 'train_accuracy'
        }

        logging_hook = tf.train.LoggingTensorHook(
            tensors = tensor_to_log, every_n_iter = 100)

        print(colored('Starting a training cycle', 'green'))
        resnet_classifier.train(
            input_fn = lambda: input_fn(True, _PATH, _BATCH_SIZE, _EPOCHS_PER_EVAL), 
            hooks = [logging_hook])

        print(colored('Starting to evaluate', 'green'))
        eval_results = resnet_classifier.evaluate(
            input_fn = lambda: input_fn(False, _PATH, _BATCH_SIZE))

        print(eval_results)



if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(argv = [sys.argv[0]] + unparsed)

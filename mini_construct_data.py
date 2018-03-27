import os
import numpy as np
import pandas as pd
from collections import defaultdict
import tarfile
from PIL import Image
import pickle
import tensorflow as tf
import glob

path = os.getcwd()

csv_path = path + '/miniImagenet/csv/'

data_path = path + '/miniImagenet/'


_IMAGE_SIZE = 224 #84

def _read_image_as_array(image, dtype='float32'):
    f = Image.open(image)
    f = f.resize((_IMAGE_SIZE, _IMAGE_SIZE))
    try:
        image = np.asarray(f, dtype=dtype)
    finally:
        if hasattr(f, 'close'):
            f.close()
    return image


def find_five_candidates():
    res = []
    csv_dir = csv_path + 'tra.csv'
    csv = pd.read_csv(csv_dir, sep=',')
    labels = csv.label.unique().tolist()
    img_number = defaultdict(list)
    for k, label in enumerate(labels):
        tar = tarfile.open(data_path + label + '.tar')
        imgs = tar.getmembers()

        img_number[label].append(len(imgs))

    selected_labels = sorted(img_number.items(), key=lambda x: x[1],  reverse=True)

    for c in selected_labels[3:8]:
        label = c[0]
        res.append(label)

    return res

def extract_file(res):

    for label in res:
        tar = tarfile.open(data_path + label + '.tar')
        if not os.path.exists(data_path + label):
            os.makedirs(data_path + label)

        tar.extractall(data_path + label)
        tar.close()

def process_data(res):
  
    tra_data = np.zeros((3000, _IMAGE_SIZE, _IMAGE_SIZE, 3), dtype=np.float32)
    val_data = np.zeros((3000, _IMAGE_SIZE, _IMAGE_SIZE, 3), dtype=np.float32)
    tra_labels = np.zeros((3000,), dtype=np.int8)
    val_labels = np.zeros((3000,), dtype=np.int8)
    for k, label in enumerate(res):
        c = 0
        imgs_folder = data_path + label
        imgs = glob.glob(imgs_folder + '/*.JPEG')
 
        for img in imgs:
            if c < 600:
                try:
                    img_array = _read_image_as_array(img)
                    img_array = np.reshape(img_array, (1,_IMAGE_SIZE, _IMAGE_SIZE, 3))
                    tra_data[c + k*600] = img_array
                    tra_labels[c + k*600] = k
                    c += 1
                except Exception as e:
                    print("skipping image, because " + str(e))
            elif c < 1200:
                try:
                    img_array = _read_image_as_array(img)
                    img_array = np.reshape(img_array, (1,_IMAGE_SIZE, _IMAGE_SIZE, 3))
                    val_data[c - 600 + k*600] = img_array
                    val_labels[c - 600 + k*600] = k
                    c += 1
                except Exception as e:
                    print("skipping image, because " + str(e))
            else:
                print(c)
                break

    # tra_data = tra_data[:nb_train_images]
    # tra_labels = tra_labels[:nb_train_images]
    # val_data = val_data[:nb_val_images]
    # val_labels = val_labels[:nb_val_images]

    tra_data = {"training_data" : tra_data, "training_label" : tra_labels}
    val_data = {"val_data": val_data, "val_label" : val_labels}
    #the name of data is miniImagenet_data
    pickle_tra_in = open("pickle_tra", "wb")
    pickle_val_in = open("pickle_val", "wb")
    pickle.dump(tra_data, pickle_tra_in)
    pickle.dump(val_data, pickle_val_in)
    print("saved successfully")
 

def load_miniImagenet(is_training):
    if is_training:
        pickle_out = open("pickle_tra", "rb")
        data = pickle.load(pickle_out)
        tr_data = data["training_data"]
        tr_label = data["training_label"]
        return tr_data, tr_label
    else:    
        pickle_out = open("pickle_val", "rb")
        data = pickle.load(pickle_out)
        val_data = data["val_data"]
        val_label = data["val_label"]
        return (val_data, val_label)


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def generate_tfrecords():
    """
    used for generating tfrecords
    """    
    tr_data, tr_label = load_miniImagenet(True)
    val_data, val_label = load_miniImagenet(False)

    nb_train_images = tr_data.shape[0]
    nb_val_images = val_data.shape[0]

    tr_filename = 'train.tfrecords'
    val_test_filename = 'eval.tfrecords'
    #generating training data
    with tf.python_io.TFRecordWriter(tr_filename) as writer:
        for index in range(nb_train_images):
            image_raw = tr_data[index].tostring()
            example = tf.train.Example(features = tf.train.Features(
                feature={
                'label': _int64_feature(int(tr_label[index])),
                'image_raw': _bytes_feature(image_raw)
                }))
            writer.write(example.SerializeToString())

    #generating validation data

    with tf.python_io.TFRecordWriter(val_test_filename) as writer:
        for index in range(nb_val_images):
            image_raw = val_data[index].tostring()
            example = tf.train.Example(features = tf.train.Features(
                feature={
                'label': _int64_feature(int(val_label[index])),
                'image_raw': _bytes_feature(image_raw)
                }))

            writer.write(example.SerializeToString())


if __name__ == '__main__':
    res = find_five_candidates()
    print(res)
    print("extract files")
    extract_file(res)
    print("generating pickle files")
    process_data(res)
    print("generating tfrecords")
    generate_tfrecords()

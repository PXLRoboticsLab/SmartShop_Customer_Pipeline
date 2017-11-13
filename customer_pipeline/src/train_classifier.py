#!/usr/bin/env python
import tensorflow as tf
import numpy as np
import rospy
import facenet
import argparse
import math
import pickle
import time
import datetime
import os
from sklearn.svm import SVC
from std_msgs.msg import String
from datetime import datetime


class TrainClassifier:
    BATCH_SIZE = 1000

    def __init__(self, args):
        self.folder = args.data_dir
        self.model = args.model
        self.topic = args.topic
        self.gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)

        with tf.Graph().as_default():
            with tf.Session(
                    config=tf.ConfigProto(gpu_options=self.gpu_options, log_device_placement=False)) as self.sess:
                print('Loading feature extraction model')
                facenet.load_model(self.model)

                self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                self.embedding_size = self.embeddings.get_shape()[1]

                self.pub = rospy.Publisher("trained_model", String, queue_size=1)

                self.sub = rospy.Subscriber(self.topic, String, self.callback)

    def callback(self, data):
        print('Training classifier')

        model = str(datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')) + '.pkl'

        classifier_filename_exp = os.path.expanduser('/home/maarten/Documents/ProjectIris/model/' + model)

        dataset = facenet.get_dataset(self.folder)
        paths, labels = facenet.get_image_paths_and_labels(dataset)

        print('Number of classes: %d' % len(dataset))
        print('Number of images: %d' % len(paths))

        nrof_images = len(paths)
        nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / self.BATCH_SIZE))
        emb_array = np.zeros((nrof_images, self.embedding_size))
        for i in range(nrof_batches_per_epoch):
            start_index = i * self.BATCH_SIZE
            end_index = min((i + 1) * self.BATCH_SIZE, nrof_images)
            paths_batch = paths[start_index:end_index]
            images = facenet.load_data(paths_batch, False, False, 160)
            feed_dict = {self.images_placeholder: images, self.phase_train_placeholder: False}
            emb_array[start_index:end_index, :] = self.sess.run(self.embeddings, feed_dict=feed_dict)

        model = SVC(kernel='linear', probability=True)
        model.fit(emb_array, labels)

        class_names = [cls.name.replace('_', ' ') for cls in dataset]

        with open(classifier_filename_exp, 'wb') as outfile:
            pickle.dump((model, class_names), outfile)

        print('Saved classifier model to file "%s"' % classifier_filename_exp)
        self.pub.publish(classifier_filename_exp)


if __name__ == '__main__':
    rospy.init_node('classifier_training')

    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str,
                        help='Path to a model protobuf (.pb) file.')
    parser.add_argument('data_dir', type=str,
                        help='Path to dir containing the aligned person photos.')
    parser.add_argument('--topic', type=str,
                        help='The ros topic to listen to.', default='/train_command')
    parser.add_argument('--gpu_memory_fraction', type=float,
                        help='Upper bound on the amount of GPU memory that will be used by the process.', default=0.20)

    args = parser.parse_args()

    trainer = TrainClassifier(args)

    while True:
        time.sleep(5)

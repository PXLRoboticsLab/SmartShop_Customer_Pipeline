#!/usr/bin/env python
import argparse
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from scipy import misc
import tensorflow as tf
import os
import time
import uuid
import facenet
from sklearn.cluster import DBSCAN
from std_msgs.msg import String


class TrainClassifier:
    def __init__(self, args):
        self.folder_unknown = args.data_dir_unknown
        self.folder_known = args.data_dir_classifier
        self.model = args.model
        self.topic = args.topic
        self.threshold = args.cluster_threshold
        self.min_size = args.min_cluster_size
        self.gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)

        self.train_pub = rospy.Publisher("train_command", String, queue_size=1)

        with tf.Graph().as_default():
            with tf.Session(
                    config=tf.ConfigProto(gpu_options=self.gpu_options, log_device_placement=False)) as self.sess:
                facenet.load_model(self.model)

                self.images_placeholder = self.sess.graph.get_tensor_by_name("input:0")
                self.embeddings = self.sess.graph.get_tensor_by_name("embeddings:0")
                self.phase_train_placeholder = self.sess.graph.get_tensor_by_name("phase_train:0")
                self.embedding_size = self.embeddings.get_shape()[1]

                self.bridge = CvBridge()

                self.image_sub = rospy.Subscriber(self.topic, Image, self.ros_callback)

    def _load_images_from_folder(self, folder):
        images = []
        paths = []
        for filename in os.listdir(folder):
            img = cv2.imread(os.path.join(folder, filename))
            if img is not None:
                images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                paths.append(os.path.join(folder, filename))
        return images, paths

    def _prepare_images(self, image_list):
        images = []
        for img in image_list:
            prewhitened = facenet.prewhiten(img)
            images.append(prewhitened)

        return images

    def _safe_cluster(self, images, image_paths, cluster, folder):
        count = 1
        rand = uuid.uuid4().hex
        path = os.path.join(folder, rand)
        os.mkdir(path)
        for image_index in cluster:
            if count < 21:
                misc.imsave(os.path.join(path, rand + '_ ' + str('%0*d' % (4, count)) + '.png'),
                            images[image_index])
            if image_index != 0:
                os.remove(image_paths[image_index])
            count += 1

        print("Safed as: " + rand)
        print("Ask customer for name please!")

    def ros_callback(self, data):
        safe = True
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        except CvBridgeError as e:
            print(e)

        image_list, paths = self._load_images_from_folder(self.folder_unknown)
        if len(image_list) > 0:
            image_list.insert(0, cv_image)
            paths.insert(0, "cv_image")
            img_list_prepared = self._prepare_images(image_list)
            feed_dict = {self.images_placeholder: img_list_prepared,
                         self.phase_train_placeholder: False}
            emb = self.sess.run(self.embeddings, feed_dict=feed_dict)
            nrof_images = len(img_list_prepared)
            dist_matrix = np.zeros((nrof_images, nrof_images))
            if nrof_images > 0:
                for m in range(nrof_images):
                    for n in range(nrof_images):
                        dist = np.sqrt(np.sum(np.square(np.subtract(emb[m, :], emb[n, :]))))
                        dist_matrix[m][n] = dist
                        if m == 0 and m != n and dist < 1:
                            safe = False

            db = DBSCAN(eps=self.threshold, min_samples=self.min_size, metric='precomputed')
            db.fit(dist_matrix)
            labels = db.labels_

            no_clusters = len(set(labels)) - (1 if -1 in labels else 0)

            if no_clusters > 0:
                for i in range(no_clusters):
                    print('Cluster {}: {}'.format(i, np.nonzero(labels == i)[0]))
                    if 0 in np.nonzero(labels == i)[0]:
                        safe = False
                        self._safe_cluster(img_list_prepared, paths, np.nonzero(labels == i)[0], self.folder_known)
                        self.train_pub.publish("Train")

            if safe:
                misc.imsave(self.folder_unknown + '/' + uuid.uuid4().hex + '.png', img_list_prepared[0])


if __name__ == '__main__':
    rospy.init_node('unidentified_customer')

    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str,
                        help='Path to a model protobuf (.pb) file.')
    parser.add_argument('data_dir_unknown', type=str,
                        help='Path to dir containing the unidentified person photos.')
    parser.add_argument('data_dir_classifier', type=str,
                        help='Path to dir containing classifier data.')
    parser.add_argument('--topic', type=str,
                        help='The ros topic to listen to.', default='/focus_vision/image/unidentified')
    parser.add_argument('--cluster_threshold', type=float,
                        help='The minimum distance for faces to be in the same cluster', default=1.0)
    parser.add_argument('--min_cluster_size', type=int,
                        help='The minimum amount of pictures required for a cluster.', default=20)
    parser.add_argument('--gpu_memory_fraction', type=float,
                        help='Upper bound on the amount of GPU memory that will be used by the process.', default=0.20)

    args = parser.parse_args()

    trainer = TrainClassifier(args)

    while True:
        time.sleep(5)

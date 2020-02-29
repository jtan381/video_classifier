from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import tensorflow as tf


class create_imageClassify():
    def __init__(self):
        self.model_file = \
            "./data/inception_v3_2016_08_28_frozen.pb"
        self.label_file = "./data/imagenet_slim_labels.txt"
        self.input_height = 299
        self.input_width = 299
        self.input_mean = 0
        self.input_std = 255
        self.input_layer = "input"
        self.output_layer = "InceptionV3/Predictions/Reshape_1"
        self.graph = None
        self.labels = None
        self.input_operation = None
        self.output_operation = None
        self.sess = None

    def load_graph(self):
        self.graph = tf.Graph()
        graph_def = tf.GraphDef()

        with open(self.model_file, "rb") as f:
            graph_def.ParseFromString(f.read())
        with self.graph.as_default():
            tf.import_graph_def(graph_def)

        input_name = "import/" + self.input_layer
        output_name = "import/" + self.output_layer
        self.input_operation = self.graph.get_operation_by_name(input_name)
        self.output_operation = self.graph.get_operation_by_name(output_name)
        self.sess = tf.compat.v1.Session(graph=self.graph)

    def read_tensor_from_image_file(self, frame):

        image_reader = frame
        float_caster = tf.cast(image_reader, tf.float32)
        dims_expander = tf.expand_dims(float_caster, 0)
        resized = tf.image.resize_bilinear(
            dims_expander, [self.input_height, self.input_width])
        normalized = tf.divide(tf.subtract(
            resized, [self.input_mean]), [self.input_std])
        sess = tf.compat.v1.Session()
        result = sess.run(normalized)

        return result

    def load_labels(self):
        label = []
        proto_as_ascii_lines = tf.gfile.GFile(self.label_file).readlines()
        for l in proto_as_ascii_lines:
            label.append(l.rstrip())
        return label

    def predict(self, frame):
        t = self.read_tensor_from_image_file(frame)

        # with tf.compat.v1.Session(graph=self.graph) as sess:
        results = self.sess.run(self.output_operation.outputs[0], {
            self.input_operation.outputs[0]: t
        })
        results = np.squeeze(results)

        top_result = results.argsort()[-1:][::-1]

        return results, top_result

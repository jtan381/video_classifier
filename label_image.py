# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

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

    def load_graph(self):
        self.graph = tf.Graph()
        graph_def = tf.GraphDef()

        with open(self.model_file, "rb") as f:
            graph_def.ParseFromString(f.read())
        with self.graph.as_default():
            tf.import_graph_def(graph_def)

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

    def init(self, frame):
        t = self.read_tensor_from_image_file(frame)
        input_name = "import/" + self.input_layer
        output_name = "import/" + self.output_layer
        input_operation = self.graph.get_operation_by_name(input_name)
        output_operation = self.graph.get_operation_by_name(output_name)

        with tf.compat.v1.Session(graph=self.graph) as sess:
            results = sess.run(output_operation.outputs[0], {
                input_operation.outputs[0]: t
            })
        results = np.squeeze(results)

        top_result = results.argsort()[-1:][::-1]

        return results, top_result

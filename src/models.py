# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import tensorflow as tf

from .utils import load_embeddings, normalize_embeddings

def _normal_init(size):
    return tf.random.normal(size, dtype=tf.float64)

class Discriminator():
    def __init__(self, params):
        self.emb_dim = params.emb_dim
        self.dis_layers = params.dis_layers
        self.dis_hid_dim = params.dis_hid_dim
        self.dis_dropout = params.dis_dropout
        self.dis_input_dropout = params.dis_input_dropout

        self._var_list = []
        for i in range(self.dis_layers + 1):
            input_dim = self.emb_dim if i == 0 else self.dis_hid_dim
            output_dim = 1 if i == self.dis_layers else self.dis_hid_dim
            W = tf.Variable(_normal_init([input_dim, output_dim]), name="dis_W_%d" % (i))
            B = tf.Variable(_normal_init([output_dim]), name="dis_B_%d" % (i))
            self._var_list += [W, B]

    def get_var_list(self):
        return self._var_list

    def call(self, x):
        assert len(x.get_shape().as_list()) == 2 and x.get_shape().as_list()[1] == self.emb_dim
        layer = tf.nn.dropout(x, keep_prob=1.0-self.dis_input_dropout)
        for i in range(self.dis_layers + 1):
            W, B = self._var_list[i*2 : i*2+2]
            layer = tf.matmul(layer, W) + B
            if i < self.dis_layers:
                layer = tf.nn.leaky_relu(layer, alpha=0.2)
                layer = tf.nn.dropout(layer, keep_prob=1.0-self.dis_dropout)
        layer = tf.sigmoid(layer)
        return tf.squeeze(layer)

class Generator():
    def __init__(self, params):
        self.emb_dim = params.emb_dim

        self.mapping = None
        if getattr(params, 'map_id_init', True):
            self.mapping = tf.Variable(tf.eye(self.emb_dim, dtype=tf.float64), name="generator_mapping")
        else:
            tf.Variable(tf.truncated_normal([self.emb_dim, self.emb_dim], dtype=tf.float64), name="generator_mapping")

        self._var_list = [self.mapping]

    def get_var_list(self):
        return self._var_list

    def eval_mapping(self, session):
        return self.mapping.eval(session=session)

    def update_mapping(self, mapping):
        assert tuple(mapping.get_shape().as_list()) == tuple(self.mapping.get_shape().as_list())
        self.mapping = mapping

    def call(self, x):
        assert len(x.get_shape().as_list()) == 2 and x.get_shape().as_list()[1] == self.emb_dim
        layer = tf.transpose(tf.matmul(self.mapping, tf.transpose(x)), name="generator_called")
        return layer

    def orthogonalize(self, beta):
        """
        Orthogonalize the mapping.
        """
        assert beta > 0
        self.update_mapping((1 + beta) * self.mapping - beta * tf.matmul(self.mapping, tf.matmul(tf.transpose(self.mapping), self.mapping)))

def build_model(params, with_dis):
    """
    Build all components of the model.
    """
    # source embeddings

    params.sess = tf.Session()
    src_dico, _src_emb = load_embeddings(params, source=True)
    params.src_dico = src_dico
    src_emb = _src_emb

    # target embeddings
    if params.tgt_lang:
        tgt_dico, _tgt_emb = load_embeddings(params, source=False)
        params.tgt_dico = tgt_dico
        tgt_emb = _tgt_emb
    else:
        tgt_emb = None

    # mapping
    generator = Generator(params)

    # discriminator
    discriminator = Discriminator(params) if with_dis else None

    # normalize embeddings
    src_emb, params.src_mean = normalize_embeddings(src_emb, params.normalize_embeddings)
    if params.tgt_lang:
        tgt_emb, params.tgt_mean = normalize_embeddings(tgt_emb, params.normalize_embeddings)

    return src_emb, tgt_emb, generator, discriminator

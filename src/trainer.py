# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
from logging import getLogger
import scipy
import scipy.linalg

import numpy as np
import tensorflow as tf

from .utils import get_optimizer, load_embeddings, normalize_embeddings, export_embeddings
from .utils import clip_parameters
from .dico_builder import build_dictionary
from .evaluation.word_translation import DIC_EVAL_PATH, load_identical_char_dico, load_dictionary


logger = getLogger()


class Trainer(object):

    def __init__(self, src_emb, tgt_emb, generator, discriminator, params):
        """
        Initialize trainer script.
        """
        self.src_emb = src_emb
        self.tgt_emb = tgt_emb
        self.src_dico = params.src_dico
        self.tgt_dico = getattr(params, 'tgt_dico', None)
        self.generator = generator
        self.discriminator = discriminator
        self.sess = params.sess
        self.params = params

        # optimizers
        if hasattr(params, 'map_optimizer'):
            optim_fn, optim_params = get_optimizer(params.map_optimizer)
            self.map_optimizer = optim_fn(**optim_params)
        if hasattr(params, 'dis_optimizer'):
            optim_fn, optim_params = get_optimizer(params.dis_optimizer)
            self.dis_optimizer = optim_fn(**optim_params)
        else:
            assert discriminator is None

        # best validation score
        self.best_valid_metric = -1e12

        self.decrease_lr = False

    def get_dis_xy(self, volatile):
        """
        Get discriminator input batch / output target.
        """
        # select random word IDs
        bs = self.params.batch_size
        mf = self.params.dis_most_frequent
        assert mf <= min(len(self.src_dico), len(self.tgt_dico))
        src_ids = tf.random.uniform((bs,), minval=0, maxval=len(self.src_dico) if mf == 0 else mf, dtype=tf.int64)
        tgt_ids = tf.random.uniform((bs,), minval=0, maxval=len(self.tgt_dico) if mf == 0 else mf, dtype=tf.int64)

        # get word embeddings
        src_emb = tf.gather(self.src_emb, src_ids)
        tgt_emb = tf.gather(self.tgt_emb, tgt_ids)
        src_emb = self.generator.call(src_emb)

        # input / target
        x = tf.concat([src_emb, tgt_emb], axis=0, name="dis_xy_x")
        y = tf.Variable(tf.zeros((2 * bs,), dtype=tf.float64), name="dis_xy_y")
        tf.assign(y, tf.convert_to_tensor(
            [1 - self.params.dis_smooth for i in range(bs)] + [self.params.dis_smooth for i in range(bs)],
            dtype=tf.float64))

        return x, y

    def dis_step(self, stats):
        """
        Train the discriminator.
        """

        # loss
        x, y = self.get_dis_xy(volatile=True)
        preds = self.discriminator.call(x)
        loss = tf.losses.softmax_cross_entropy(y, preds)
        #stats['DIS_COSTS'].append(loss.data[0])

        '''
        # check NaN
        if (loss != loss).data.any():
            logger.error("NaN detected (discriminator)")
            exit()
        '''

        # optim
        self.dis_optimizer.minimize(loss, var_list=self.discriminator.get_var_list())
        clip_parameters(self.discriminator, self.params.dis_clip_weights)

    def mapping_step(self, stats):
        """
        Fooling discriminator training step.
        """
        if self.params.dis_lambda == 0:
            return 0

        # loss
        x, y = self.get_dis_xy(volatile=False)
        preds = self.discriminator.call(x)
        loss = tf.losses.softmax_cross_entropy(1 - y, preds) * self.params.dis_lambda

        '''
        # check NaN
        if (loss != loss).data.any():
            logger.error("NaN detected (fool discriminator)")
            exit()
        '''

        # optim
        self.map_optimizer.minimize(loss, var_list=self.generator.get_var_list())
        self.orthogonalize()

        return 2 * self.params.batch_size

    def load_training_dico(self, dico_train):
        """
        Load training dictionary.
        """
        word2id1 = self.src_dico.word2id
        word2id2 = self.tgt_dico.word2id

        # identical character strings
        if dico_train == "identical_char":
            self.dico = load_identical_char_dico(word2id1, word2id2)
        # use one of the provided dictionary
        elif dico_train == "default":
            filename = '%s-%s.0-5000.txt' % (self.params.src_lang, self.params.tgt_lang)
            self.dico = load_dictionary(
                os.path.join(DIC_EVAL_PATH, filename),
                word2id1, word2id2
            )
        # dictionary provided by the user
        else:
            self.dico = load_dictionary(dico_train, word2id1, word2id2)

    def build_dictionary(self):
        """
        Build a dictionary from aligned embeddings.
        """
        src_emb = self.generator.call(self.src_emb)
        tgt_emb = self.tgt_emb
        src_emb = tf.divide(src_emb, tf.norm(src_emb, ord=2, axis=1, keepdims=True))
        tgt_emb = tf.divide(tgt_emb, tf.norm(tgt_emb, ord=2, axis=1, keepdims=True))
        self.dico = build_dictionary(src_emb, tgt_emb, self.params)

    def procrustes(self):
        """
        Find the best orthogonal matrix mapping using the Orthogonal Procrustes problem
        https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
        """
        A = tf.gather(self.src_emb, self.dico[:, 0])
        B = tf.gather(self.tgt_emb, self.dico[:, 1])
        M = tf.matmul(tf.transpose(B), A)
        U, S, V_t = scipy.linalg.svd(M.eval(session=self.sess), full_matrices=True)
        self.generator.update_mapping(tf.convert_to_tensor(U.dot(V_t), dtype=tf.float64))

    def orthogonalize(self):
        """
        Orthogonalize the mapping.
        """
        if self.params.map_beta > 0:
            self.generator.orthogonalize(self.params.map_beta)

    def update_lr(self, to_log, metric):
        """
        Update learning rate when using SGD.
        """
        if 'sgd' not in self.params.map_optimizer:
            return
        old_lr = self.map_optimizer.param_groups[0]['lr']
        new_lr = max(self.params.min_lr, old_lr * self.params.lr_decay)
        if new_lr < old_lr:
            logger.info("Decreasing learning rate: %.8f -> %.8f" % (old_lr, new_lr))
            self.map_optimizer.param_groups[0]['lr'] = new_lr

        if self.params.lr_shrink < 1 and to_log[metric] >= -1e7:
            if to_log[metric] < self.best_valid_metric:
                logger.info("Validation metric is smaller than the best: %.5f vs %.5f"
                            % (to_log[metric], self.best_valid_metric))
                # decrease the learning rate, only if this is the
                # second time the validation metric decreases
                if self.decrease_lr:
                    old_lr = self.map_optimizer.param_groups[0]['lr']
                    self.map_optimizer.param_groups[0]['lr'] *= self.params.lr_shrink
                    logger.info("Shrinking the learning rate: %.5f -> %.5f"
                                % (old_lr, self.map_optimizer.param_groups[0]['lr']))
                self.decrease_lr = True

    def save_best(self, to_log, metric):
        """
        Save the best model for the given validation metric.
        """
        # best mapping for the given validation criterion
        if to_log[metric] > self.best_valid_metric:
            # new best mapping
            self.best_valid_metric = to_log[metric]
            logger.info('* Best value for "%s": %.5f' % (metric, to_log[metric]))
            # save the mapping
            W = self.generator.eval_mapping(self.sess)
            path = os.path.join(self.params.exp_path, 'best_mapping.pth')
            logger.info('* Saving the mapping to %s ...' % path)
            W.dump(path)

    def reload_best(self):
        """
        Reload the best mapping.
        """
        path = os.path.join(self.params.exp_path, 'best_mapping.pth')
        logger.info('* Reloading the best model from %s ...' % path)
        # reload the model
        assert os.path.isfile(path)
        to_reload = tf.convert_to_tensor(np.load(path), dtype=tf.float64)
        self.generator.update_mapping(to_reload)

    def export(self):
        """
        Export embeddings.
        """
        params = self.params

        # load all embeddings
        logger.info("Reloading all embeddings for mapping ...")
        params.src_dico, src_emb = load_embeddings(params, source=True, full_vocab=True)
        params.tgt_dico, tgt_emb = load_embeddings(params, source=False, full_vocab=True)

        # apply same normalization as during training
        normalize_embeddings(src_emb, params.normalize_embeddings, mean=params.src_mean)
        normalize_embeddings(tgt_emb, params.normalize_embeddings, mean=params.tgt_mean)

        # map source embeddings to the target space
        bs = 4096
        logger.info("Map source embeddings to the target space ...")
        for i, k in enumerate(range(0, len(src_emb), bs)):
            x = tf.Variable(src_emb[k:k + bs])
            src_emb[k:k + bs] = self.generator.call(x).eval(session=params.sess)

        # write embeddings to the disk
        export_embeddings(src_emb, tgt_emb, params)

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
import tensorflow as tf

from .utils import get_nn_avg_dist

logger = getLogger()


def get_candidates(emb1, emb2, params):
    """
    Get best translation pairs candidates.
    """
    bs = 128

    all_scores = []
    all_targets = []

    # number of source words to consider
    n_src = emb1.get_shape().as_list()[0]
    if params.dico_max_rank > 0 and not params.dico_method.startswith('invsm_beta_'):
        n_src = min(params.dico_max_rank, n_src)

    # nearest neighbors
    if params.dico_method == 'nn':

        # for every source word
        for i in range(0, n_src, bs):

            # compute target words scores
            scores = tf.tensordot(emb1[i:min(n_src, i + bs)], tf.transpose(emb2), 1)
            best_scores, best_targets = tf.nn.top_k(scores, k=2, sorted=True)

            # update scores / potential targets
            all_scores.append(best_scores)
            all_targets.append(best_targets)

        all_scores = tf.concat(all_scores, axis=0)
        all_targets = tf.concat(all_targets, axis=0)

    # inverted softmax
    elif params.dico_method.startswith('invsm_beta_'):

        beta = float(params.dico_method[len('invsm_beta_'):])

        # for every target word
        for i in range(0, emb2.size(0), bs):

            # compute source words scores
            scores = tf.exp(tf.multiply(tf.tensordot(emb1, tf.transpose(emb2[i:min(n_src, i + bs)]), 1), beta))
            scores = tf.divide(scores, tf.reduce_sum(scores, axis=0, keepdims=True))

            best_scores, best_targets = tf.nn.top_k(scores, k=2, sorted=True)

            # update scores / potential targets
            all_scores.append(best_scores)
            all_targets.append((best_targets + i))

        all_scores = tf.concat(all_scores, axis=1)
        all_targets = tf.concat(all_targets, axis=1)

        all_scores, best_targets = tf.nn.top_k(all_scores, k=2, sorted=True)
        all_targets = tf.gather(all_targets, best_targets, axis=1)

    # contextual dissimilarity measure
    elif params.dico_method.startswith('csls_knn_'):

        knn = params.dico_method[len('csls_knn_'):]
        assert knn.isdigit()
        knn = int(knn)

        # average distances to k nearest neighbors
        average_dist1 = tf.convert_to_tensor(get_nn_avg_dist(emb2, emb1, knn), dtype=tf.float64)
        average_dist2 = tf.convert_to_tensor(get_nn_avg_dist(emb1, emb2, knn), dtype=tf.float64)

        # for every source word
        for i in range(0, n_src, bs):

            # compute target words scores
            scores = tf.tensordot(emb1[i:min(n_src, i + bs)], tf.transpose(emb2), 1)
            scores = tf.multiply(scores, 2)
            scores = tf.subtract(scores, tf.add(average_dist1[i:min(n_src, i + bs)][:, None], average_dist2[None, :]))
            best_scores, best_targets = tf.nn.top_k(scores, k=2, sorted=True)

            # update scores / potential targets
            all_scores.append(best_scores)
            all_targets.append(best_targets)

        all_scores = tf.concat(all_scores, axis=0)
        all_targets = tf.concat(all_targets, axis=0)

    all_pairs = tf.concat([
        tf.expand_dims(tf.range(0, all_targets.get_shape().as_list()[0], dtype=tf.int32), axis=1),
        tf.expand_dims(all_targets[:, 0], axis=1)
    ], 1)

    # sanity check
    assert tuple(all_scores.get_shape().as_list()) == tuple(all_pairs.get_shape().as_list()) == (n_src, 2)

    # sort pairs by score confidence
    diff = all_scores[:, 0] - all_scores[:, 1]
    reordered = tf.nn.top_k(diff, k = diff.get_shape().as_list()[0], sorted=True)[1]
    all_scores = tf.gather(all_scores, reordered)
    all_pairs = tf.gather(all_pairs, reordered)

    # max dico words rank
    if params.dico_max_rank > 0:
        tops, _ = tf.nn.top_k(all_pairs, k=1, sorted = True)
        selected = tops <= params.dico_max_rank
        #mask = tf.broadcast_to(selected, all_scores.get_shape())
        all_scores = tf.boolean_mask(all_scores, tf.squeeze(selected, axis=1))
        all_pairs = tf.boolean_mask(all_pairs, tf.squeeze(selected, axis=1))

    # max dico size
    if params.dico_max_size > 0:
        all_scores = all_scores[:params.dico_max_size]
        all_pairs = all_pairs[:params.dico_max_size]

    # min dico size
    diff = all_scores[:, 0] - all_scores[:, 1]
    if params.dico_min_size > 0:
        diff[:params.dico_min_size] = 1e9

    # confidence threshold
    if params.dico_threshold > 0:
        selected = diff > params.dico_threshold
        logger.info("Selected %i / %i pairs above the confidence threshold." %
                    (tf.reduce_sum(tf.cast(selected, tf.int32)).eval(session=params.sess), diff.get_shape().as_list[0]))
        all_pairs = tf.boolean_mask(all_pairs, tf.squeeze(selected, axis=1))

    return all_pairs


def build_dictionary(src_emb, tgt_emb, params, s2t_candidates=None, t2s_candidates=None):
    """
    Build a training dictionary given current embeddings / mapping.
    """
    logger.info("Building the train dictionary ...")
    s2t = 'S2T' in params.dico_build
    t2s = 'T2S' in params.dico_build
    assert s2t or t2s

    if s2t:
        if s2t_candidates is None:
            s2t_candidates = get_candidates(src_emb, tgt_emb, params)
    if t2s:
        if t2s_candidates is None:
            t2s_candidates = get_candidates(tgt_emb, src_emb, params)
        t2s_candidates = tf.concat([t2s_candidates[:, 1:], t2s_candidates[:, :1]], 1)

    if params.dico_build == 'S2T':
        dico = s2t_candidates
    elif params.dico_build == 'T2S':
        dico = t2s_candidates
    else:
        s2t_candidates = set([(a, b) for a, b in s2t_candidates.eval(session=params.sess)])
        t2s_candidates = set([(a, b) for a, b in t2s_candidates.eval(session=params.sess)])
        if params.dico_build == 'S2T|T2S':
            final_pairs = s2t_candidates | t2s_candidates
        else:
            assert params.dico_build == 'S2T&T2S'
            final_pairs = s2t_candidates & t2s_candidates
            if len(final_pairs) == 0:
                logger.warning("Empty intersection ...")
                return None
        dico = tf.convert_to_tensor([(int(a), int(b),) for (a, b) in final_pairs], dtype=tf.int64)

    logger.info('New train dictionary of %i pairs.' % dico.eval(session=params.sess).shape[0])
    return dico

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import io
from logging import getLogger
import numpy as np
import tensorflow as tf

from ..utils import get_nn_avg_dist


DIC_EVAL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'crosslingual', 'dictionaries')


logger = getLogger()


def load_identical_char_dico(word2id1, word2id2):
    """
    Build a dictionary of identical character strings.
    """
    pairs = [(w1, w1) for w1 in word2id1.keys() if w1 in word2id2]
    if len(pairs) == 0:
        raise Exception("No identical character strings were found. "
                        "Please specify a dictionary.")

    logger.info("Found %i pairs of identical character strings." % len(pairs))

    # sort the dictionary by source word frequencies
    pairs = sorted(pairs, key=lambda x: word2id1[x[0]])
    dico = tf.convert_to_tensor([(word2id1[word1], word2id2[word2]) for (word1, word2) in pairs], dtype=tf.int64)

    return dico


def load_dictionary(path, word2id1, word2id2):
    """
    Return a tensorflow tensor of size (n, 2) where n is the size of the
    loader dictionary, and sort it by source word frequency.
    """
    assert os.path.isfile(path)

    pairs = []
    not_found = 0
    not_found1 = 0
    not_found2 = 0

    with io.open(path, 'r', encoding='utf-8') as f:
        for _, line in enumerate(f):
            assert line == line.lower()
            word1, word2 = line.rstrip().split()
            if word1 in word2id1 and word2 in word2id2:
                pairs.append((word1, word2))
            else:
                not_found += 1
                not_found1 += int(word1 not in word2id1)
                not_found2 += int(word2 not in word2id2)

    logger.info("Found %i pairs of words in the dictionary (%i unique). "
                "%i other pairs contained at least one unknown word "
                "(%i in lang1, %i in lang2)"
                % (len(pairs), len(set([x for x, _ in pairs])),
                   not_found, not_found1, not_found2))

    # sort the dictionary by source word frequencies
    pairs = sorted(pairs, key=lambda x: word2id1[x[0]])
    dico = tf.convert_to_tensor([(word2id1[word1], word2id2[word2]) for (word1, word2) in pairs], dtype=tf.int64)

    return dico


def get_word_translation_accuracy(lang1, word2id1, emb1, lang2, word2id2, emb2, method, dico_eval, session):
    """
    Given source and target word embeddings, and a dictionary,
    evaluate the translation accuracy using the precision@k.
    """
    if dico_eval == 'default':
        path = os.path.join(DIC_EVAL_PATH, '%s-%s.5000-6500.txt' % (lang1, lang2))
    else:
        path = dico_eval
    dico = load_dictionary(path, word2id1, word2id2)

    assert tf.reduce_max(dico[:, 0]).eval(session=session) < emb1.shape[0]
    assert tf.reduce_max(dico[:, 1]).eval(session=session) < emb2.shape[0]

    # normalize word embeddings
    emb1 = tf.divide(emb1, tf.norm(emb1, ord=2, axis=1, keepdims=True))
    emb2 = tf.divide(emb2, tf.norm(emb2, ord=2, axis=1, keepdims=True))

    # nearest neighbors
    if method == 'nn':
        query = tf.gather(emb1, dico[:, 0])
        scores = tf.tensordot(query, tf.transpose(emb2), 1)

    # inverted softmax
    elif method.startswith('invsm_beta_'):
        beta = float(method[len('invsm_beta_'):])
        bs = 128
        word_scores = []
        for i in range(0, emb2.size(0), bs):
            scores = tf.exp(tf.multiply(tf.tensordot(emb1, tf.transpose(emb2[i:i + bs]), 1), beta))
            scores = tf.divide(scores, tf.reduce_sum(scores, axis=0, keepdims=True))
            word_scores.append(scores.index_select(0, dico[:, 0]))
        scores = tf.concat(word_scores, axis=1)

    # contextual dissimilarity measure
    elif method.startswith('csls_knn_'):
        # average distances to k nearest neighbors
        knn = method[len('csls_knn_'):]
        assert knn.isdigit()
        knn = int(knn)
        average_dist1 = get_nn_avg_dist(emb2, emb1, knn)
        average_dist2 = get_nn_avg_dist(emb1, emb2, knn)
        average_dist1 = tf.convert_to_tensor(average_dist1, dtype=tf.float64)
        average_dist2 = tf.convert_to_tensor(average_dist2, dtype=tf.float64)
        # queries / scores
        query = tf.gather(emb1, dico[:, 0])
        scores = tf.multiply(tf.tensordot(query, tf.transpose(emb2), 1), 2)
        scores = tf.subtract(scores, tf.gather(average_dist1, dico[:, 0])[:, None] + average_dist2[None, :])

    else:
        raise Exception('Unknown method: "%s"' % method)

    results = []
    _, top_matches = tf.nn.top_k(scores, k=10, sorted=True)
    top_matches = tf.to_int64(top_matches)
    for k in [1, 5, 10]:
        top_k_matches = top_matches[:, :k]
        _matching = tf.reduce_sum(tf.cast(tf.math.equal(top_k_matches, dico[:, 1][:, None]), tf.int32), axis=1).eval(session=session)
        # allow for multiple possible translations
        matching = {}
        for i, src_id in enumerate(dico[:, 0].eval(session=session)):
            matching[src_id] = min(matching.get(src_id, 0) + _matching[i], 1)
        # evaluate precision@k
        precision_at_k = 100 * np.mean(list(matching.values()))
        logger.info("%i source words - %s - Precision at k = %i: %f" %
                    (len(matching), method, k, precision_at_k))
        results.append(('precision_at_%i' % k, precision_at_k))

    return results

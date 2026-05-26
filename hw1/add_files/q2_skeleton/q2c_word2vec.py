#!/usr/bin/env python
import random

from flask import g
import numpy as np

from helpers.utils import normalize_rows, sigmoid, get_negative_samples
from q2a_softmax import softmax
from q2b_gradcheck import gradcheck_naive


def naive_softmax_loss_and_gradient(
        center_word_vec,
        outside_word_idx,
        outside_vectors,
        dataset
):
    """ Naive Softmax loss & gradient function for word2vec models """
    
    # 1. Compute the scores (dot product of center word and all outside vectors)
    # outside_vectors is (V, D), center_word_vec is (D,) -> scores is (V,)
    scores = np.dot(outside_vectors, center_word_vec)
    
    # 2. Apply softmax to get the predicted probabilities (y_hat)
    y_hat = softmax(scores)
    
    # 3. Calculate the loss: J = -log(y_hat[actual_outside_word])
    loss = -np.log(y_hat[outside_word_idx])
    
    # 4. Calculate the gradient with respect to the scores (y_hat - y)
    # y is a one-hot vector where the true outside word is 1, rest are 0.
    d_scores = y_hat.copy()
    d_scores[outside_word_idx] -= 1.0
    
    # 5. Calculate gradient with respect to the center word vector (v_c)
    # dJ / dv_c = U^T * (y_hat - y)
    grad_center_vec = np.dot(outside_vectors.T, d_scores)
    
    # 6. Calculate gradient with respect to the outside word vectors (U)
    # dJ / dU = (y_hat - y) * v_c^T  -> This is an outer product
    grad_outside_vecs = np.outer(d_scores, center_word_vec)
    
    return loss, grad_center_vec, grad_outside_vecs
    ### END YOUR CODE

def neg_sampling_loss_and_gradient(
        center_word_vec,
        outside_word_idx,
        outside_vectors,
        dataset,
        K=10
):

    neg_sample_word_indices = get_negative_samples(outside_word_idx, dataset, K)

    uo = outside_vectors[outside_word_idx]
    U_neg = outside_vectors[neg_sample_word_indices]

    loss = -np.log(sigmoid(uo @ center_word_vec)) - np.sum(np.log(sigmoid(-U_neg @ center_word_vec)))

    # gradients
    sig_pos = sigmoid(uo @ center_word_vec)
    sig_neg = sigmoid(-U_neg @ center_word_vec)

    # dJ/dvc
    # using broadcast to scale each uk by (1 - sig_neg[k])
    grad_center_vec = (sig_pos - 1) * uo + np.sum((1 - sig_neg)[:, None] * U_neg, axis=0)

    # dJ/∂duo
    grad_outside_vecs = np.zeros_like(outside_vectors)
    grad_outside_vecs[outside_word_idx] += (sig_pos - 1) * center_word_vec

    # dJ/∂uk  for each negative sample
    for i, k in enumerate(neg_sample_word_indices):
        grad_outside_vecs[k] += (sig_neg[i] - 1) * (-center_word_vec)
    # after summing we get the full dJ/dU

    return loss, grad_center_vec, grad_outside_vecs

def skipgram(current_center_word, outside_words, word2ind,
             center_word_vectors, outside_vectors, dataset,
             word2vec_loss_and_gradient=naive_softmax_loss_and_gradient):
    """ Skip-gram model in word2vec

    Implement the skip-gram model in this function.

    Arguments:
    current_center_word -- a string of the current center word
    outside_words -- list of no more than 2*window_size strings, the outside words
    word2ind -- a dictionary that maps words to their indices in
              the word vector list
    center_word_vectors -- center word vectors (as rows) for all words in vocab
                        (V in pdf handout)
    outside_vectors -- outside word vectors (as rows) for all words in vocab
                    (U in pdf handout)
    word2vec_loss_and_gradient -- the loss and gradient function for
                               a prediction vector given the outside_word_idx
                               word vectors, could be one of the two
                               loss functions you implemented above.

    Return:
    loss -- the loss function value for the skip-gram model
            (J in the pdf handout)
    grad_center_vecs -- the gradient with respect to the center word vectors
            (dJ / dV in the pdf handout)
    grad_outside_vectors -- the gradient with respect to the outside word vectors
                        (dJ / dU in the pdf handout)
    """
    loss = 0.0
    grad_center_vecs = np.zeros(center_word_vectors.shape)
    grad_outside_vectors = np.zeros(outside_vectors.shape)
    
    center_word_index = word2ind[current_center_word]
    vc = center_word_vectors[center_word_index]

    for outside_word in outside_words:
        outside_word_index = word2ind[outside_word]
        curr_loss, curr_grad_center_vec, curr_grad_outside_vecs = word2vec_loss_and_gradient(
            vc, outside_word_index, outside_vectors, dataset)
        loss += curr_loss
        grad_center_vecs[center_word_index] += curr_grad_center_vec
        grad_outside_vectors += curr_grad_outside_vecs

    return loss, grad_center_vecs, grad_outside_vectors


#############################################
# Testing functions below. DO NOT MODIFY!   #
#############################################


def word2vec_sgd_wrapper(word2vec_model, word2ind, word_vectors, dataset,
                         window_size, word2vec_loss_and_gradient=naive_softmax_loss_and_gradient):
    batchsize = 50
    loss = 0.0
    grad = np.zeros(word_vectors.shape)
    N = word_vectors.shape[0]
    center_word_vectors = word_vectors[:int(N / 2), :]
    outside_vectors = word_vectors[int(N / 2):, :]
    for i in range(batchsize):
        window_size1 = random.randint(1, window_size)
        center_word, context = dataset.getRandomContext(window_size1)

        c, gin, gout = word2vec_model(
            center_word, context, word2ind, center_word_vectors,
            outside_vectors, dataset, word2vec_loss_and_gradient
        )
        loss += c / batchsize
        grad[:int(N / 2), :] += gin / batchsize
        grad[int(N / 2):, :] += gout / batchsize

    return loss, grad


def test_word2vec_basic():
    """ Test the two word2vec implementations, before running on Stanford Sentiment Treebank """
    dataset = type('dummy', (), {})()

    def dummy_sample_token_idx():
        return random.randint(0, 4)

    def get_random_context(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0, 4)], \
            [tokens[random.randint(0, 4)] for _ in range(2*C)]
    dataset.sampleTokenIdx = dummy_sample_token_idx
    dataset.getRandomContext = get_random_context

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalize_rows(np.random.randn(10, 3))
    dummy_tokens = dict([("a", 0), ("b", 1), ("c", 2), ("d", 3), ("e", 4)])
    '''
    print("==== Gradient check for skip-gram with naive_softmax_loss_and_gradient ====")
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, naive_softmax_loss_and_gradient),
        dummy_vectors, "naive_softmax_loss_and_gradient Gradient")

    '''
    
    print("==== Gradient check for skip-gram with neg_sampling_loss_and_gradient ====")
    print(neg_sampling_loss_and_gradient(dummy_vectors[0], 1, dummy_vectors[5:], dataset, K=5)[0])
    '''
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, neg_sampling_loss_and_gradient),
                    dummy_vectors, "neg_sampling_loss_and_gradient Gradient")
    '''

    print("\n=== Results ===")
    '''
    print("Skip-Gram with naive_softmax_loss_and_gradient")

    print("Your Result:")
    print("Loss: {}\nGradient wrt Center Vectors (dJ/dV):\n {}\nGradient wrt Outside Vectors (dJ/dU):\n {}\n".format(
            *skipgram("c", ["a", "b", "e", "d", "b", "c"], dummy_tokens,
                      dummy_vectors[:5, :], dummy_vectors[5:, :], dataset)
        )
    )

    print("Expected Result: Value should approximate these:")
    print("""Loss: 11.16610900153398
Gradient wrt Center Vectors (dJ/dV):
 [[ 0.          0.          0.        ]
 [ 0.          0.          0.        ]
 [-1.26947339 -1.36873189  2.45158957]
 [ 0.          0.          0.        ]
 [ 0.          0.          0.        ]]
Gradient wrt Outside Vectors (dJ/dU):
 [[-0.41045956  0.18834851  1.43272264]
 [ 0.38202831 -0.17530219 -1.33348241]
 [ 0.07009355 -0.03216399 -0.24466386]
 [ 0.09472154 -0.04346509 -0.33062865]
 [-0.13638384  0.06258276  0.47605228]]
    """)
    '''

    print("Skip-Gram with neg_sampling_loss_and_gradient")
    print("Your Result:")
    print("Loss: {}\nGradient wrt Center Vectors (dJ/dV):\n {}\n Gradient wrt Outside Vectors (dJ/dU):\n {}\n".format(
        *skipgram("c", ["a", "b"], dummy_tokens, dummy_vectors[:5, :], dummy_vectors[5:, :],
                  dataset, neg_sampling_loss_and_gradient)
        )
    )
    print("Expected Result: Value should approximate these:")
    print("""Loss: 16.15119285363322
Gradient wrt Center Vectors (dJ/dV):
 [[ 0.          0.          0.        ]
 [ 0.          0.          0.        ]
 [-4.54650789 -1.85942252  0.76397441]
 [ 0.          0.          0.        ]
 [ 0.          0.          0.        ]]
 Gradient wrt Outside Vectors (dJ/dU):
 [[-0.69148188  0.31730185  2.41364029]
 [-0.22716495  0.10423969  0.79292674]
 [-0.45528438  0.20891737  1.58918512]
 [-0.31602611  0.14501561  1.10309954]
 [-0.80620296  0.36994417  2.81407799]]
    """)


if __name__ == "__main__":
    test_word2vec_basic()

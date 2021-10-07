# The tagger.py starter code for CSC384 A4.

import os
import sys

import numpy as np
from collections import Counter

UNIVERSAL_TAGS = [
    "VERB",
    "NOUN",
    "PRON",
    "ADJ",
    "ADV",
    "ADP",
    "CONJ",
    "DET",
    "NUM",
    "PRT",
    "X",
    ".",
]

N_tags = len(UNIVERSAL_TAGS)

def read_data_train(path):
    return [tuple(line.split(' : ')) for line in open(path, 'r').read().split('\n')[:-1]]

def read_data_test(path):
    return open(path, 'r').read().split('\n')[:-1]

def read_data_ind(path):
    return [int(line) for line in open(path, 'r').read().split('\n')[:-1]]

def write_results(path, results):
    with open(path, 'w') as f:
        f.write('\n'.join(results))

def train_HMM(train_file_name):
    """
    Estimate HMM parameters from the provided training data.

    Input: Name of the training files. Two files are provided to you:
            - file_name.txt: Each line contains a pair of word and its Part-of-Speech (POS) tag
            - file_name.ind: The i'th line contains an integer denoting the starting index of the i'th sentence in the text-POS data above

    Output: Three pieces of HMM parameters stored in LOG PROBABILITIES :
 
            - prior:        - An array of size N_tags
                            - Each entry corresponds to the prior log probability of seeing the i'th tag in UNIVERSAL_TAGS at the beginning of a sequence
                            - i.e. prior[i] = log P(tag_i)

            - transition:   - A 2D-array of size (N_tags, N_tags)
                            - The (i,j)'th entry stores the log probablity of seeing the j'th tag given it is a transition coming from the i'th tag in UNIVERSAL_TAGS
                            - i.e. transition[i, j] = log P(tag_j|tag_i)

            - emission:     - A dictionary type containing tuples of (str, str) as keys
                            - Each key in the dictionary refers to a (TAG, WORD) pair
                            - The TAG must be an element of UNIVERSAL_TAGS, however the WORD can be anything that appears in the training data
                            - The value corresponding to the (TAG, WORD) key pair is the log probability of observing WORD given a TAG
                            - i.e. emission[(tag, word)] = log P(word|tag)
                            - If a particular (TAG, WORD) pair has never appeared in the training data, then the key (TAG, WORD) should not exist.

    Hints: 1. Think about what should be done when you encounter those unseen emission entries during deccoding.
           2. You may find Python's builtin Counter object to be particularly useful 
    """

    pos_data = read_data_train(train_file_name+'.txt')
    sent_inds = read_data_ind(train_file_name+'.ind')

    ####################
    # STUDENT CODE HERE
    sent_data = []
    for start, end in zip(sent_inds, sent_inds[1:]):
        sent_data.append(pos_data[start:end])
    sent_data.append(pos_data[sent_inds[-1]:])  # don't forget the last sentence!

    prior = np.zeros(N_tags)
    transition = np.zeros((N_tags, N_tags))
    emission = dict()

    for sent in sent_data:
        prior[UNIVERSAL_TAGS.index(sent[0][1])] += 1
    prior = np.log(prior / prior.sum())

    for sent in sent_data:
        for first, second in zip(sent, sent[1:]):
            transition[UNIVERSAL_TAGS.index(first[1])][UNIVERSAL_TAGS.index(second[1])] += 1
    for i, row in enumerate(transition):
        transition[i] = [np.log(float(i)/sum(row)) for i in row]
    transition = np.asarray(transition)

    tag_count = {t: 0 for t in UNIVERSAL_TAGS}
    for pos in pos_data:
        emission.setdefault((pos[1], pos[0]), 0)
        emission[(pos[1], pos[0])] += 1
        tag_count[pos[1]] += 1
    for key in emission:
        emission[key] = np.log(float(emission[key])/tag_count[key[0]])
    ####################

    return prior, transition, emission
    

def tag(train_file_name, test_file_name):
    """
    Train your HMM model, run it on the test data, and finally output the tags.

    Your code should write the output tags to (test_file_name).pred, where each line contains a POS tag as in UNIVERSAL_TAGS
    """

    prior, transition, emission = train_HMM(train_file_name)

    pos_data = read_data_test(test_file_name+'.txt')
    sent_inds = read_data_ind(test_file_name+'.ind')

    ####################
    # STUDENT CODE HERE
    sent_data = []
    for start, end in zip(sent_inds, sent_inds[1:]):
        sent_data.append(pos_data[start:end])
    sent_data.append(pos_data[sent_inds[-1]:])  # don't forget the last sentence!

    results = []
    eps = 0.00001  # smoothing value for '-inf' log probability
    for sent in sent_data:
        prob_trellis = np.zeros((N_tags, len(sent)))
        path_trellis = np.zeros((N_tags, len(sent))).tolist()
        for i in range(N_tags):
            prob_trellis[i, 0] = prior[i] + emission.setdefault((UNIVERSAL_TAGS[i], sent[0]), np.log(eps))
            path_trellis[i][0] = [UNIVERSAL_TAGS[i]]
        for i in range(1, len(sent)):
            o = sent[i]
            for j, s in enumerate(UNIVERSAL_TAGS):
                x = np.argmax([prob_trellis[k, i-1] + transition[k, j] + emission.setdefault((s, o), np.log(eps)) for k in range(N_tags)])
                prob_trellis[j, i] = prob_trellis[x, i-1] + transition[x, j] + emission.setdefault((s, o), np.log(eps))
                path_trellis[j][i] = path_trellis[x][i-1] + [s]
        result = path_trellis[np.argmax(prob_trellis[:, -1])][-1]
        results += result
    ####################

    write_results(test_file_name+'.pred', results)

if __name__ == '__main__':
    # Run the tagger function.
    print("Starting the tagging process.")

    # Tagger expects the input call: "python3 tagger.py -d <training file> -t <test file>"
    # E.g. python3 tagger.py -d data/train-public -t data/test-public-small
    parameters = sys.argv
    train_file_name = parameters[parameters.index("-d")+1]
    test_file_name = parameters[parameters.index("-t")+1]

    # Start the training and tagging operation.
    tag(train_file_name, test_file_name)

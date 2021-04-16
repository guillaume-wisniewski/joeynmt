"""Simple script to extract the representation of the words of the
sentence as predicted by the encoder.

Input:
- the source sentence
- the pickle file storing the representations extracted from `joeynmt`

Output:
- a list of sentences
- each sentence is a list of tuples (word, word representation)

Todo:
- extract the representation for all the layers?
"""

import pickle
import argparse

from itertools import zip_longest

import torch

import pandas as pd


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--representations", required=True, type=argparse.FileType("rb"))
parser.add_argument("--input_sentences", required=True, type=argparse.FileType("r"))
parser.add_argument("--layer", default=-1, type=int,
                    help="select the encoder layer on which the representation should be extracted"\
                    " (as a python index --> -1 for the last one, 0 for the first)")
parser.add_argument("--output", required=True, type=argparse.FileType("wb"))
args = parser.parse_args()

n_sentences = 0
for batch in pickle.load(args.representations):

    sort_reverse_index = batch['sort_reverse_index']
    hidden_state = batch["hidden_state"][args.layer]

    if n_sentences == 0:
        print(f"extracting representation from the {hidden_state['layer_id']}-th layer")

    # output is a tensor of dimensions: n_examples x max_ex_length x repr_dimension
    output = hidden_state["output"]
    mask = hidden_state["output_mask"]

    # all values in paddind dimensions are set to 0
    #
    # XXX not sure why we have to transpose the mask here. Maybe, this
    # allows to make the mask independant of the dimensions of the
    # representation (i.e. we can use the same mask to remove value when
    # computing the attention and when computing the word representation.
    masked_output = output * mask.transpose(1, 2)

    # restore the order of the sentences in the batch (we have to apply
    # the mask first to avoid forgetting to sort the mask **and** the
    # batch data)
    masked_output = masked_output[sort_reverse_index,:,:]

    # extract the representation
    all_representations = []
    for i in range(masked_output.shape[0]):
        n_sentences += 1
        input_sentence = next(args.input_sentences)
        input_sentence = input_sentence.split() + ["<eos>"]

        o = masked_output[i, :, :]
        sentence_representation = o[o.sum(dim=1) != 0,:]

        assert len(input_sentence) == sentence_representation.shape[0]

        all_representations.append(list(zip(input_sentence, sentence_representation)))

    pickle.dump(all_representations, args.output)

print(f"extracted the representation of {n_sentences:,} sentences")

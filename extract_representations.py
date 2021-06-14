"""Simple script to extract the representation of the words of the
sentence as predicted by the encoder.

Input:
- the source sentence
- the pickle file storing the representations extracted from `joeynmt`

Output:

- a pickle file with the representation of the source sentence
 predicted by the encoder. This file contains a list of sentences;
 each sentence is a list of tuples (word, word representation) where
 word representation is a vector (a `tensor.Tensor` object)

- a json/yaml file with the self-attention between the words of the
  input sentences. This file contains a list of sentences; each
  sentence is a list of dictionaries; each dictionnary has two keys:

  - `head`: an integer that identifies the head (there are 8 heads in
    a "vanilla" transformer model)
  - `attention_scores`: a list the i-th element of which are the
    attention of the i-th sentence. Attentions are described by a list
    of tuples `(source_word, attention_scores)`; `attention_scores` is
    a list `(target_word, `attention`) in which `attention` is a
    probability describing the attention between `source_word` and
    `target_word`.

Todo:
- extract the representation for all the layers?

"""

import pickle
import argparse
import json

from itertools import zip_longest
from pathlib import Path

import yaml
import torch


parser = argparse.ArgumentParser()
parser.add_argument("--representations", required=True, type=argparse.FileType("rb"))
parser.add_argument("--input_sentences", required=True, type=argparse.FileType("r"))
parser.add_argument("--layer", default=-1, type=int,
                    help="select the encoder layer on which the representation should be extracted"\
                    " (as a python index --> -1 for the last one, 0 for the first)")
parser.add_argument("--repr_output", required=True, type=argparse.FileType("wb"))
parser.add_argument("--attn_output", required=True, type=lambda x: Path(x))
args = parser.parse_args()

n_sentences = 0
all_representations = []
all_self_attentions = []
for batch in pickle.load(args.representations):
    sort_reverse_index = batch['sort_reverse_index']
    hidden_state = batch["hidden_state"][args.layer]

    if n_sentences == 0:
        print(f"extracting representation from the {hidden_state['layer_id']}-th layer")

        
    # output is a tensor of dimensions: n_examples x max_ex_length x repr_dimension
    output = hidden_state["output"]
    mask = hidden_state["output_mask"]
    self_attentions = hidden_state["self_attn"]

    # all values in padding dimensions are set to 0
    #
    # XXX not sure why we have to transpose the mask here. Maybe, this
    # allows to make the mask independant of the dimensions of the
    # representation (i.e. we can use the same mask to remove value when
    # computing the attention and when computing the word
    # representation.
    #
    # XXX We should also apply the mask to the attentions matrix (but
    # I have to find the correct broadcast operation)
    masked_output = output * mask.transpose(1, 2)
    
    # restore the order of the sentences in the batch (we have to apply
    # the mask first to avoid forgetting to sort the mask **and** the
    # batch data)
    masked_output = masked_output[sort_reverse_index,:,:]
    self_attentions = self_attentions[sort_reverse_index,:,:]

    for i in range(masked_output.shape[0]):
        n_sentences += 1
        input_sentence = next(args.input_sentences)
        input_sentence = input_sentence.split() + ["<eos>"]

        # extract the representation
        o = masked_output[i, :, :]
        sentence_representation = o[o.sum(dim=1) != 0,:]
        assert len(input_sentence) == sentence_representation.shape[0], f"{len(input_sentence)} vs {sentence_representation.shape[0]}"

        all_representations.append(list(zip(input_sentence, sentence_representation)))

        # extract the attentions
        all_heads = self_attentions[i, :, :]
        all_source_attn = []
        for head in range(all_heads.shape[0]):
            this_head = all_heads[head, :, :]
            
            # this_head.sum(axis=1) == 1 --> attentions are
            # probability --> the attention between word w1 and word
            # w2 is to be read at this_head[w1, w2] (w1 = row, w2 =
            # column)
            source_attn = []
            for source_word in range(len(input_sentence)):
                attn = [[input_sentence[target_word], float(this_head[source_word, target_word])]
                        for target_word in range(len(input_sentence))]
                source_attn.append((input_sentence[source_word], attn))

            all_source_attn.append({"head": head,
                                    "attention_scores": source_attn })

        all_self_attentions.append(all_source_attn)

pickle.dump(all_representations, args.repr_output)

print(f"extracted the representation of {n_sentences:,} sentences")

with open(args.attn_output.with_suffix(".yaml"), "w") as ofile:
    yaml.dump(all_self_attentions, ofile)

with open(args.attn_output.with_suffix(".json"), "w") as ofile:
    json.dump(all_self_attentions, ofile)

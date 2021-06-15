"""
Input: representations of French sentences extracted by `extract_representations.py`
Output: Three representations of the `son` token 
- a representation averaged over all tokens
- the representation of son in the first sentence (translation correctly predicted: ▁the ▁captain ▁has ▁completed ▁his ▁work ▁.)
- the representation of son in the 4-th sentnce (translation correctly predicted: 

"""
import argparse
import pickle
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--repr", required=True, type=argparse.FileType("rb"))

args = parser.parse_args()

data = pickle.load(args.repr)


all_son_representations = []

for idx, sentence in enumerate(data):
    words = [s[0] for s in sentence]

    assert words[-4] == "▁son"

    all_son_representations.append(sentence[-4][1].tolist())

    # le facteur a terminé son travail
    if idx == 4:
        male_son = np.array(sentence[-4][1].tolist())

    # la pharmacienne a terminé son travail
    if idx == 141:
        female_son = np.array(sentence[-4][1].tolist())

all_son_representations = np.array(all_son_representations)

avg_son = np.mean(all_son_representations, axis=0)

print(f"avg: {avg_son.shape}")
print(f"male: {male_son.shape}")
print(f"female: {female_son.shape}")

pickle.dump(avg_son, open("average_son.pkl", "wb"))
pickle.dump(male_son, open("male_son.pkl", "wb"))
pickle.dump(female_son, open("female_son.pkl", "wb"))

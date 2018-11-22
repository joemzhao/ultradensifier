from __future__ import print_function
from scipy.stats import kendalltau as mytau


gold = []
with open("gold.txt", "r") as f:
    for line in f.readlines():
        line = line.strip().split("\t")
        gold.append((line[0], float(line[1])))
gold = sorted(gold, key=lambda x: x[1], reverse=True)
print ("Finish loading the ground truth lexicon ...")

tocompare = []
with open("en_twitter_sentiment.txt", "r") as f:
    for line in f.readlines():
        line = line.strip().split(" ")
        tocompare.append((line[0], float(line[1])))
tocompare = sorted(tocompare, key=lambda x: x[1], reverse=True)
print ("Finish loading the eval lexicon ...")
mytocompare = {}
for idx, pair in enumerate(tocompare):
    mytocompare[pair[0]] = idx

X, Y = [], []
for idx, pair in enumerate(gold):
    try:
        Y.append(mytocompare[pair[0]])
    except KeyError:
        continue
    X.append(idx)

print (mytau(X, Y))

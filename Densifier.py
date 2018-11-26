from __future__ import print_function
from __future__ import division

import os
os.environ["MKL_NUM_THREADS"] = "40"
os.environ["NUMEXPR_NUM_THREADS"] = "40"
os.environ["OMP_NUM_THREADS"] = "40"

from helpers import *

import tqdm
import itertools
import copy
import time


def parse_words(add_bib):
    pos, neg = [], []
    with open("sentiment-lx/mypos.txt", "r") as f:
        for line in f.readlines():
            if not add_bib:
                pos.append(line.strip())
            else:
                pos.append(line.strip()+"@bib")
    with open("sentiment-lx/myneg.txt", "r") as f:
        for line in f.readlines():
            if not add_bib:
                neg.append(line.strip())
            else:
                neg.append(line.strip()+"@bib")
    return pos, neg


def batches(it, size=16):
    batch = []
    for item in it:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if len(batch) > 0: yield batch # yield the last several items


class Densifier(object):
    def __init__(self, d, ds, lr, batch_size, seed=3):
        self.d = d
        self.ds = ds
        self.Q = np.matrix(scipy.stats.ortho_group.rvs(d, random_state=seed))
        self.Q[1:, :] = 0.
        self.P = np.matrix(np.eye(ds, d))
        self.D = np.transpose(self.P) * self.P
        self.zeros_d = np.matrix(np.zeros((self.d, self.d)))
        self.lr = lr
        self.batch_size = batch_size
        self.alpha = 0.4

    def train(self, num_epoch, pos_vecs, neg_vecs, save_to, save_every):
        bs = self.batch_size
        save_step = 0
        diff_ps = list(itertools.product(pos_vecs, neg_vecs))
        same_ps = list(itertools.combinations(pos_vecs, 2)) + \
                  list(itertools.combinations(neg_vecs, 2))
        self.E, self.EET = {}, {} # EET is computed in the first pass
        for (ew, ev) in diff_ps+same_ps:
            self.E[(id(ew), id(ev))] = np.asarray(ew) - np.asarray(ev)
        for e in xrange(num_epoch):
            random.shuffle(diff_ps)
            random.shuffle(same_ps)
            steps_orth = 0
            steps_print = 0
            steps_same_loss, steps_diff_loss = [], []
            for (mini_diff, mini_same) in zip(batches(diff_ps, bs), batches(same_ps, bs)):
                steps_orth += 1
                steps_print += 1
                save_step += 1

                VEC_DIFF, diff_grad= [], []
                for (ew, ev) in mini_diff:
                    VEC_DIFF.append(self.E[(id(ew), id(ev))])
                VEC_DIFF = np.asarray(VEC_DIFF)
                DIFF_LOSS = np.absolute(VEC_DIFF * self.Q[0, :].reshape(self.d,1))
                for idx, (ew, ev) in enumerate(mini_diff):
                    _loss = DIFF_LOSS[idx]
                    if _loss == 0:
                        diff_grad.append(self.zeros_d[0, :])
                        continue
                    elif (id(ew), id(ev)) not in self.EET:
                        _e = self.E[(id(ew), id(ev))].reshape(self.d, 1)
                        self.EET[(id(ew), id(ev))] = _e * np.transpose(_e)
                    _diff_step_grad = self.Q[0, :] * self.EET[(id(ew), id(ev))] / _loss
                    diff_grad.append(_diff_step_grad)

                VEC_SAME, same_grad = [], []
                for (ew, ev) in mini_same:
                    VEC_SAME.append(self.E[(id(ew), id(ev))])
                VEC_SAME = np.asarray(VEC_SAME)
                SAME_LOSS = np.absolute(VEC_SAME * self.Q[0, :].reshape(self.d,1))
                for idx, (ew, ev) in enumerate(mini_same):
                    _loss = SAME_LOSS[idx]
                    if _loss == 0:
                        same_grad.append(self.zeros_d[0, :])
                        continue
                    elif (id(ew), id(ev)) not in self.EET:
                        _e = self.E[(id(ew), id(ev))].reshape(self.d, 1)
                        self.EET[(id(ew), id(ev))] = _e * np.transpose(_e)
                    _same_step_grad = self.Q[0, :] * self.EET[(id(ew), id(ev))] / _loss
                    same_grad.append(_same_step_grad)

                diff_grad = np.mean(diff_grad, axis=0)
                same_grad = np.mean(same_grad, axis=0)

                self.Q[0, :] -= self.lr * (-1. * self.alpha * diff_grad + (1.-self.alpha) * same_grad)
                steps_same_loss.append(np.mean(SAME_LOSS))
                steps_diff_loss.append(np.mean(DIFF_LOSS))
                if steps_print % 10 == 0:
                    print ("=" * 25)
                    try:
                        print ("Diff-loss: {:4f}, Same-loss: {:4f}, LR: {:4f}".format(
                        np.mean(steps_diff_loss), np.mean(steps_same_loss), self.lr))
                    except:
                        print (np.mean(steps_diff_loss))
                        print (np.mean(steps_same_loss))
                        print (self.lr)
                    steps_same_loss, steps_diff_loss = [], []
                if steps_orth % 10 == 0:
                    self.Q = Densifier.make_orth(self.Q)
                    self.lr *= 0.999
                if save_step % save_every == 0:
                    self.save(save_to)
                    print ("Model saved! Step: {}".format(save_step))
            print ("="*25 + " one epoch finished! ({}) ".format(e) + "="*25)
        print ("Training finished ...")
        self.save(save_to)

    def save(self, save_to):
        with open(save_to, "w") as f:
            pickle.dump(self.__dict__, f)
        print ("Trained model saved ...")

    @staticmethod
    def make_orth(Q):
        U, _, V = np.linalg.svd(Q)
        return U * V


if __name__ == "__main__":
    from gensim.models import KeyedVectors
    from sys import exit

    import argparse
    import random

    random.seed(3)

    parser = argparse.ArgumentParser()
    parser.add_argument("--LR", type=float, default=5.)
    parser.add_argument("--EPC", type=int, default=2)
    parser.add_argument("--OUT_DIM", type=int, default=1)
    parser.add_argument("--BIBLE_SEED_EMB", type=int, default=1)
    parser.add_argument("--BATCH_SIZE", type=int, default=100)
    parser.add_argument("--EMB_SPACE", type=str, default="embeddings/mybible_400.vec")
    parser.add_argument("--SAVE_EVERY", type=int, default=1000)
    parser.add_argument("--SAVE_TO", type=str, default="trained_densifier.pkl")
    args = parser.parse_args()

    pos_words, neg_words = parse_words(add_bib=True)
    myword2vec = word2vec(args.EMB_SPACE)
    print ("Finish loading embedding ...")

    map(lambda x: random.shuffle(x), [pos_words, neg_words])
    pos_vecs, neg_vecs = map(lambda x: emblookup(x, myword2vec), [pos_words, neg_words])
    pos_vecs = pos_vecs[:5]
    neg_vecs = neg_vecs[:5]
    print (len(pos_vecs), len(neg_vecs))

    assert len(pos_vecs) > 0
    assert len(neg_vecs) > 0
    mydensifier = Densifier(400, args.OUT_DIM, args.LR, args.BATCH_SIZE)
    mydensifier.train(args.EPC, pos_vecs, neg_vecs, args.SAVE_TO, args.SAVE_EVERY)

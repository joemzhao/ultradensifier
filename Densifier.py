from __future__ import print_function
from __future__ import division

from helpers import *

import tqdm


def parse_words():
    pos, neg = [], []
    with open("sentiment-lx/mypos.txt", "r") as f:
        for line in f.readlines():
            pos.append(line.strip())
    with open("sentiment-lx/myneg.txt", "r") as f:
        for line in f.readlines():
            neg.append(line.strip())
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
        self.P = np.matrix(np.eye(ds, d))
        self.D = np.transpose(self.P) * self.P
        self.zeros_d = np.matrix(np.zeros((self.d, self.d)))
        self.loss = {}
        self.lr = lr
        self.batch_size = batch_size
        self.alpha = 0.5

    def step_loss(self, ew, ev):
        vec_diff = (ew - ev).reshape(self.d, 1)
        loss = np.linalg.norm(self.P * self.Q * vec_diff, ord=2)
        return loss, vec_diff

    def gradient(self, ew, ev):
        step_loss, vec_diff = self.step_loss(ew, ev)
        if step_loss == 0.:
            print ("WARNING: check if there are replicated seed words!")
            print ("         grad set to 0.")
            return self.zeros_d, 0.
        return self.D * self.Q * vec_diff * np.transpose(vec_diff) / step_loss, step_loss

    def train(self, num_epoch, pos_vecs, neg_vecs, save_to, save_every):
        bs = self.batch_size
        save_step = 0

        diff_ps = list(itertools.product(pos_vecs, neg_vecs))
        same_ps = list(itertools.combinations(pos_vecs, 2)) + \
                  list(itertools.combinations(neg_vecs, 2))

        for e in xrange(num_epoch):
            random.shuffle(diff_ps)
            random.shuffle(same_ps)
            steps_orth = 0
            steps_print = 0
            steps_same_loss, steps_diff_loss = [], []
            steps_same_grad, steps_diff_grad = [], []

            for (mini_diff, mini_same) in zip(batches(diff_ps, bs), batches(same_ps, bs)):
                steps_orth += 1
                steps_print += 1
                save_step += 1

                diff_grad = np.matrix(np.zeros((self.d, self.d)))
                same_grad = np.matrix(np.zeros((self.d, self.d)))

                batch_diff_loss = []
                batch_same_loss = []
                for ew, ev in mini_diff:
                    diff_gred_step, diff_loss_step = self.gradient(ew, ev)
                    diff_grad += diff_gred_step
                    batch_diff_loss.append(diff_loss_step)

                for ew, ev in mini_same:
                    same_grad_step, same_loss_step = self.gradient(ew, ev)
                    same_grad += same_grad_step
                    batch_same_loss.append(same_loss_step)

                diff_grad /= len(mini_diff)
                same_grad /= len(mini_same)
                self.Q -= self.lr * (self.alpha * (-diff_grad) + (1.-self.alpha) * same_grad)

                steps_same_grad.append(same_grad)
                steps_diff_grad.append(diff_grad)
                steps_same_loss.append(np.mean(same_loss))
                steps_diff_loss.append(np.mean(diff_loss))

                self.lr *= 0.99

                if steps_print % 1 == 0:
                    print ("=" * 25)
                    print ("Diff-loss: {:4f}, Same-loss: {:4f}, LR: {:4f}".format(
                    np.mean(steps_diff_loss), np.mean(steps_same_loss), self.lr))
                    steps_same_loss, steps_diff_loss = [], []
                if steps_orth % 1 == 0:
                    self.Q = Densifier.make_orth(self.Q)
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
    parser.add_argument("--EMB_SPACE", type=str, default="embeddings/debug_emb_bible.vec")
    parser.add_argument("--SAVE_EVERY", type=int, default=1000)
    parser.add_argument("--SAVE_TO", type=str, default="trained_densifier.pkl")
    args = parser.parse_args()

    pos_words, neg_words = parse_words()
    myword2vec = word2vec(args.EMB_SPACE)
    print ("Finish loading embedding ...")

    map(lambda x: random.shuffle(x), [pos_words, neg_words])
    pos_vecs, neg_vecs = map(lambda x: emblookup(x, myword2vec), [pos_words, neg_words])
    print (len(pos_vecs), len(neg_vecs))

    assert len(pos_vecs) > 0
    assert len(neg_vecs) > 0
    mydensifier = Densifier(100, args.OUT_DIM, args.LR, args.BATCH_SIZE)
    mydensifier.train(args.EPC, np.asarray(pos_vecs), np.asarray(neg_vecs), args.SAVE_TO, args.SAVE_EVERY)

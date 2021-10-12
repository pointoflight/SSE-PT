import sys
import copy
import random
import numpy as np
from collections import defaultdict

from tensorboard.plugins.hparams import api as hp

# HP_LR = hp.HParam('learning rate', hp.Discrete([0.0005]))
# HP_MAXLEN = hp.HParam('max len', hp.Discrete([50, 100, 150, 200]))
# HP_BS = hp.HParam('batch size', hp.Discrete([128]))
# HP_USERHIDDEN = hp.HParam('user hidden units', hp.Discrete([50, 100]))
# HP_ITEMHIDDEN = hp.HParam('item hidden units', hp.Discrete([50, 100]))
# HP_NUMBLOCKS = hp.HParam('num blocks', hp.Discrete([2, 5]))
# HP_NUMHEADS = hp.HParam('num heads', hp.Discrete([1, 2]))
# HP_DROPOUT = hp.HParam('dropout rate', hp.Discrete([0.2]))
# HP_SSEU = hp.HParam('sse prob user', hp.Discrete([0.08, 0.2, 0.9]))
# HP_SSEI = hp.HParam('sse prob item', hp.Discrete([0.9, 0.99, 0.08]))
# HP_L2 = hp.HParam('l2 emb', hp.Discrete([0.0]))

HP_LR = hp.HParam('learning rate', hp.Discrete([0.0005]))
HP_MAXLEN = hp.HParam('max len', hp.Discrete([50, 100]))
HP_BS = hp.HParam('batch size', hp.Discrete([128]))
HP_USERHIDDEN = hp.HParam('user hidden units', hp.Discrete([50, 100]))
HP_ITEMHIDDEN = hp.HParam('item hidden units', hp.Discrete([50, 100]))
HP_NUMBLOCKS = hp.HParam('num blocks', hp.Discrete([2, 5]))
HP_NUMHEADS = hp.HParam('num heads', hp.Discrete([1]))
HP_DROPOUT = hp.HParam('dropout rate', hp.Discrete([0.2]))
HP_SSEU = hp.HParam('sse prob user', hp.Discrete([0.08, 0.9]))
HP_SSEI = hp.HParam('sse prob item', hp.Discrete([0.9, 0.08]))
HP_L2 = hp.HParam('l2 emb', hp.Discrete([0.0]))


def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open('data/%s.txt' % fname, 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    return [user_train, user_valid, user_test, usernum, itemnum]


def evaluate(model, dataset, args, hparams, sess):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:

        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([hparams[HP_MAXLEN]], dtype=np.int32)
        idx = hparams[HP_MAXLEN] - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        rated = set(train[u])
        rated.add(0)
        item_idx = [test[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(sess, [u], [seq], item_idx)
        predictions = predictions[0]
        #print(predictions)
        rank = predictions.argsort().argsort()[0]

        valid_user += 1

        if rank < args.k:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 1000 == 0:
            #print '.',
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user


def evaluate_valid(model, dataset, args, hparams, sess):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([hparams[HP_MAXLEN]], dtype=np.int32)
        idx = hparams[HP_MAXLEN] - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)
        item_idx = [valid[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(sess, [u], [seq], item_idx)
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0]

        valid_user += 1

        if rank < args.k:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            #print '.',
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user

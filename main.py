import os
import logging 

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

import time
import pickle
import argparse
import tensorflow as tf
from sampler import WarpSampler
from model_v1 import *
#from tqdm import tqdm
from util import *

# %load_ext tensorboard


METRIC_NDCG = 'ndcg'

# tf.summary.create_file_writer("/tmp/mylogs")
# writer = tf.summary.FileWriter('./graphs', sess.graph)

with tf.summary.FileWriter('logs/hparam_tuning'):
  hp.hparams_config(
    hparams=[HP_LR, HP_MAXLEN, HP_BS, HP_USERHIDDEN, HP_ITEMHIDDEN, HP_NUMBLOCKS, HP_NUMHEADS, HP_DROPOUT, HP_SSEI, HP_SSEU, HP_L2],
    metrics=[hp.Metric(METRIC_NDCG, display_name='NDCG@10')],
  )
 
def str2bool(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='ml1m')
parser.add_argument('--train_dir', default='default')
# parser.add_argument('--batch_size', default=128, type=int)
# parser.add_argument('--lr', default=0.001, type=float)
# parser.add_argument('--maxlen', default=50, type=int)
# parser.add_argument('--user_hidden_units', default=50, type=int)
# parser.add_argument('--item_hidden_units', default=50, type=int)
# parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=2000, type=int)
# parser.add_argument('--num_heads', default=1, type=int)
# parser.add_argument('--dropout_rate', default=0.5, type=float)
# parser.add_argument('--threshold_user', default=1.0, type=float)
# parser.add_argument('--threshold_item', default=1.0, type=float)
# parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--print_freq', default=50, type=int)
parser.add_argument('--k', default=10, type=int)

args = parser.parse_args()
if not os.path.isdir(args.dataset + '_' + args.train_dir):
    os.makedirs(args.dataset + '_' + args.train_dir)
with open(os.path.join(args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f:
    params = '\n'.join([str(k) + ',' + str(v) 
        for k, v in sorted(vars(args).items(), key=lambda x: x[0])])
    print(params)
    f.write(params)

os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)


dataset = data_partition(args.dataset)
[user_train, user_valid, user_test, usernum, itemnum] = dataset
cc = 0.0
max_len = 0
for u in user_train:
    cc += len(user_train[u])
    max_len = max(max_len, len(user_train[u]))
print("\nThere are {0} users {1} items \n".format(usernum, itemnum))
print("Average sequence length: {0}\n".format(cc / len(user_train)))
print("Maximum length of sequence: {0}\n".format(max_len))

f = open(os.path.join(args.dataset + '_' + args.train_dir, 'log.txt'), 'w')

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
sess = tf.Session(config=config)



def run(run_dir, args, hparams):

    num_batch = len(user_train) // hparams[HP_BS]

    sampler = WarpSampler(user_train, usernum, itemnum,
        batch_size=hparams[HP_BS], maxlen=hparams[HP_MAXLEN],
        threshold_user=hparams[HP_SSEU],
        threshold_item=hparams[HP_SSEI],
        n_workers=3)

    with tf.summary.FileWriter(run_dir):
        hp.hparams(hparams)  # record the values used in this trial

        model = Model(usernum, itemnum, hparams)
        sess.run(tf.global_variables_initializer())
        
        T = 0.0
        t_test = evaluate(model, dataset, args, hparams, sess)
        t_valid = evaluate_valid(model, dataset, args, hparams, sess)
        print("[0, 0.0, {0:.5f}, {1:.5f}, {2:.5f}, {3:.5f}],".format(t_valid[0], t_valid[1], t_test[0], t_test[1]))
        
        t0 = time.time()

        for epoch in range(1, args.num_epochs + 1):
            for step in range(num_batch):
                u, seq, pos, neg = sampler.next_batch()
                user_emb_table, item_emb_table, attention, auc, loss, _ = sess.run([model.user_emb_table, model.item_emb_table, model.attention, model.auc, model.loss, model.train_op],
                                            {model.u: u, model.input_seq: seq, model.pos: pos, model.neg: neg,
                                             model.is_training: True})
        
            if epoch % args.print_freq == 0:
                t1 = time.time() - t0
                T += t1
                #print 'Evaluating',
                t_test = evaluate(model, dataset, args, hparams, sess)
                t_valid = evaluate_valid(model, dataset, args, hparams, sess)
                #print ''
                #print 'epoch:%d, time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f), test (NDCG@10: %.4f, HR@10: %.4f)' % (
                #epoch, T, t_valid[0], t_valid[1], t_test[0], t_test[1])
                print("[{0}, {1:.2f}, {2:.5f}, {3:.5f}, {4:.5f}, {5:.5f}],".format(epoch, T, t_valid[0], t_valid[1], t_test[0], t_test[1]))

        # t_test = evaluate(model, dataset, args, hparams, sess)
        # t_valid = evaluate_valid(model, dataset, args, hparams, sess)
        # print("[{0:.5f}, {1:.5f}, {2:.5f}, {3:.5f}]".format(t_valid[0], t_valid[1], t_test[0], t_test[1]))
        accuracy = t_valid[0]
        tf.summary.scalar(METRIC_NDCG, accuracy)

    sampler.close()
    



session_num = 0

for hp_lr in HP_LR.domain.values:
    for hp_maxlen in HP_MAXLEN.domain.values:
        for hp_bs in HP_BS.domain.values:
            for hp_user in HP_USERHIDDEN.domain.values:
                for hp_item in HP_ITEMHIDDEN.domain.values:
                    for hp_numblocks in HP_NUMBLOCKS.domain.values:
                        for hp_numheads in HP_NUMHEADS.domain.values:
                            for hp_dropout in HP_DROPOUT.domain.values:
                                for hp_sseu in HP_SSEU.domain.values:
                                    for hp_ssei in HP_SSEI.domain.values:
                                        for hp_l2 in HP_L2.domain.values:
                                            hparams = {
                                              HP_LR: hp_lr,
                                              HP_MAXLEN: hp_maxlen,
                                              HP_BS: hp_bs,
                                              HP_USERHIDDEN: hp_user,
                                              HP_ITEMHIDDEN: hp_item,
                                              HP_NUMBLOCKS: hp_numblocks,
                                              HP_NUMHEADS: hp_numheads,
                                              HP_DROPOUT: hp_dropout,
                                              HP_SSEU: hp_sseu,
                                              HP_SSEI: hp_ssei,
                                              HP_L2: hp_l2,
                                            }
                                            run_name = "run-%d" % session_num
                                            print('--- Starting trial: %s' % run_name)
                                            print({h.name: hparams[h] for h in hparams})
                                            run('logs/hparam_tuning/' + run_name, args, hparams)
                                            session_num+=1



f.close()
# print("Done")

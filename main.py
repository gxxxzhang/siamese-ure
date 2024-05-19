import argparse
import math
import random
import shutil
import datetime
from sklearn.cluster import KMeans
from tqdm import tqdm
import numpy as np

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import models.model_builder
from models import bert_model
from transformers import BertTokenizer
import torch
from torch import nn
import os
import json
from torch.utils.data import TensorDataset
import re
import time
from evaluation import usoon_eval, ClusterEvaluation

from termcolor import colored
from memory import MemoryBank
from transformers import logging

logging.set_verbosity_error()

parser = argparse.ArgumentParser(description='PyTorch SimSiam_URE for relation extraction Training')

parser.add_argument('data', metavar='DIR',
                    help='path to dataset', default='tacred')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--adaptive-clustering', action='store_true',
                    help='Add adaptive clustering to the model')
parser.add_argument('-b', '--batch-size', default=10, type=int,
                    metavar='N',
                    help='mini-batch size (default: 512), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.00001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://localhost:10001', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument("--save_path", default="", type=str)
parser.add_argument('--seed', default=128, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--raw_dim', default=768 * 2, type=int,
                    help='feature dimension ')
parser.add_argument('--hidden-dim', default=512, type=int,
                    help='hidden dimension of the predictor (default: 512)')
parser.add_argument('--pred-dim', default=256, type=int,
                    help='pred dimension of the predictor (default: 256)')
parser.add_argument('--num-cluster', default=10, type=int,
                    help='number of cluster')
parser.add_argument('--aug-plus', action='store_true',
                    help='use moco-v2/SimCLR data augmentation')

parser.add_argument('--exp-dir', default='new_test', type=str,
                    help='experiment directory')
parser.add_argument('--max-length', default=128, type=int,
                    help='max length of sentence to be feed into bert (default 128)')
parser.add_argument('--eps', default=1e-8, type=float,
                    help='adam_epsilon (default: 1e-8)')
parser.add_argument('--add-word-num', default=2, type=int,
                    help='data augmentation words number (default 3)')
parser.add_argument('--use-relation-span', action='store_true',
                    help='whether using relation span for data augmentation')
parser.add_argument('--fix-pred-lr', action='store_true',
                    help='Fix learning rate for the predictor')
parser.add_argument('--warmup-epoch', default=5, type=int,
                    help='number of warm-up epochs to only train with InfoNCE loss')
parser.add_argument('--entropy_weight', default=1.0, type=float)
# ------------------------init parameters----------------------------

CUDA = "0"
os.environ['CUDA_VISIBLE_DEVICES'] = CUDA


def main():
    args = parser.parse_args()

    if args.seed is not None:
        seed_all(args.seed)

    bert_encoder = bert_model.RelationClassification.from_pretrained(
        "bert-base-uncased",  # Use the 12-layer BERT model, with an uncased vocab.
        # You can increase this for multi-class tasks.
        output_attentions=False,  # Whether the model returns attentions weights.
        output_hidden_states=False,  # Whether the model returns all hidden-states.
    )
    tokenizer = get_tokenizer(args)
    bert_encoder.resize_token_embeddings(len(tokenizer))
    model = models.model_builder.SiamURE(
        bert_encoder,
        args.raw_dim, args.hidden_dim, args.pred_dim)

    time = str(datetime.datetime.now()).replace(' ', '_')
    save_path_ = os.path.join(args.save_path, f"{time}")
    create_directory(save_path_)
    args.save_path = save_path_
    # init_lr = args.lr * args.batch_size / 256
    init_lr = args.lr
    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)

    # cosine
    criterion = nn.CosineSimilarity(dim=1).cuda(args.gpu)
    criterion2 = get_criterion(args)
    criterion2.cuda(args.gpu)

    if args.fix_pred_lr:
        optim_params = [{'params': model.encoder.parameters(), 'fix_lr': False},
                        {'params': model.predictor.parameters(), 'fix_lr': True}]
    else:
        optim_params = model.parameters()

    optimizer = torch.optim.SGD(optim_params, init_lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    sentence_train = json.load(open(args.data + 'train_sentence.json', 'r'))
    sentence_train_label = json.load(open(args.data + 'train_label_id.json', 'r'))
    sentence_test = json.load(open(args.data + 'test_sentence.json', 'r'))
    sentence_test_label = json.load(open(args.data + 'test_label_id.json', 'r'))
    # train_set
    train_dataset = pre_processing(sentence_train, sentence_train_label, args)
    # test_set
    test_dataset = pre_processing(sentence_test, sentence_test_label, args)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=args.workers)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, num_workers=args.workers)

    gold = []
    for eval_data in train_dataset:
        gold.append(eval_data[2].item())
    gold_test = []
    for test_data in test_dataset:
        gold_test.append(test_data[2].item())

    best_score = -1
    for epoch in range(args.start_epoch, args.epochs):
        features = compute_features(train_loader, model, args)
        cluster_centers, relation2cluster = clustering_method(features, args)
        trian_score = test_one_epoch(gold, relation2cluster)

        features_test = compute_features(test_loader, model, args)
        _, relation2cluster_test = clustering_method(features_test, args)
        test_score = test_one_epoch(gold_test, relation2cluster_test)

        if trian_score > best_score:
            best_score = trian_score
            print("new test best result: {}".format(test_score))
            ckpt_file = os.path.join(args.save_path, "best.ckpt")
            print(f"saving model checkpoint into {ckpt_file} ...")
            torch.save(model.state_dict(), ckpt_file)
        if epoch > args.warmup_epoch:
            print(colored('Build MemoryBank', 'blue'))
            memory_bank_train = MemoryBank(features.shape[0], 2 * 768, 10, 0.07)
            memory_bank_train.cuda()
            topk = 30
            print(colored('Mine the nearest neighbors (Train)(Top-%d)' % (topk), 'blue'))
            # load
            fill_memory_bank(train_loader, model, memory_bank_train)
            indices, acc = memory_bank_train.mine_nearest_neighbors(topk)
            print('Accuracy of top-%d nearest neighbors on train set is %.2f' % (topk, 100 * acc))
            train_scan_dataset = pre_processing(sentence_train, sentence_train_label, args, indices=indices)
            train_scan_loader = torch.utils.data.DataLoader(
                train_scan_dataset, batch_size=args.batch_size, shuffle=True,
                num_workers=args.workers)
            train(train_scan_loader, model, criterion, criterion2, optimizer, epoch, args, cluster_centers)
        # train for one epoch
        else:
            train(train_loader, model, criterion, criterion2, optimizer, epoch, args, cluster_centers)

        adjust_learning_rate(optimizer, init_lr, epoch, args)


def train(train_loader, model, criterion, criterion2, optimizer, epoch, args, cluster_centers):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    consistency_losses = AverageMeter('Consistency Loss', ':.4f')
    total_losses = AverageMeter('Total Loss', ':.4f')


    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, consistency_losses, total_losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, batch in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            # batch[0] = batch[0].cuda(args.gpu, non_blocking=True)
            # batch[1] = batch[1].cuda(args.gpu, non_blocking=True)
            for b_i in range(len(batch)):
                batch[b_i] = batch[b_i].cuda(args.gpu, non_blocking=True)

        if epoch > args.warmup_epoch:
            p1, p2, z1, z2, z1_, z1_neigh = model(x1=batch[:5], x2=batch[5:10])
            loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5

            consistency_loss = -(criterion(z1_, z1_neigh).mean())
            total_loss = loss + consistency_loss
            consistency_losses.update(consistency_loss.item(), len(batch))
            losses.update(loss.item(), len(batch))
        else:
            p1, p2, z1, z2 = model(x1=batch[:5])
            loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5
            total_loss = loss
            losses.update(loss.item(), len(batch))
        total_losses.update(total_loss.item(), len(batch))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr


def clustering_method(x, args):
    print('performing kmeans clustering')
    clu = KMeans(n_clusters=args.num_cluster, random_state=0, algorithm='full')
    relation2cluster = clu.fit_predict(x)
    cluster_centers = clu.cluster_centers_
    # obtain cluster centres
    cluster_centers = torch.tensor(cluster_centers, dtype=torch.float).cuda()
    return cluster_centers, relation2cluster


def get_tokenizer(args):
    """ Tokenize all of the sentences and map the tokens to their word IDs."""
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    special_tokens = []
    if not args.use_relation_span:
        special_tokens.append('<e1>')
        special_tokens.append('</e1>')
        special_tokens.append('<e2>')
        special_tokens.append('</e2>')
    else:
        # ent_type = ['LOCATION', 'MISC', 'ORGANIZATION', 'PERSON']  # NYT
        ent_type = ['PERSON', 'ORGANIZATION', 'NUMBER', 'DATE', 'NATIONALITY', 'LOCATION', 'TITLE', 'CITY', 'MISC',
                    'COUNTRY', 'CRIMINAL_CHARGE', 'RELIGION', 'DURATION', 'URL', 'STATE_OR_PROVINCE', 'IDEOLOGY',
                    'CAUSE_OF_DEATH']  # Tacred

        for r in ent_type:
            special_tokens.append('<e1:' + r + '>')
            special_tokens.append('<e2:' + r + '>')
            special_tokens.append('</e1:' + r + '>')
            special_tokens.append('</e2:' + r + '>')
    special_tokens_dict = {'additional_special_tokens': special_tokens}  # add special token
    tokenizer.add_special_tokens(special_tokens_dict)
    return tokenizer


def get_token_word_id(sen, tokenizer, args):
    """ Get special token word id """
    if not args.use_relation_span:
        # return 2487,2475
        e1 = '<e1>'
        e2 = '<e2>'
    else:
        e1 = re.search('(<e1:.*?>)', sen).group(1)
        e2 = re.search('(<e2:.*?>)', sen).group(1)
    e1_tks_id = tokenizer.convert_tokens_to_ids(e1)
    e2_tks_id = tokenizer.convert_tokens_to_ids(e2)
    # print('id:',e1_tks_id,'   ',e2_tks_id)
    return e1_tks_id, e2_tks_id
    pass


def pre_processing(sentence_train, sentence_train_label, args, indices=None):
    """Main function for pre-processing data """
    input_ids = []
    attention_masks = []
    labels = []
    e1_pos = []
    e2_pos = []
    index_arr = []

    neighbor_index_one = []
    neighbor_input_ids = []
    neighbor_attention_masks = []
    neighbor_e1_pos = []
    neighbor_e2_pos = []
    print('Loading BERT tokenizer...')
    tokenizer = get_tokenizer(args)
    counter = 0
    # pre-processing sentenses to BERT pattern
    train_sen_file = open('train_sen_file.csv', 'w')
    train_sen_file.write('{}\t{}\t{}\n'.format('index', 'sentence', 'label'))
    train_sen_arr = []

    for i in range(len(sentence_train)):
        encoded_dict = tokenizer.encode_plus(
            sentence_train[i],  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=args.max_length,  # Pad & truncate all sentences.
            truncation=True,  # explicitely truncate examples to max length
            pad_to_max_length=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
        )
        try:
            # Find e1(id:2487) and e2(id:2475) position
            e1_tks_id, e2_tks_id = get_token_word_id(sentence_train[i], tokenizer, args)
            pos1 = (encoded_dict['input_ids'] == e1_tks_id).nonzero(as_tuple=False)[0][1].item()
            pos2 = (encoded_dict['input_ids'] == e2_tks_id).nonzero(as_tuple=False)[0][1].item()
            e1_pos.append(pos1)
            e2_pos.append(pos2)

            # Add the encoded sentence to the list.
            input_ids.append(encoded_dict['input_ids'])
            # And its attention mask (simply differentiates padding from non-padding).
            attention_masks.append(encoded_dict['attention_mask'])
            labels.append(sentence_train_label[i])
            index_arr.append(counter)
            train_sen_file.write('{}\t{}\t{}\n'.format(i, sentence_train[i], sentence_train_label[i]))
            train_sen_arr.append(sentence_train[i])
            counter += 1

        except Exception as e:
            # print(sentence_train[i])
            print(e)
            pass

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)
    e1_pos = torch.tensor(e1_pos)
    e2_pos = torch.tensor(e2_pos)
    index_arr = torch.tensor(index_arr)
    if indices is None:
        # eval_dataset = TensorDataset(input_ids, attention_masks, labels, e1_pos, e2_pos, index_arr)

        # Combine the training inputs into a TensorDataset.
        train_dataset = TensorDataset(input_ids, attention_masks, labels, e1_pos, e2_pos,
                                      index_arr)
        train_sen_file.close()
        return train_dataset
    else:
        for i in range(len(indices)):
            neighbor_index = np.random.choice(indices[i][1:], 1)[0]
            neighbor_index_one.append(neighbor_index)
        print(neighbor_index_one[:10])
        for i in range(len(neighbor_index_one)):
            neighbor_input_ids.append(input_ids[neighbor_index_one[i]])
            neighbor_attention_masks.append(attention_masks[neighbor_index_one[i]])
            neighbor_e1_pos.append(e1_pos[neighbor_index_one[i]])
            neighbor_e2_pos.append(e2_pos[neighbor_index_one[i]])
        # numpy to list
        neighbor_input_ids = [aa.tolist() for aa in neighbor_input_ids]
        neighbor_attention_masks = [aa.tolist() for aa in neighbor_attention_masks]
        neighbor_e1_pos = [aa.tolist() for aa in neighbor_e1_pos]
        neighbor_e2_pos = [aa.tolist() for aa in neighbor_e2_pos]
        # list to tensor
        neighbor_input_ids = torch.tensor(neighbor_input_ids)
        neighbor_attention_masks = torch.tensor(neighbor_attention_masks)
        neighbor_e1_pos = torch.tensor(neighbor_e1_pos)
        neighbor_e2_pos = torch.tensor(neighbor_e2_pos)
        train_dataset = TensorDataset(input_ids, attention_masks, labels, e1_pos, e2_pos, neighbor_input_ids,
                                      neighbor_attention_masks, labels, neighbor_e1_pos, neighbor_e2_pos, index_arr)
        train_sen_file.close()
        return train_dataset


def compute_features(eval_loader, model, args):
    print('Computing features...')
    model.eval()
    features = torch.zeros(len(eval_loader.dataset), 2 * 768).cuda()
    for i, sentence in enumerate(tqdm(eval_loader)):
        with torch.no_grad():
            # sentence = sentence.cuda(non_blocking=True)
            for j in range(len(sentence)):
                sentence[j] = sentence[j].cuda(non_blocking=True)
            # pdb.set_trace()
            feat = model(x1=sentence, pat='test')
            features[sentence[-1]] = feat  # .view(args.low_dim*args.batch_size)
    return features.cpu()


def test_one_epoch(ground_truth, label_pred):
    cluster_eval = ClusterEvaluation(ground_truth, label_pred).printEvaluation()
    B3_f1, B3_prec, B3_rec, v_f1, v_hom, v_comp, ARI = usoon_eval(ground_truth, label_pred)
    print("B3_f1:{}, B3_prec:{}, B3_rec:{}, v_f1:{}, v_hom:{}, v_comp:{}, ARI:{}".format(B3_f1, B3_prec, B3_rec, v_f1,
                                                                                         v_hom, v_comp, ARI))
    print(cluster_eval)
    return cluster_eval['F1']


def get_criterion(args):
    criterion = models.model_builder.SCANLoss(args.entropy_weight)
    return criterion


def seed_all(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def create_directory(d):
    if d and not os.path.exists(d):
        os.makedirs(d)
    return d
@torch.no_grad()
def fill_memory_bank(loader, model, memory_bank):
    model.eval()
    memory_bank.reset()

    for i, batch in enumerate(loader):
        for j in range(len(batch)):
            batch[j] = batch[j].cuda(non_blocking=True)
        sentence = batch
        targets = batch[2]
        with torch.no_grad():
            feat = model(x1=sentence, pat='test')
        memory_bank.update(feat, targets)
        if i % 100 == 0:
            print('Fill Memory Bank [%d/%d]' % (i, len(loader)))


if __name__ == '__main__':
    main()

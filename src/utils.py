import sys
from networks import RelationClassification, LabelGeneration
from torch.optim import AdamW
from transformers import BertTokenizer
import torch
from torch import nn
import torch.nn.functional as F
import random
import numpy as np
import os
import time
import json
import datetime
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, random_split, Subset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import f1_score
from collections import Counter

CUDA = 0
DATASET = 'AL'  # dataset selection tacred,SemEval
NUM_LABELS = 49  # TACRED:42, SemEval:19
MAX_LENGTH = 128
BATCH_SIZE = 16
LR = 1e-4
EPS = 1e-8
EPOCHS = 5  #
TOTAL_EPOCHS = 1
MATE_EPOCHS = 10
seed_val = 19
UNLABEL_OF_TRAIN = 0.5  # Unlabel ratio
LABEL_OF_TRAIN = 0.1  # Label ratio
LAMBD = 0.2
Z = 10    # Incremental Epoch Number
Z_RATIO = Z / BATCH_SIZE
LOG_DIR = DATASET + '_' + str(int(LABEL_OF_TRAIN * 100))
SAVE_DIR = DATASET + "_saved_model"
os.system('mkdir ' + LOG_DIR)
os.system('mkdir ' + SAVE_DIR)


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    non_zero_idx = (labels_flat != 0)
    return np.sum(pred_flat[non_zero_idx] == labels_flat[non_zero_idx]) / len(labels_flat[non_zero_idx])


# Takes a time in seconds and returns a string hh:mm:ss
def format_time(elapsed):
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


# cited: https://github.com/INK-USC/DualRE/blob/master/utils/scorer.py#L26
def score(key, prediction, verbose=True, NO_RELATION=0):
    correct_by_relation = Counter()
    guessed_by_relation = Counter()
    gold_by_relation = Counter()

    # Loop over the data to compute a score
    for row in range(len(key)):
        gold = key[row]
        guess = prediction[row]

        if gold == NO_RELATION and guess == NO_RELATION:
            pass
        elif gold == NO_RELATION and guess != NO_RELATION:
            guessed_by_relation[guess] += 1
        elif gold != NO_RELATION and guess == NO_RELATION:
            gold_by_relation[gold] += 1
        elif gold != NO_RELATION and guess != NO_RELATION:
            guessed_by_relation[guess] += 1
            gold_by_relation[gold] += 1
            if gold == guess:
                correct_by_relation[guess] += 1

    # Print the aggregate score
    if verbose:
        print("Final Score:")
    prec_micro = 1.0
    if sum(guessed_by_relation.values()) > 0:
        prec_micro = float(sum(correct_by_relation.values())) / float(
            sum(guessed_by_relation.values()))
    recall_micro = 0.0
    if sum(gold_by_relation.values()) > 0:
        recall_micro = float(sum(correct_by_relation.values())) / float(
            sum(gold_by_relation.values()))
    f1_micro = 0.0
    if prec_micro + recall_micro > 0.0:
        f1_micro = 2.0 * prec_micro * recall_micro / (prec_micro + recall_micro)
    print("SET NO_RELATION ID: ", NO_RELATION)
    print("Precision (micro): {:.3%}".format(prec_micro))
    print("   Recall (micro): {:.3%}".format(recall_micro))
    print("       F1 (micro): {:.3%}".format(f1_micro))
    return prec_micro, recall_micro, f1_micro


def pre_processing(sentence_train, sentence_train_label):
    input_ids = []
    attention_masks = []
    labels = []
    e1_pos = []
    e2_pos = []

    # Load tokenizer.
    print('Loading BERT tokenizer...')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    # pre-processing sentenses to BERT pattern
    for i in range(len(sentence_train)):
        encoded_dict = tokenizer.encode_plus(
            sentence_train[i],  # Sentence to encode.
            add_special_tokens=False,  # Add '[CLS]' and '[SEP]'
            max_length=MAX_LENGTH,  # Pad & truncate all sentences.
            padding="max_length",
            truncation=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
        )
        try:
            # Find e1(id:2487) and e2(id:2475) position
            pos1 = (encoded_dict['input_ids'] == 2487).nonzero()[0][1].item()
            pos2 = (encoded_dict['input_ids'] == 2475).nonzero()[0][1].item()
            e1_pos.append(pos1)
            e2_pos.append(pos2)
            # Add the encoded sentence to the list.
            input_ids.append(encoded_dict['input_ids'])
            # And its attention mask (simply differentiates padding from non-padding).
            attention_masks.append(encoded_dict['attention_mask'])
            labels.append(sentence_train_label[i])
        except:
            pass
            # print(sent)

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0).to(device)
    attention_masks = torch.cat(attention_masks, dim=0).to(device)
    labels = torch.tensor(labels, device='cuda')
    e1_pos = torch.tensor(e1_pos, device='cuda')
    e2_pos = torch.tensor(e2_pos, device='cuda')
    w = torch.ones(len(e1_pos), device='cuda')

    # Combine the training inputs into a TensorDataset.
    train_dataset = TensorDataset(input_ids, attention_masks, labels, e1_pos, e2_pos, w)

    return train_dataset


def stratified_sample(dataset, ratio):
    data_dict = {}
    for i in range(len(dataset)):
        if not data_dict.get(dataset[i][2].item()):
            data_dict[dataset[i][2].item()] = []
        data_dict[dataset[i][2].item()].append(i)
    sampled_indices = []
    rest_indices = []
    for indices in data_dict.values():
        random.shuffle(indices)
        sampled_indices += indices[0:int(len(indices) * ratio)]
        rest_indices += indices[int(len(indices) * ratio):len(indices)]
    return [Subset(dataset, sampled_indices), Subset(dataset, rest_indices)]

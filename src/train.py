from transformers import BertPreTrainedModel, BertModel, BertForSequenceClassification
from transformers import BertConfig
from torch.optim import AdamW
from transformers import BertTokenizer
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from collections import OrderedDict
import random
import numpy as np
import os
import time
import json
import datetime
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from utils import *

# If there's a GPU available...
if torch.cuda.is_available():
    # Tell PyTorch to use the GPU.
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(CUDA))
    os.environ['CUDA_VISIBLE_DEVICES'] = str(CUDA)
# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

# Tokenize all of the sentences and map the tokens to thier word IDs.


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


sentence_train = json.load(open('data/' + DATASET + '/train_sentence.json', 'r'))
sentence_train_label = json.load(open('data/' + DATASET + '/train_label_id.json', 'r'))
train_dataset = pre_processing(sentence_train, sentence_train_label)

# split training data to labeled set and unlabeled set
labeled_dataset, unlabeled_dataset_total = stratified_sample(train_dataset, LABEL_OF_TRAIN)

unlabeled_dataset = []
for i in range(MATE_EPOCHS):
    unlabeled_dataset_now, unlabeled_dataset_total = stratified_sample(unlabeled_dataset_total,
                                                                       UNLABEL_OF_TRAIN / MATE_EPOCHS)
    unlabeled_dataset.append(unlabeled_dataset_now)

# Create the DataLoaders for our label and unlabel sets.
labeled_dataloader = DataLoader(
    labeled_dataset,  # The training samples.
    sampler=RandomSampler(labeled_dataset),  # Select batches randomly
    batch_size=BATCH_SIZE  # Trains with this batch size.
)
unlabeled_dataloader = []
for i in range(MATE_EPOCHS):
    unlabeled_dataloader_now = DataLoader(
        unlabeled_dataset[i],  # The training samples.
        sampler=RandomSampler(unlabeled_dataset[i]),  # Select batches randomly
        batch_size=BATCH_SIZE  # Trains with this batch size.
    )
    unlabeled_dataloader.append(unlabeled_dataloader_now)

sentence_val = json.load(open('data/' + DATASET + '/test_sentence.json', 'r'))
sentence_val_label = json.load(open('data/' + DATASET + '/test_label_id.json', 'r'))
val_dataset = pre_processing(sentence_val, sentence_val_label)

validation_dataloader = DataLoader(
    val_dataset,  # The validation samples.
    sampler=SequentialSampler(val_dataset),  # Pull out batches sequentially.
    batch_size=BATCH_SIZE  # Evaluate with this batch size.
)

# Load models
modelf1 = RelationClassification.from_pretrained(
    "bert-base-uncased",  # Use the 12-layer BERT model, with an uncased vocab.
    num_labels=NUM_LABELS,  # The number of output labels--2 for binary classification.
    # You can increase this for multi-class tasks.
    output_attentions=False,  # Whether the model returns attentions weights.
    output_hidden_states=False,  # Whether the model returns all hidden-states.
)

modelf1 = nn.DataParallel(modelf1)
modelf1.to(device)

modelg2 = LabelGeneration.from_pretrained(
    "bert-base-uncased",  # Use the 12-layer BERT model, with an uncased vocab.
    num_labels=NUM_LABELS,  # The number of output labels--2 for binary classification.
    # You can increase this for multi-class tasks.
    output_attentions=False,  # Whether the model returns attentions weights.
    output_hidden_states=False,  # Whether the model returns all hidden-states.
)

modelg2 = nn.DataParallel(modelg2)
modelg2.to(device)

optimizer1 = AdamW(modelf1.parameters(),
                   lr=LR,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                   eps=EPS  # args.adam_epsilon  - default is 1e-8.
                   )
optimizer2 = AdamW(modelg2.parameters(),
                   lr=LR,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                   eps=EPS  # args.adam_epsilon  - default is 1e-8.
                   )

total_steps1 = 0
for i in range(MATE_EPOCHS + 1):
    total_steps1 += len(labeled_dataloader)
    for j in range(i):
        total_steps1 += len(unlabeled_dataloader[j])
total_steps1 = total_steps1 * EPOCHS * TOTAL_EPOCHS
total_steps2 = len(labeled_dataloader) * EPOCHS * MATE_EPOCHS * TOTAL_EPOCHS

# Create the learning rate scheduler.
scheduler1 = get_linear_schedule_with_warmup(optimizer1,
                                             num_warmup_steps=0,
                                             num_training_steps=total_steps1)
scheduler2 = get_linear_schedule_with_warmup(optimizer2,
                                             num_warmup_steps=0,
                                             num_training_steps=total_steps2)

# Set the seed value all over the place to make this reproducible.
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# validation accuracy, and timings.
training_stats = []
# Measure the total training time for the whole run.
total_t0 = time.time()


print("training start...")
cnt = 0

# count how many rounds the whole big model has to train
# a round means all unlabel data are labeled
for total_epoch in range(TOTAL_EPOCHS):

    # Add an inner loop to judge the entire unlabeled data set finished
    train_dataloader = labeled_dataloader

    for mate_epoch in range(MATE_EPOCHS):

        # -------train f1---------

        for epoch_i in range(0, EPOCHS):
            print('--- Epoch {:} / {:} ---'.format(epoch_i + 1, EPOCHS))
            # Measure how long the training epoch takes.
            t0 = time.time()

            # Reset the total loss for this epoch.
            total_train_loss = 0
            # Put the model into training mode
            modelf1.train()

            for step, batch in enumerate(train_dataloader):
                # Progress update every 40 batches.
                if step % 40 == 0 and not step == 0:
                    # Calculate elapsed time in minutes.
                    elapsed = format_time(time.time() - t0)

                # Unpack this training batch from our dataloader.
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)
                b_e1_pos = batch[3].to(device)
                b_e2_pos = batch[4].to(device)
                b_w = batch[5].to(device)

                modelf1.zero_grad()

                # Perform a forward pass (evaluate the model on this training batch)
                loss, logits, _ = modelf1(b_input_ids,
                                          token_type_ids=None,
                                          attention_mask=b_input_mask,
                                          labels=b_labels,
                                          e1_pos=b_e1_pos,
                                          e2_pos=b_e2_pos,
                                          w=b_w)

                # Accumulate the training loss over all of the batches
                total_train_loss += loss.sum().item()
                # Perform a backward pass to calculate the gradients.
                loss.sum().backward()
                # Clip the norm of the gradients to 1.0 to prevent exploding gradients problem.
                torch.nn.utils.clip_grad_norm_(modelf1.parameters(), 1.0)
                # Update parameters and take a step using the computed gradient
                optimizer1.step()
                # Update the learning rate.
                scheduler1.step()

            # Calculate the average loss over all of the batches.
            avg_train_loss = total_train_loss / len(train_dataloader)
            # Measure how long this epoch took.
            training_time = format_time(time.time() - t0)

            print(" f1 Average training loss: {0:.4f}".format(avg_train_loss))
            # print(" f1 Training epcoh took: {:}".format(training_time))

            # --------f1 validation-----------

            print("validation start...")
            t0 = time.time()

            # Put the model in evaluation mode--the dropout layers behave differently during evaluation.
            modelf1.eval()

            # Tracking variables
            total_eval_accuracy = 0
            total_eval_loss = 0
            nb_eval_steps = 0
            all_prediction = np.array([])
            all_ground_truth = np.array([])

            # Evaluate data for one epoch
            for batch in validation_dataloader:
                # Unpack this training batch from our dataloader.
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)
                b_e1_pos = batch[3].to(device)
                b_e2_pos = batch[4].to(device)
                b_w = batch[5].to(device)

                with torch.no_grad():
                    # Forward pass, calculate logit predictions.
                    (loss, logits, _) = modelf1(b_input_ids,
                                                token_type_ids=None,
                                                attention_mask=b_input_mask,
                                                labels=b_labels,
                                                e1_pos=b_e1_pos,
                                                e2_pos=b_e2_pos,
                                                w=b_w)

                # Accumulate the validation loss.
                total_eval_loss += loss.sum().item()
                # Move logits and labels to CPU
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()

                pred_flat = np.argmax(logits, axis=1).flatten()
                labels_flat = label_ids.flatten()
                all_prediction = np.concatenate((all_prediction, pred_flat), axis=None)
                all_ground_truth = np.concatenate((all_ground_truth, labels_flat), axis=None)

            # Calculate the average loss over all of the batches.
            avg_val_loss = total_eval_loss / len(validation_dataloader)

            # Measure how long the validation run took.
            validation_time = format_time(time.time() - t0)

            score(all_ground_truth, all_prediction)

        # -------f1 predict and g2 update---------

        print("f1 predict and g2 update start...")

        for epoch_i in range(0, EPOCHS):
            print('--- Epoch {:} / {:} ---'.format(epoch_i + 1, EPOCHS))
            # Measure how long the training epoch takes.
            t0 = time.time()

            # Reset the total loss for this epoch.
            total_train_loss = 0

            modelg2.train()
            modelf1.eval()

            for step, batch in enumerate(labeled_dataloader):
                # Progress update every 40 batches.
                if step % 40 == 0 and not step == 0:
                    # Calculate elapsed time in minutes.
                    elapsed = format_time(time.time() - t0)
                    # Report progress.

                # Unpack this training batch from our dataloader.
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)
                b_e1_pos = batch[3].to(device)
                b_e2_pos = batch[4].to(device)
                b_w = batch[5].to(device)

                # f1 predict
                modelf1.zero_grad()
                # with torch.no_grad():
                # Forward pass, calculate logit predictions.
                loss1, logits1, _ = modelf1(b_input_ids,
                                            token_type_ids=None,
                                            attention_mask=b_input_mask,
                                            labels=b_labels,
                                            e1_pos=b_e1_pos,
                                            e2_pos=b_e2_pos,
                                            w=b_w)

                # g2 train
                modelg2.zero_grad()
                # Perform a forward pass
                loss2, logits2, _ = modelg2(b_input_ids,
                                            token_type_ids=None,
                                            attention_mask=b_input_mask,
                                            labels=b_labels,
                                            e1_pos=b_e1_pos,
                                            e2_pos=b_e2_pos,
                                            w=b_w)

                # Perform a backward pass to calculate the gradients.
                # loss1.sum().backward()
                (loss1.sum() + 0.4 * loss2.sum()).backward()
                # Clip the norm of the gradients to 1.0 to prevent exploding gradients problem.
                torch.nn.utils.clip_grad_norm_(modelg2.parameters(), 1.0)
                # Update parameters and take a step using the computed gradient
                optimizer2.step()
                # Update the learning rate.
                scheduler2.step()

                # Accumulate the training loss over all of the batches
                total_train_loss += loss2.sum().item()

            # Calculate the average loss over all of the batches.
            avg_train_loss = total_train_loss / len(labeled_dataloader)
            # Measure how long this epoch took.
            training_time = format_time(time.time() - t0)

            print(" g2 Average training loss by g2: {0:.4f}".format(avg_train_loss))
            # print(" g2 Training epcoh took: {:}".format(training_time))

        # -------g2 generate pseudo label---------

        print("g2 generate pseudo label")

        modelg2.eval()
        input_ids = []
        input_mask = []
        # labels = []
        gold_labels = []
        e1_pos = []
        e2_pos = []
        w = []
        all_logits = []
        all_prediction = np.array([])
        all_ground_truth = np.array([])
        all_weights = np.array([])
        all_pseudo_prediction = np.array([])
        all_pseudo_ground_truth = np.array([])
        all_pseudo_weights = np.array([])
        # Evaluate data for one epoch
        for batch in unlabeled_dataloader[mate_epoch]:  # unlabeled_dataloader
            # Unpack this training batch from our dataloader.
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            b_e1_pos = batch[3].to(device)
            b_e2_pos = batch[4].to(device)
            b_w = batch[5].to(device)

            with torch.no_grad():
                # Forward pass, calculate logit predictions.
                (loss, logits, _) = modelg2(b_input_ids,
                                            token_type_ids=None,
                                            attention_mask=b_input_mask,
                                            labels=b_labels,
                                            e1_pos=b_e1_pos,
                                            e2_pos=b_e2_pos,
                                            w=b_w)

            logits = logits.detach()
            all_logits.append(logits)
            input_ids.append(b_input_ids)
            input_mask.append(b_input_mask)
            gold_labels.append(b_labels)
            e1_pos.append(b_e1_pos)
            e2_pos.append(b_e2_pos)
            w.append(b_w)

        logits = torch.cat(all_logits, dim=0)
        probs = F.softmax(logits, dim=1)
        input_ids = torch.cat(input_ids, dim=0)
        input_mask = torch.cat(input_mask, dim=0)
        gold_labels = torch.cat(gold_labels, dim=0)
        e1_pos = torch.cat(e1_pos, dim=0)
        e2_pos = torch.cat(e2_pos, dim=0)
        label_weights, labels = torch.max(probs, dim=1)

        # evaluate generate label quality
        logits = logits.detach().cpu().numpy()
        label_ids = gold_labels.to('cpu').numpy()
        pred_weights = label_weights.to('cpu').numpy()
        pred_flat = np.argmax(logits, axis=1).flatten()
        labels_flat = label_ids.flatten()
        pred_weights = pred_weights.flatten()
        all_prediction = np.concatenate((all_prediction, pred_flat), axis=None)
        all_ground_truth = np.concatenate((all_ground_truth, labels_flat), axis=None)
        all_weights = np.concatenate((all_weights, pred_weights), axis=None)

        # Pseudo Label Selection, top Z%
        sort = torch.argsort(label_weights, descending=True)
        sort = sort[0:int(len(sort) * Z_RATIO)]
        input_ids = input_ids[sort]
        input_mask = input_mask[sort]
        pseudo_labels = labels[sort]
        e1_pos = e1_pos[sort]
        e2_pos = e2_pos[sort]
        w = label_weights[sort]
        pseudo_gold_labels = gold_labels[sort]

        # evaluate pseudo label quality
        pred = pseudo_labels.to('cpu').numpy()
        label_ids = pseudo_gold_labels.to('cpu').numpy()
        weights = w.to('cpu').numpy()
        pred_flat = pred.flatten()
        labels_flat = label_ids.flatten()
        weights_flat = weights.flatten()
        all_pseudo_prediction = np.concatenate((all_pseudo_prediction, pred_flat), axis=None)
        all_pseudo_ground_truth = np.concatenate((all_pseudo_ground_truth, labels_flat), axis=None)
        all_pseudo_weights = np.concatenate((all_pseudo_weights, weights_flat), axis=None)

        #  evaluate generate label quality
        np_sort = np.argsort(all_weights)
        print("  Generate F1 score")
        score(all_ground_truth, all_prediction)
        np.save(LOG_DIR + '/' + LOG_DIR + '_all_prediction{}'.format(cnt), all_prediction[np_sort])
        np.save(LOG_DIR + '/' + LOG_DIR + '_all_label{}'.format(cnt), all_ground_truth[np_sort])
        np.save(LOG_DIR + '/' + LOG_DIR + '_all_weights{}'.format(cnt), all_weights[np_sort])
        #  evaluate pseudo label quality
        print("  Pseudo F1 score")
        score(all_pseudo_ground_truth, all_pseudo_prediction)
        np.save(LOG_DIR + '/' + LOG_DIR + '_all_pseudo_prediction{}'.format(cnt), all_pseudo_prediction)
        np.save(LOG_DIR + '/' + LOG_DIR + '_all_pseudo_label{}'.format(cnt), all_pseudo_ground_truth)
        np.save(LOG_DIR + '/' + LOG_DIR + '_all_pseudo_weights{}'.format(cnt), all_pseudo_weights)
        cnt += 1

        # update training data
        train_add_dataset = train_dataloader.dataset + \
            TensorDataset(input_ids, input_mask, pseudo_labels, e1_pos, e2_pos, w)
        train_dataloader = DataLoader(
            train_add_dataset,  # The training samples.
            sampler=RandomSampler(train_add_dataset),  # Select batches randomly
            batch_size=BATCH_SIZE  # Trains with this batch size.
        )

    # train f1 with all data
    for epoch_i in range(0, EPOCHS):
        print('--- Epoch {:} / {:} ---'.format(epoch_i + 1, EPOCHS))
        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_train_loss = 0
        # Put the model into training mode
        modelf1.train()

        for step, batch in enumerate(train_dataloader):
            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)

            # Unpack this training batch from our dataloader.
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            b_e1_pos = batch[3].to(device)
            b_e2_pos = batch[4].to(device)
            b_w = batch[5].to(device)

            modelf1.zero_grad()

            # Perform a forward pass (evaluate the model on this training batch)
            loss, logits, _ = modelf1(b_input_ids,
                                      token_type_ids=None,
                                      attention_mask=b_input_mask,
                                      labels=b_labels,
                                      e1_pos=b_e1_pos,
                                      e2_pos=b_e2_pos,
                                      w=b_w)

            # Accumulate the training loss over all of the batches
            total_train_loss += loss.sum().item()
            # Perform a backward pass to calculate the gradients.
            loss.sum().backward()
            # Clip the norm of the gradients to 1.0 to prevent exploding gradients problem.
            torch.nn.utils.clip_grad_norm_(modelf1.parameters(), 1.0)
            # Update parameters and take a step using the computed gradient
            optimizer1.step()
            # Update the learning rate.
            scheduler1.step()

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)
        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        print(" f1 Average training loss: {0:.4f}".format(avg_train_loss))
        # print(" f1 Training epcoh took: {:}".format(training_time))

     # final f1 validation
    print("final validation start...")
    t0 = time.time()

    # Put the model in evaluation mode--the dropout layers behave differently during evaluation.
    modelf1.eval()

    # Tracking variables
    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0

    all_prediction = np.array([])
    all_ground_truth = np.array([])

    # Evaluate data for one epoch
    for batch in validation_dataloader:
        # Unpack this training batch from our dataloader.
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        b_e1_pos = batch[3].to(device)
        b_e2_pos = batch[4].to(device)
        b_w = batch[5].to(device)

        with torch.no_grad():
            # Forward pass, calculate logit predictions.
            (loss, logits, _) = modelf1(b_input_ids,
                                        token_type_ids=None,
                                        attention_mask=b_input_mask,
                                        labels=b_labels,
                                        e1_pos=b_e1_pos,
                                        e2_pos=b_e2_pos,
                                        w=b_w)

        # Accumulate the validation loss.
        total_eval_loss += loss.sum().item()
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        pred_flat = np.argmax(logits, axis=1).flatten()
        labels_flat = label_ids.flatten()
        all_prediction = np.concatenate((all_prediction, pred_flat), axis=None)
        all_ground_truth = np.concatenate((all_ground_truth, labels_flat), axis=None)

    # Calculate the average loss over all of the batches.
    avg_val_loss = total_eval_loss / len(validation_dataloader)

    validation_time = format_time(time.time() - t0)
    non_zero_idx = (all_ground_truth != 0)
    validation_acc = np.sum(all_prediction[non_zero_idx] ==
                            all_ground_truth[non_zero_idx]) / len(all_ground_truth[non_zero_idx])
    validation_f1_score = f1_score(all_ground_truth[non_zero_idx], all_prediction[non_zero_idx], average="micro")

    score(all_ground_truth, all_prediction)

    torch.save({'epoch': total_epoch,
                'model_state_dict': modelf1.state_dict(),
                'optimizer_state_dict': optimizer1.state_dict()},
               SAVE_DIR)

# ----------------------training complete-----------------------

print("Training complete!")
print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))

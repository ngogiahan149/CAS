import os
import re
from datetime import datetime
import random
import numpy as np
import pandas as pd
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
import wandb, gc
import random
from transformers import AdamW
from transformers.optimization import get_linear_schedule_with_warmup
import evaluate
import logging
import datasets
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import get_scheduler, default_data_collator
import math
from tqdm.auto import tqdm
import argparse
import torch
import torch.nn as nn
import sys
# model
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
config = {
    'experiment_name': 'transformers',
    'task': 'train',
    'make_dict': True,
    'data_preprocessing': True,

    'train' : 'train_transformers.csv',
    'test' : 'test_transformers.csv',
    'none_other_train': 'train_transformers_new.csv',
    'none_other_test': 'test_transformers_new.csv',
    'directory': './model_identifier_nodash/',
    'model_result': 'result_finetune.csv',
    'training':{
        'num_epochs': 30,
        'learning_rate': 1e-4,
        'optimizer': 'adam',
        'num_round_per_epoch' : 30000,
    },
    
        
    'model':{
        'name': 'transformers',
        'embed_size': 1030,
        'batch_size': 10,
        'encoder':{
            'hidden_size': 150,
            'num_layers': 1,
            'bidirectional': True,
            'dropout': 0.1,
        },
    },   
    
    'result':{
        'filename':'result.txt',
        'filepath':'res/',
    }
}


def calculate_amount(pred, label):
    TP_count, TN_count, FP_count, FN_count, gold_pos = 0, 0, 0, 0, 0
    for i in range(len(pred)):
        if label[i]!= 0:
            gold_pos += 1
            if (pred[i] == label[i]):
                TP_count += 1
            elif(pred[i]!=0 and pred[i] != label[i]):
                FP_count +=1
            else:
                FN_count += 1

        elif label[i] == 0:
            if pred[i] != 0:
                FP_count += 1
            else:
                TN_count +=1
    return TP_count, TN_count, FP_count, FN_count, gold_pos
def compute_metrics(TP_count, TN_count, FP_count, FN_count, gold_pos):
    precision = TP_count/(TP_count+FP_count) if (TP_count+FP_count) != 0. else 0.
    recall = TP_count/gold_pos if gold_pos != 0. else 0.
    F1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0. else 0.
    accuracy = (TP_count+TN_count)/(TP_count + TN_count + FP_count + FN_count)
    return precision, recall, F1_score, accuracy


# +
def plot_confusion_matrix(pred_labels, gt_labels, classes):
    num_classes = len(classes)
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for gt, pred in zip(gt_labels, pred_labels):
        confusion_matrix[int(gt), int(pred)] += 1

    # Create a heatmap plot of the confusion matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(confusion_matrix[::-1], cmap="Blues")
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, classes, rotation=50)
    plt.yticks(tick_marks, classes[::-1])
    
    # Add labels to each cell
    thresh = confusion_matrix.max() / 2
    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, confusion_matrix[::-1][i, j], ha="center", va="center",
                     color="white" if confusion_matrix[::-1][i, j] > thresh else "black")

    plt.xlabel("Predicted Labels")
    plt.ylabel("Ground Truth Labels")
    plt.tight_layout()
    return plt


def transform_labels_stage5(label):
    label = label['labels']
    num = 0
    # Remember to change this for each stage
    if label == 'None':
        num = 0
    elif label == 'No':
        num = 1
    elif label == 'Novel':
        num = 2
    return {'labels': num}
def transform_labels_stage6(label):
    label = label['labels']
    num = 0
    # Remember to change this for each stage
    if label == 'No':
        num = 0
    elif label == 'Novel':
        num = 1
    return {'labels': num}

def transform_labels_stage4(label):
    label = label['labels']
    num = 0
    # Remember to change this for each stage
    if label == 'None':
        num = 0
    elif label == 'Association':
        num = 1
    elif label == 'Positive_Correlation':
        num = 2
    elif label == 'Negative_Correlation':
        num = 3
    if label == 'Bind':
        num = 4
    elif label == 'Cotreatment':
        num = 5
    elif label == 'Comparison':
        num = 6
    elif label == 'Drug_Interaction':
        num = 7
    elif label == 'Conversion':
        num = 8
    return {'labels': num}
def transform_labels_stage7(label):
    label_mapping = {}
    label = label['labels']
    # Generate label-to-number mappings automatically
    df = pd.read_csv('dataset/entity_identifier.csv', keep_default_na = False)
    df['identifier'] = df['identifier'].drop_duplicates()
    list_labels = df['identifier']  # Add your labels here
    for idx, label_name in enumerate(list_labels):
        label_mapping[label_name] = idx
    
    num = label_mapping.get(label, -1)
#     print(num)
    return {'labels': num}
def train(checkpoint, file, file_test, classes, stage):
     # start a new wandb run to track this script
    print(sys.path)
    wandb.init(
    project="BioCreative-Transformer",
    config={
    "learning_rate": 1e-5,
    "architecture": "Transformer",
    "dataset": "BIORED",
    "epochs": 50,
    })
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    logger = logging.getLogger(__name__)
    # Tokenize the texts
    special_tokens = {'additional_special_tokens': ["@ChemicalEntitySrc$", 
                           "@ChemicalEntityTgt$", 
                           "@DiseaseOrPhenotypicFeatureSrc$", 
                           "@DiseaseOrPhenotypicFeatureTgt$", 
                           "@GeneOrGeneProductSrc$", 
                           "@GeneOrGeneProductTgt$",
                            "@/ChemicalEntitySrc$", 
                           "@/ChemicalEntityTgt$", 
                           "@/DiseaseOrPhenotypicFeatureSrc$", 
                           "@/DiseaseOrPhenotypicFeatureTgt$", 
                           "@/GeneOrGeneProductSrc$", 
                           "@/GeneOrGeneProductTgt$"]}
    
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint,
        num_labels = len(classes)).to(device)
    
    tokenizer.add_special_tokens(special_tokens)
    tokenizer.save_pretrained(config['directory'])
    # Define the dataset
    df = pd.read_csv(file, keep_default_na=False)
    features = datasets.Features(
                {'labels': datasets.Value('string'),
                 'text': datasets.Value('string')})
    data_dict = {}
#     data_dict['labels']  = [label for label in df.iloc[:,6]]
    data_dict['labels']  = [label for label in df['ground truth']]
    data_dict['text']    = [text for text in df['Document']]
    data_dataset = datasets.Dataset.from_dict(data_dict, features=features)
    data_dataset = data_dataset.map(
            lambda example: tokenizer.batch_encode_plus(
                example['text'], 
                truncation     = True, 
                max_length     = 512, 
                padding        = "max_length",
                stride         = 128
            ),
            batched=True,
        )
    
    # Get the transform_labels function based on different stage
    transform_labels = globals().get(f"transform_labels_stage{stage}")
    
    data_dataset = data_dataset.map(transform_labels, remove_columns = ['text', 'labels'])
    data_dataset.set_format = data_dataset.set_format("torch")
#     train_size = int(0.85 * len(data_dataset))
#     test_size = len(data_dataset) - train_size
#     train_dataset_split, valid_dataset_split = torch.utils.data.random_split(data_dataset, [train_size, test_size])
    
    # Define the valid dataset
    df_valid = pd.read_csv(file_test, keep_default_na=False)
    data_dict_valid = {}
    data_dict_valid['labels']  = [label for label in df_valid['ground truth']]
    #data_dict['text'] = []
    data_dict_valid['text']    = [text for text in df_valid['Document']]
    data_dataset_valid = datasets.Dataset.from_dict(data_dict_valid, features=features)
    data_dataset_valid = data_dataset_valid.map(
            lambda example: tokenizer.batch_encode_plus(
                example['text'], 
                truncation     = True, 
                max_length     = 512, 
                padding        = "max_length",
                stride         = 128
            ),
            batched=True,
        )
    
    data_dataset_valid = data_dataset_valid.map(transform_labels, remove_columns = ['text', 'labels'])
    data_dataset_valid.set_format = data_dataset_valid.set_format("torch")
    valid_size = int(0.5 * len(data_dataset_valid))
    test_size = len(data_dataset_valid) - valid_size
    valid_dataset_split, test_dataset_split = torch.utils.data.random_split(data_dataset_valid, [valid_size, test_size])
    # collate_fn to bring to GPU
    train_dataloader = DataLoader(data_dataset, batch_size=config['model']['batch_size'], collate_fn=default_data_collator, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset_split, batch_size=config['model']['batch_size'], collate_fn=default_data_collator, shuffle=True)
    # Define optimizer, learning rate scheduler
    optimizer = AdamW(model.parameters(), lr = 1e-5)
    num_training_steps = config['training']['num_epochs'] * len(train_dataloader)
    
    # Default learning rate scheduler from Trainer
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=2, num_training_steps=num_training_steps
    )
    
    metric = evaluate.load("accuracy")
    from tqdm.auto import tqdm

    
    # Training process
    train_loss, valid_loss = [], []
    min_loss = 10.0
    for epoch in range(config['training']['num_epochs']):
        model.train()
        if config['training']['num_round_per_epoch'] <= len(train_dataloader):
            progress_bar_train = tqdm(range(config['training']['num_round_per_epoch']))
        else:
            progress_bar_train = tqdm(range(len(train_dataloader)))
        for step, batch in enumerate(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            train_loss.append(loss.cpu().detach().item())
            torch.cuda.empty_cache()
            del loss
            progress_bar_train.update(1)
            if step == config['training']['num_round_per_epoch']:
                break
        wandb.log({"train loss": np.mean(train_loss)})
        
        # Record predictions and ground truth for drawing confusion matrix
        pred_label_valid, true_label_valid = [], []
        TP_sum_valid,TN_sum_valid, FP_sum_valid, FN_sum_valid, gold_pos_valid = 0, 0, 0, 0, 0
        model.eval()
        best_acc = 0
        progress_bar_valid = tqdm(range(len(valid_dataloader)))
        for batch in valid_dataloader:
            with torch.no_grad():
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
            logits = outputs.logits.detach().cpu()
            predictions = torch.argmax(logits, dim=-1)
            
            
            metric.add_batch(predictions=predictions, references=batch["labels"])
            valid_loss.append(outputs.loss.cpu().detach().item())
            
            true_labels = batch["labels"].cpu().detach().numpy()
            pred_labels = predictions.numpy()
            
            pred_label_valid = np.append(pred_label_valid, pred_labels)
            true_label_valid = np.append(true_label_valid, true_labels)
            
            TP, TN, FP, FN, gold_pos = calculate_amount(pred_labels, true_labels)
            TP_sum_valid += TP
            TN_sum_valid += TN
            FP_sum_valid += FP
            FN_sum_valid += FN
            gold_pos_valid += gold_pos
            progress_bar_valid.update(1)
            torch.cuda.empty_cache()
        # Calculate eval metric
        eval_metric = metric.compute()
        logger.info(f"epoch {epoch}: {eval_metric}")
        
        # Accuracy for 1 epoch
        valid_precision, valid_recall, valid_f1, valid_accuracy = compute_metrics(TP_sum_valid, TN_sum_valid, FP_sum_valid, FN_sum_valid, gold_pos_valid)
        
        print('Epoch: {}, Train loss: {}, Valid loss: {}, F1: {} Learning rate: {}\n'.format(epoch, np.mean(train_loss), np.mean(valid_loss), valid_f1, optimizer.param_groups[0]["lr"]))
        
        # log metrics to wandb
        wandb.log({"valid loss": np.mean(valid_loss)})
        wandb.log({"validation accuracy ": valid_accuracy})
        wandb.log({"validation precision ": valid_precision})
        wandb.log({"validation f1 ": valid_f1})
        wandb.log({"validation recall ": valid_recall})
        
        
        # Log confusion matrix to wandb
        plt_valid = plot_confusion_matrix(pred_label_valid, true_label_valid, classes)
        wandb.log({"Epoch " + str(epoch) + " validation": wandb.Image(plt_valid)})
        
        # Close plt to avoid overlapping pictures
        plt_valid.clf()
        plt_valid.cla()
        plt_valid.close()
        plt_valid.close()
        
        # Save model with highest f1
        if best_acc <= valid_f1:
            best_acc = valid_f1
            model.save_pretrained(config['directory'])
        if np.mean(valid_loss) <= min_loss:
            min_loss = np.mean(valid_loss)
#         if np.mean(valid_loss) - min_loss >= 0.05:
#             break
def train_novelty(checkpoint, file, file_test, classes, stage):
     # start a new wandb run to track this script
    wandb.init(
    project="BioCreative-Transformer",
    config={
    "learning_rate": 1e-5,
    "architecture": "Transformer",
    "dataset": "BIORED",
    "epochs": 50,
    })
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    logger = logging.getLogger(__name__)
    # Tokenize the texts
    special_tokens = {'additional_special_tokens': ["@ChemicalEntitySrc$", 
                           "@ChemicalEntityTgt$", 
                           "@DiseaseOrPhenotypicFeatureSrc$", 
                           "@DiseaseOrPhenotypicFeatureTgt$", 
                           "@GeneOrGeneProductSrc$", 
                           "@GeneOrGeneProductTgt$",
                            "@/ChemicalEntitySrc$", 
                           "@/ChemicalEntityTgt$", 
                           "@/DiseaseOrPhenotypicFeatureSrc$", 
                           "@/DiseaseOrPhenotypicFeatureTgt$", 
                           "@/GeneOrGeneProductSrc$", 
                           "@/GeneOrGeneProductTgt$"]}
    
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint,
        num_labels = len(classes)).to(device)
    
    tokenizer.add_special_tokens(special_tokens)
    tokenizer.save_pretrained(config['directory'])
    # Define the dataset
    df = pd.read_csv(file, keep_default_na=False)
#     df = df[df['Novelty'] != 'None']
    print(df)
    features = datasets.Features(
                {'labels': datasets.Value('string'),
                 'text': datasets.Value('string')})
    data_dict = {}
#     data_dict['labels']  = [label for label in df.iloc[:,6]]
    data_dict['labels']  = [label for label in df['Novelty']]
    data_dict['text']    = [text for text in df['Document']]
    data_dataset = datasets.Dataset.from_dict(data_dict, features=features)
    data_dataset = data_dataset.map(
            lambda example: tokenizer.batch_encode_plus(
                example['text'], 
                truncation     = True, 
                max_length     = 512, 
                padding        = "max_length",
                stride         = 128
            ),
            batched=True,
        )
    
    # Get the transform_labels function based on different stage
    transform_labels = globals().get(f"transform_labels_stage{stage}")
    
    data_dataset = data_dataset.map(transform_labels, remove_columns = ['text', 'labels'])
    data_dataset.set_format = data_dataset.set_format("torch")
#     train_size = int(0.85 * len(data_dataset))
#     test_size = len(data_dataset) - train_size
#     train_dataset_split, valid_dataset_split = torch.utils.data.random_split(data_dataset, [train_size, test_size])
    
    # Define the valid dataset
    df_valid = pd.read_csv(file_test, keep_default_na=False)
#     df_valid = df_valid[df_valid['Novelty'] != 'None']
    print(df_valid)
    data_dict_valid = {}
    data_dict_valid['labels']  = [label for label in df_valid['Novelty']]
    #data_dict['text'] = []
    data_dict_valid['text']    = [text for text in df_valid['Document']]
    data_dataset_valid = datasets.Dataset.from_dict(data_dict_valid, features=features)
    data_dataset_valid = data_dataset_valid.map(
            lambda example: tokenizer.batch_encode_plus(
                example['text'], 
                truncation     = True, 
                max_length     = 512, 
                padding        = "max_length",
                stride         = 128
            ),
            batched=True,
        )
    
    data_dataset_valid = data_dataset_valid.map(transform_labels, remove_columns = ['text', 'labels'])
    data_dataset_valid.set_format = data_dataset_valid.set_format("torch")
    valid_size = int(0.5 * len(data_dataset_valid))
    test_size = len(data_dataset_valid) - valid_size
    valid_dataset_split, test_dataset_split = torch.utils.data.random_split(data_dataset_valid, [valid_size, test_size])
    # collate_fn to bring to GPU
    train_dataloader = DataLoader(data_dataset, batch_size=config['model']['batch_size'], collate_fn=default_data_collator, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset_split, batch_size=config['model']['batch_size'], collate_fn=default_data_collator, shuffle=True)
    # Define optimizer, learning rate scheduler
    optimizer = AdamW(model.parameters(), lr = 1e-5)
    num_training_steps = config['training']['num_epochs'] * len(train_dataloader)
    
    # Default learning rate scheduler from Trainer
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=2, num_training_steps=num_training_steps
    )
    
    metric = evaluate.load("accuracy")
    from tqdm.auto import tqdm

    
    # Training process
    train_loss, valid_loss = [], []
    min_loss = 10.0
    for epoch in range(config['training']['num_epochs']):
        model.train()
        if config['training']['num_round_per_epoch'] <= len(train_dataloader):
            progress_bar_train = tqdm(range(config['training']['num_round_per_epoch']))
        else:
            progress_bar_train = tqdm(range(len(train_dataloader)))
        for step, batch in enumerate(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            train_loss.append(loss.cpu().detach().item())
            torch.cuda.empty_cache()
            del loss
            progress_bar_train.update(1)
            if step == config['training']['num_round_per_epoch']:
                break
        wandb.log({"train loss": np.mean(train_loss)})
        
        # Record predictions and ground truth for drawing confusion matrix
        pred_label_valid, true_label_valid = [], []
        TP_sum_valid,TN_sum_valid, FP_sum_valid, FN_sum_valid, gold_pos_valid = 0, 0, 0, 0, 0
        model.eval()
        best_acc = 0
        progress_bar_valid = tqdm(range(len(valid_dataloader)))
        for batch in valid_dataloader:
            with torch.no_grad():
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
            logits = outputs.logits.detach().cpu()
            predictions = torch.argmax(logits, dim=-1)
            
            
            metric.add_batch(predictions=predictions, references=batch["labels"])
            valid_loss.append(outputs.loss.cpu().detach().item())
            
            true_labels = batch["labels"].cpu().detach().numpy()
            pred_labels = predictions.numpy()
            
            pred_label_valid = np.append(pred_label_valid, pred_labels)
            true_label_valid = np.append(true_label_valid, true_labels)
            
            TP, TN, FP, FN, gold_pos = calculate_amount(pred_labels, true_labels)
            TP_sum_valid += TP
            TN_sum_valid += TN
            FP_sum_valid += FP
            FN_sum_valid += FN
            gold_pos_valid += gold_pos
            progress_bar_valid.update(1)
            torch.cuda.empty_cache()
        # Calculate eval metric
        eval_metric = metric.compute()
        logger.info(f"epoch {epoch}: {eval_metric}")
        
        # Accuracy for 1 epoch
        valid_precision, valid_recall, valid_f1, valid_accuracy = compute_metrics(TP_sum_valid, TN_sum_valid, FP_sum_valid, FN_sum_valid, gold_pos_valid)
        
        print('Epoch: {}, Train loss: {}, Valid loss: {}, F1: {} Learning rate: {}\n'.format(epoch, np.mean(train_loss), np.mean(valid_loss), valid_f1, optimizer.param_groups[0]["lr"]))
        
        # log metrics to wandb
        wandb.log({"valid loss": np.mean(valid_loss)})
        wandb.log({"validation accuracy ": valid_accuracy})
        wandb.log({"validation precision ": valid_precision})
        wandb.log({"validation f1 ": valid_f1})
        wandb.log({"validation recall ": valid_recall})
        
        
        # Log confusion matrix to wandb
        plt_valid = plot_confusion_matrix(pred_label_valid, true_label_valid, classes)
        wandb.log({"Epoch " + str(epoch) + " validation": wandb.Image(plt_valid)})
        
        # Close plt to avoid overlapping pictures
        plt_valid.clf()
        plt_valid.cla()
        plt_valid.close()
        plt_valid.close()
        
        # Save model with highest f1
        if best_acc <= valid_f1:
            best_acc = valid_f1
            model.save_pretrained(config['directory'])
        if np.mean(valid_loss) <= min_loss:
            min_loss = np.mean(valid_loss)
#         if np.mean(valid_loss) - min_loss >= 0.05:
#             break
def predict_relation_no_novel(path_relation, path_novel, df_eval, relation_class, novel_class, relation_stage, novel_stage):
    tokenizer = AutoTokenizer.from_pretrained(path_relation)
    model_relation = AutoModelForSequenceClassification.from_pretrained(
        path_relation,
        num_labels = len(relation_class)).to(device)
    model_novel = AutoModelForSequenceClassification.from_pretrained(
        path_novel,
        num_labels = len(novel_class)).to(device)
    # Load test dataset
    # df_eval = pd.read_csv(filedata, sep = ',')
    features = datasets.Features(
                {'labels': datasets.Value('string'),
                 'text': datasets.Value('string')})
    data_dict = {}
    data_dict['labels']  = [label for label in df_eval['ground truth']]
    data_dict['text']    = [text for text in df_eval['Document']]
    data_dataset = datasets.Dataset.from_dict(data_dict, features=features)
    data_dataset = data_dataset.map(
            lambda example: tokenizer.batch_encode_plus(
                example['text'], 
                truncation     = True, 
                max_length     = 512, 
                padding        = "max_length",
                stride         = 128
            ),
            batched=True,
        )
    transform_labels = globals().get(f"transform_labels_stage{relation_stage}")
    data_dataset = data_dataset.map(transform_labels, remove_columns = ['text', 'labels'])
    data_dataset.set_format = data_dataset.set_format("torch")
    test_dataloader = DataLoader(data_dataset, batch_size=config['model']['batch_size'])
    
    # Call test loader for novelty
    data_dict_novel = {}
    data_dict_novel['labels']  = [label for label in df_eval['Novelty']]
    data_dict_novel['text']    = [text for text in df_eval['Document']]
    data_dataset_novel = datasets.Dataset.from_dict(data_dict_novel, features=features)
    data_dataset_novel = data_dataset_novel.map(
            lambda example: tokenizer.batch_encode_plus(
                example['text'], 
                truncation     = True, 
                max_length     = 512, 
                padding        = "max_length",
                stride         = 128
            ),
            batched=True,
        )
    transform_labels_novel = globals().get(f"transform_labels_stage{novel_stage}")
    data_dataset_novel = data_dataset_novel.map(transform_labels_novel, remove_columns = ['text', 'labels'])
    data_dataset_novel.set_format = data_dataset_novel.set_format("torch")
    test_dataloader_novel = DataLoader(data_dataset_novel, batch_size=config['model']['batch_size'])
    
    model_relation.eval()
    model_relation.to( device )
    model_novel.eval()
    model_novel.to( device )
    index = 0
    metric = evaluate.load("accuracy")
    column_names = [ 'pmid', 'identifier_1', 'identifier_2', 'Type1', 'Type2', 'Document', 'ground truth', 'Novelty', 'relation_predicted', 'novel_predicted']
    df_result = pd.DataFrame( columns = column_names )
    TP_count, TN_count, FP_count, FN_count, gold_pos_sum = 0, 0, 0, 0, 0
    TP_count_novel, TN_count_novel, FP_count_novel, FN_count_novel, gold_pos_sum_novel = 0, 0, 0, 0, 0
    eval_bar = tqdm(range(len(test_dataloader)))
    for step, (batch, batch_novel) in enumerate(zip(test_dataloader, test_dataloader_novel)):
        with torch.no_grad():
            batch = {k: v.to(device) for k, v in batch.items()}
            batch_novel = {k: v.to(device) for k, v in batch_novel.items()}
            outputs_relation = model_relation(**batch)
            outputs_novel = model_novel(**batch_novel)
        logits_relation = outputs_relation.logits.cpu().detach()
        predictions_relation = torch.argmax(logits_relation, dim=-1)
        
        logits_novel = outputs_novel.logits.cpu().detach()
        predictions_novel = torch.argmax(logits_novel, dim=-1)
        metric.add_batch(predictions=predictions_relation, references=batch["labels"])
        # Relation
        TP, TN, FP, FN, gold_pos = calculate_amount(predictions_relation, batch["labels"])
        TP_count += TP
        TN_count += TN
        FP_count += FP
        FN_count += FN
        gold_pos_sum += gold_pos
        # Novelty
        TP_novel, TN_novel, FP_novel, FN_novel, gold_pos_novel = calculate_amount(predictions_novel, batch_novel["labels"])
        TP_count_novel += TP_novel
        TN_count_novel += TN_novel
        FP_count_novel += FP_novel
        FN_count_novel += FN_novel
        gold_pos_sum_novel += gold_pos_novel
        for idx in range(logits_relation.shape[0]):
            single_answer = {}
            for col in column_names:
                if col == 'relation_predicted':
                    single_answer[ col ] = idx2str(relation_class, predictions_relation[ idx ] )
                elif col == 'novel_predicted' and idx2str(relation_class, predictions_relation[ idx ] ) != 'None':
                    single_answer[ col ] = idx2str(novel_class, predictions_novel[ idx ] )
                elif col == 'novel_predicted' and idx2str(relation_class, predictions_relation[ idx ] ) == 'None':
                    single_answer[ col ] = 'None'
                else:
                    single_answer[ col ] = df_eval.iloc[index][ col ]
            index +=1
            df_single = pd.DataFrame( single_answer, index = [0] )
            df_result = pd.concat( [ df_result, df_single ], ignore_index = True )
            # end of single row                
        # end of single batch
        eval_bar.update( 1 )
#         if (step+1) == 20:
#             break
    # Calculate metrics
    test_precision, test_recall, test_f1, test_accuracy = compute_metrics(TP_count, TN_count, FP_count, FN_count, gold_pos_sum)
    
    print("\nTP: {} TN: {} FP: {} FN: {} gold_pos: {}".format(TP_count, TN_count, FP_count, FN_count, gold_pos_sum))
    print("Accuracy: {} \tPrecision: {}\tRecall: {}\tF1: {}".format(test_accuracy, test_precision, test_recall, test_f1))
    
    test_precision_novel, test_recall_novel, test_f1_novel, test_accuracy_novel = compute_metrics(TP_count_novel, TN_count_novel, FP_count_novel, FN_count_novel, gold_pos_sum_novel)
    print("Novelty results: -----------------------")
    print("\nTP: {} TN: {} FP: {} FN: {} gold_pos: {}".format(TP_count_novel, TN_count_novel, FP_count_novel, FN_count_novel, gold_pos_sum_novel))
    print("Accuracy: {} \tPrecision: {}\tRecall: {}\tF1: {}".format(test_accuracy_novel, test_precision_novel, test_recall_novel, test_f1_novel))
    print(df_result)
    return df_result
def predict_relation_no_none_novel(path_relation, path_novel, df_eval, relation_class, novel_class, relation_stage, novel_stage):
    tokenizer = AutoTokenizer.from_pretrained(path_relation)
    model_relation = AutoModelForSequenceClassification.from_pretrained(
        path_relation,
        num_labels = len(relation_class)).to(device)
    model_novel = AutoModelForSequenceClassification.from_pretrained(
        path_novel,
        num_labels = len(novel_class)).to(device)
    # Load test dataset
    # df_eval = pd.read_csv(filedata, sep = ',')
    features = datasets.Features(
                {'labels': datasets.Value('string'),
                 'text': datasets.Value('string')})
    data_dict = {}
    data_dict['labels']  = [label for label in df_eval['ground truth']]
    data_dict['text']    = [text for text in df_eval['Document']]
    data_dataset = datasets.Dataset.from_dict(data_dict, features=features)
    data_dataset = data_dataset.map(
            lambda example: tokenizer.batch_encode_plus(
                example['text'], 
                truncation     = True, 
                max_length     = 512, 
                padding        = "max_length",
                stride         = 128
            ),
            batched=True,
        )
    transform_labels = globals().get(f"transform_labels_stage{relation_stage}")
    data_dataset = data_dataset.map(transform_labels, remove_columns = ['text', 'labels'])
    data_dataset.set_format = data_dataset.set_format("torch")
    test_dataloader = DataLoader(data_dataset, batch_size=config['model']['batch_size'])
    
    # Call test loader for novelty
    data_dict_novel = {}
    data_dict_novel['labels']  = [label for label in df_eval['Novelty']]
    data_dict_novel['text']    = [text for text in df_eval['Document']]
    data_dataset_novel = datasets.Dataset.from_dict(data_dict_novel, features=features)
    data_dataset_novel = data_dataset_novel.map(
            lambda example: tokenizer.batch_encode_plus(
                example['text'], 
                truncation     = True, 
                max_length     = 512, 
                padding        = "max_length",
                stride         = 128
            ),
            batched=True,
        )
    transform_labels_novel = globals().get(f"transform_labels_stage{novel_stage}")
    data_dataset_novel = data_dataset_novel.map(transform_labels_novel, remove_columns = ['text', 'labels'])
    data_dataset_novel.set_format = data_dataset_novel.set_format("torch")
    test_dataloader_novel = DataLoader(data_dataset_novel, batch_size=config['model']['batch_size'])
    
    model_relation.eval()
    model_relation.to( device )
    model_novel.eval()
    model_novel.to( device )
    index = 0
    metric = evaluate.load("accuracy")
    column_names = [ 'pmid', 'identifier_1', 'identifier_2', 'Type1', 'Type2', 'Document', 'ground truth', 'Novelty', 'relation_predicted', 'novel_predicted']
    df_result = pd.DataFrame( columns = column_names )
    TP_count, TN_count, FP_count, FN_count, gold_pos_sum = 0, 0, 0, 0, 0
    TP_count_novel, TN_count_novel, FP_count_novel, FN_count_novel, gold_pos_sum_novel = 0, 0, 0, 0, 0
    eval_bar = tqdm(range(len(test_dataloader)))
    for step, (batch, batch_novel) in enumerate(zip(test_dataloader, test_dataloader_novel)):
        with torch.no_grad():
            batch = {k: v.to(device) for k, v in batch.items()}
            batch_novel = {k: v.to(device) for k, v in batch_novel.items()}
            outputs_relation = model_relation(**batch)
            outputs_novel = model_novel(**batch_novel)
        logits_relation = outputs_relation.logits.cpu().detach()
        predictions_relation = torch.argmax(logits_relation, dim=-1)
        
        logits_novel = outputs_novel.logits.cpu().detach()
        predictions_novel = torch.argmax(logits_novel, dim=-1)
        metric.add_batch(predictions=predictions_relation, references=batch["labels"])
        # Relation
        TP, TN, FP, FN, gold_pos = calculate_amount(predictions_relation, batch["labels"])
        TP_count += TP
        TN_count += TN
        FP_count += FP
        FN_count += FN
        gold_pos_sum += gold_pos
        # Novelty
        TP_novel, TN_novel, FP_novel, FN_novel, gold_pos_novel = calculate_amount(predictions_novel, batch_novel["labels"])
        TP_count_novel += TP_novel
        TN_count_novel += TN_novel
        FP_count_novel += FP_novel
        FN_count_novel += FN_novel
        gold_pos_sum_novel += gold_pos_novel
        for idx in range(logits_relation.shape[0]):
            single_answer = {}
            for col in column_names:
                if col == 'novel_predicted':
                    single_answer[ col ] = idx2str(novel_class, predictions_novel[ idx ] )
                elif col == 'relation_predicted' and idx2str(novel_class, predictions_novel[ idx ] ) != 'None':
                    single_answer[ col ] = df_eval.iloc[index]['ground truth']
                elif col == 'relation_predicted' and idx2str(novel_class, predictions_novel[ idx ] ) == 'None':
                    single_answer[ col ] = 'None'
                else:
                    single_answer[ col ] = df_eval.iloc[index][ col ]
            index +=1
            df_single = pd.DataFrame( single_answer, index = [0] )
            df_result = pd.concat( [ df_result, df_single ], ignore_index = True )
            # end of single row                
        # end of single batch
        eval_bar.update( 1 )
#         if (step+1) == 20:
#             break
    # Calculate metrics
    test_precision, test_recall, test_f1, test_accuracy = compute_metrics(TP_count, TN_count, FP_count, FN_count, gold_pos_sum)
    
    print("\nTP: {} TN: {} FP: {} FN: {} gold_pos: {}".format(TP_count, TN_count, FP_count, FN_count, gold_pos_sum))
    print("Accuracy: {} \tPrecision: {}\tRecall: {}\tF1: {}".format(test_accuracy, test_precision, test_recall, test_f1))
    
    test_precision_novel, test_recall_novel, test_f1_novel, test_accuracy_novel = compute_metrics(TP_count_novel, TN_count_novel, FP_count_novel, FN_count_novel, gold_pos_sum_novel)
    print("Novelty results: -----------------------")
    print("\nTP: {} TN: {} FP: {} FN: {} gold_pos: {}".format(TP_count_novel, TN_count_novel, FP_count_novel, FN_count_novel, gold_pos_sum_novel))
    print("Accuracy: {} \tPrecision: {}\tRecall: {}\tF1: {}".format(test_accuracy_novel, test_precision_novel, test_recall_novel, test_f1_novel))
    print(df_result)
    return df_result
def predict_no_none_novel(path_novel, df_eval, novel_class, novel_stage):
    tokenizer = AutoTokenizer.from_pretrained(path_novel)
    model_novel = AutoModelForSequenceClassification.from_pretrained(
        path_novel,
        num_labels = len(novel_class)).to(device)
    # Load test dataset
    # df_eval = pd.read_csv(filedata, sep = ',')
    features = datasets.Features(
                {'labels': datasets.Value('string'),
                 'text': datasets.Value('string')})
    # Call test loader for novelty
    data_dict_novel = {}
    data_dict_novel['labels']  = [label for label in df_eval['Novelty']]
    data_dict_novel['text']    = [text for text in df_eval['Document']]
    data_dataset_novel = datasets.Dataset.from_dict(data_dict_novel, features=features)
    data_dataset_novel = data_dataset_novel.map(
            lambda example: tokenizer.batch_encode_plus(
                example['text'], 
                truncation     = True, 
                max_length     = 512, 
                padding        = "max_length",
                stride         = 128
            ),
            batched=True,
        )
    transform_labels_novel = globals().get(f"transform_labels_stage{novel_stage}")
    data_dataset_novel = data_dataset_novel.map(transform_labels_novel, remove_columns = ['text', 'labels'])
    data_dataset_novel.set_format = data_dataset_novel.set_format("torch")
    test_dataloader_novel = DataLoader(data_dataset_novel, batch_size=config['model']['batch_size'])
    model_novel.eval()
    model_novel.to( device )
    index = 0
    metric = evaluate.load("accuracy")
    column_names = [ 'pmid', 'identifier_1', 'identifier_2', 'Type1', 'Type2', 'Document', 'ground truth', 'Novelty', 'relation_predicted', 'novel_predicted']
    df_result = pd.DataFrame( columns = column_names )
    TP_count, TN_count, FP_count, FN_count, gold_pos_sum = 0, 0, 0, 0, 0
    TP_count_novel, TN_count_novel, FP_count_novel, FN_count_novel, gold_pos_sum_novel = 0, 0, 0, 0, 0
    eval_bar = tqdm(range(len(test_dataloader_novel)))
    for (step, batch_novel) in enumerate(test_dataloader_novel):
        with torch.no_grad():
            batch_novel = {k: v.to(device) for k, v in batch_novel.items()}
            outputs_novel = model_novel(**batch_novel)
        logits_novel = outputs_novel.logits.cpu().detach()
        predictions_novel = torch.argmax(logits_novel, dim=-1)
        # Novelty
        TP_novel, TN_novel, FP_novel, FN_novel, gold_pos_novel = calculate_amount(predictions_novel, batch_novel["labels"])
        TP_count_novel += TP_novel
        TN_count_novel += TN_novel
        FP_count_novel += FP_novel
        FN_count_novel += FN_novel
        gold_pos_sum_novel += gold_pos_novel
        for idx in range(logits_novel.shape[0]):
            single_answer = {}
            for col in column_names:
                if col == 'novel_predicted':
                    single_answer[ col ] = idx2str(novel_class, predictions_novel[ idx ] )
                elif col == 'relation_predicted' and idx2str(novel_class, predictions_novel[ idx ] ) != 'None':
                    single_answer[ col ] = df_eval.iloc[index]['ground truth']
                elif col == 'relation_predicted' and idx2str(novel_class, predictions_novel[ idx ] ) == 'None':
                    single_answer[ col ] = 'None'
                else:
                    single_answer[ col ] = df_eval.iloc[index][ col ]
            index +=1
            df_single = pd.DataFrame( single_answer, index = [0] )
            df_result = pd.concat( [ df_result, df_single ], ignore_index = True )
            # end of single row                
        # end of single batch
        eval_bar.update( 1 )
#         if (step+1) == 20:
#             break
    # Calculate metrics
    
    test_precision_novel, test_recall_novel, test_f1_novel, test_accuracy_novel = compute_metrics(TP_count_novel, TN_count_novel, FP_count_novel, FN_count_novel, gold_pos_sum_novel)
    print("Novelty results: -----------------------")
    print("\nTP: {} TN: {} FP: {} FN: {} gold_pos: {}".format(TP_count_novel, TN_count_novel, FP_count_novel, FN_count_novel, gold_pos_sum_novel))
    print("Accuracy: {} \tPrecision: {}\tRecall: {}\tF1: {}".format(test_accuracy_novel, test_precision_novel, test_recall_novel, test_f1_novel))
    print(df_result)
    return df_result
def predict_no_novel(path_novel, df_eval, novel_class, novel_stage):
    tokenizer = AutoTokenizer.from_pretrained(path_novel)
    model_novel = AutoModelForSequenceClassification.from_pretrained(
        path_novel,
        num_labels = len(novel_class)).to(device)
    # Load test dataset
    # df_eval = pd.read_csv(filedata, sep = ',')
    features = datasets.Features(
                {'labels': datasets.Value('string'),
                 'text': datasets.Value('string')})
    # Call test loader for novelty
    data_dict_novel = {}
    data_dict_novel['labels']  = [label for label in df_eval['ground truth']]
    data_dict_novel['text']    = [text for text in df_eval['Document']]
    data_dataset_novel = datasets.Dataset.from_dict(data_dict_novel, features=features)
    data_dataset_novel = data_dataset_novel.map(
            lambda example: tokenizer.batch_encode_plus(
                example['text'], 
                truncation     = True, 
                max_length     = 512, 
                padding        = "max_length",
                stride         = 128
            ),
            batched=True,
        )
    transform_labels_novel = globals().get(f"transform_labels_stage{novel_stage}")
    data_dataset_novel = data_dataset_novel.map(transform_labels_novel, remove_columns = ['text', 'labels'])
    data_dataset_novel.set_format = data_dataset_novel.set_format("torch")
    test_dataloader_novel = DataLoader(data_dataset_novel, batch_size=config['model']['batch_size'])
    model_novel.eval()
    model_novel.to( device )
    index = 0
    metric = evaluate.load("accuracy")
    column_names = [ 'pmid', 'identifier_1', 'identifier_2', 'Type1', 'Type2', 'Document', 'ground truth', 'relation_predicted', 'novel_predicted']
    df_result = pd.DataFrame( columns = column_names )
    TP_count, TN_count, FP_count, FN_count, gold_pos_sum = 0, 0, 0, 0, 0
    TP_count_novel, TN_count_novel, FP_count_novel, FN_count_novel, gold_pos_sum_novel = 0, 0, 0, 0, 0
    eval_bar = tqdm(range(len(test_dataloader_novel)))
    for (step, batch_novel) in enumerate(test_dataloader_novel):
        with torch.no_grad():
            batch_novel = {k: v.to(device) for k, v in batch_novel.items()}
            outputs_novel = model_novel(**batch_novel)
        logits_novel = outputs_novel.logits.cpu().detach()
        predictions_novel = torch.argmax(logits_novel, dim=-1)
        # Novelty
        TP_novel, TN_novel, FP_novel, FN_novel, gold_pos_novel = calculate_amount(predictions_novel, batch_novel["labels"])
        TP_count_novel += TP_novel
        TN_count_novel += TN_novel
        FP_count_novel += FP_novel
        FN_count_novel += FN_novel
        gold_pos_sum_novel += gold_pos_novel
        for idx in range(logits_novel.shape[0]):
            single_answer = {}
            for col in column_names:
                if col == 'novel_predicted':
                    single_answer[ col ] = idx2str(novel_class, predictions_novel[ idx ] )
                elif col == 'relation_predicted':
                    single_answer[ col ] = df_eval.iloc[index]['ground truth']
                else:
                    single_answer[ col ] = df_eval.iloc[index][ col ]
            index +=1
            df_single = pd.DataFrame( single_answer, index = [0] )
            df_result = pd.concat( [ df_result, df_single ], ignore_index = True )
            # end of single row                
        # end of single batch
        eval_bar.update( 1 )
#         if (step+1) == 20:
#             break
    # Calculate metrics
    
    test_precision_novel, test_recall_novel, test_f1_novel, test_accuracy_novel = compute_metrics(TP_count_novel, TN_count_novel, FP_count_novel, FN_count_novel, gold_pos_sum_novel)
    print("Novelty results: -----------------------")
    print("\nTP: {} TN: {} FP: {} FN: {} gold_pos: {}".format(TP_count_novel, TN_count_novel, FP_count_novel, FN_count_novel, gold_pos_sum_novel))
    print("Accuracy: {} \tPrecision: {}\tRecall: {}\tF1: {}".format(test_accuracy_novel, test_precision_novel, test_recall_novel, test_f1_novel))
    print(df_result)
    return df_result
def idx2str(table, idx) -> str:
    return table[ idx ]

def dataframe_refinement( df: pd.DataFrame ) -> pd.DataFrame:
    print( df.head( 3 ) )
    df_final = pd.DataFrame( columns = df.columns ) # 生成新答案資料表
    groups_final = df.groupby( "pmid" ) # 依照 PMID 分組
    group_names_final = groups_final.groups.keys() # 取得各 PMID
    print( 'group_names_final:', group_names_final )
    bar = tqdm( total = len( group_names_final ), desc = 'dataframe refinement' )
    for name in group_names_final:
        # 分離出各 PubMedID
        df_sub = groups_final.get_group( name ) # 抓取各 PMID 對應的答案( 複數個 )
        groups_sub = df_sub.groupby( [ 'identifier_1', 'identifier_2' ] )
        group_names_sub = groups_sub.groups.keys() # 取得各組關係組合
        # value_counts() 輸出為 series
        for name_sub in group_names_sub:
            # 分離出各組 entities pair
            df_pair = groups_sub.get_group( name_sub ) # 抓取各組關係對應的答案( 複數個 )
            series_vc = df_pair[ 'predicted' ].value_counts() 
            max_name = series_vc.idxmax() # 找出最大者( 類似投票，選出現最多次的 Predict )
            # 填入答案
            single_row = {
                'pmid' : df_pair.iloc[ 0 ][ 'pmid' ],
                'identifier_1' : df_pair.iloc[ 0 ][ 'identifier_1' ],
                'identifier_2' : df_pair.iloc[ 0 ][ 'identifier_2' ],
                'ground truth' : df_pair.iloc[ 0 ][ 'ground truth' ],
                'predicted' : max_name,
            }
            df_single = pd.DataFrame( single_row, index = [0] )
            df_final = pd.concat( [ df_final, df_single ], ignore_index = True )
        bar.update( 1 )    
            
    return df_final

def evaluate_pipeline( filename, 
                    tokenizer,
                    model_encode):
    global config
    print( 'evaluate_pipeline: start.' )
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print( 'device: {}'.format( device ) )
    config[ 'model' ][ 'device' ] = device
    
    # model
    model = BertClassification( config = config )
    if os.path.exists( filename ):
        with open( filename, 'rb' ) as f:
            record_dict = torch.load( f )
            # record_dict 內容：epoch, model_weight, optimizer
            model.load_state_dict( record_dict, strict = True )
            print( 'pre-trained weight loaded.' )
    model.eval()
    model.to( device )
    
    # data preprocessing
    flag = True
    if os.path.exists( config[ 'test' ] ) is True and flag is True:
        print( 'Processed DataFrame exists.' )
        df_eval = pd.read_csv( config[ 'test' ] ).reset_index( drop = True )
        df_eval = df_eval.applymap(str)
    print( 'evaluation pipeline, df_eval shape: {}'.format( df_eval.shape ) )
    
    # 產生資料集
    num_total = int( len( df_eval ) )
    global_range = list( range( num_total ) ) # 產生一個 indexing table 
    # 評估時，同一篇文章( 相同 PubMedID )需擺在一起，不要 shuffle
    valid_set = Data( df = df_eval, 
                     idx_range = global_range,
                     tokenizer=tokenizer,
                     model_encode=model_encode,)
    # 建立 DataLoader ，再重複一次，不要 shuffle
    valid_loader = DataLoader( dataset = valid_set, batch_size = config[ 'model' ][ 'batch_size' ])
    
    # 儲存結果
    column_names = [ 'pmid', 'identifier_1', 'identifier_2', 'ground truth', 'predicted' ]
    df_result = pd.DataFrame( columns = column_names )
    TP_sum, TN_sum, FP_sum, FN_sum, gold_pos_sum = 0, 0, 0, 0, 0
    with torch.no_grad():
        eval_bar = tqdm( total = len( valid_loader ), desc = 'evaluation( full ver. )')
        for batch_cnt, data in enumerate( valid_loader, 0 ):
            
            # get data
            document, label, type1, type2, idx = data
            label = torch.Tensor( label ).to( device )
            type1 = torch.Tensor(type1).to( device )
            type2 = torch.Tensor(type2).to( device )
            document = document.to( device )
            
            # Input to the model
            output = model(document, type1, type2)
            
            output = output.cpu().detach()
            output = torch.argmax(output, dim = 1)
            output = output.detach().numpy()
            
            label = label.cpu().detach()
            label = torch.argmax(label, dim = 1)
            label = label.detach().numpy()
            
            TP, TN, FP, FN, gold_pos = calculate_amount(output, label)
            TP_sum += TP
            TN_sum += TN
            FP_sum += FP
            FN_sum += FN
            gold_pos_sum += gold_pos
            idx = idx.detach().numpy()
            for cnt in range( idx.shape[ 0 ] ):
                single_answer = {}
                for col in column_names:
                    if col == 'predicted':
                        single_answer[ col ] = idx2str( output[ cnt ] )
                    else:
                        single_answer[ col ] = df_eval.iloc[ idx[ cnt ] ][ col ]
                df_single = pd.DataFrame( single_answer, index = [0] )
                # pd.DataFrame.append 即將 depricated ，請改用 pd.concat
                # 關於由 dict 轉 DataFrame 時，僅存在 1 row 並報錯( ValueError )的解法：
                # https://blog.csdn.net/weixin_39750084/article/details/81429037
                df_result = pd.concat( [ df_result, df_single ], ignore_index = True )
                # end of single row                
            # end of single batch
            eval_bar.update( 1 )
        # end of iteration
        # Accuracy for 1 epoch
        test_precision, test_recall, test_f1, test_accuracy = compute_metrics(TP_sum, TN_sum, FP_sum, FN_sum, gold_pos_sum)
        print(TP_sum, TN_sum, FP_sum, FN_sum, gold_pos_sum)
        print("Accuracy: {} \tPrecision: {}\tRecall: {}\tF1: {}".format(test_accuracy, test_precision, test_recall, test_f1))
        df_result.to_csv(config['model_result'])
    # end of no_grad() part
            
    return

# +
def pipeline(tokenizer, model_encode):
    global config
    
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print( 'device: {}'.format( device ) )
    config[ 'model' ][ 'device' ] = device
    
    flag = True
    if os.path.exists( config[ 'train' ] ) is True and flag is True:
        print( 'Processed DataFrame exists.' )
        df_combine = pd.read_csv(config[ 'train' ])
        df_combine = df_combine.applymap(str)
    
    
    
    num_train = int( len( df_combine ) * 0.9 )
    global_range = list( range( len( df_combine ) ) ) # 產生一個 indexing table 
    print("Train: ",len(global_range[ : num_train ]))
    print(len(global_range[ num_train : ]))
    training_set = Data( df = df_combine, 
                    tokenizer=tokenizer,
                    model_encode=model_encode,
                    idx_range = global_range[ : num_train ], )
    test_set = Data( df = df_combine, 
                tokenizer=tokenizer,
                model_encode=model_encode,
                idx_range = global_range[ num_train : ],)
    
    # model
    model = BertClassification( config = config )
    
    # Freeze the transformers layer, only update the classifier
    #model.transformer.requires_grad_(False)
    
#     class_weight = torch.FloatTensor([1.35890177e-01, 1.16111365e+00, 4.24000000e+01, 9.08571429e+01,
#  8.48000000e+02, 8.20645161e+01, 2.31272727e+02, 3.33420708e+00,
#  2.33823529e+00]).to(device)
    criterion = torch.nn.CrossEntropyLoss()

    # optimizer
    learning_rate = config['training']['learning_rate']
    if config['training']['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, model.parameters()), lr=learning_rate, weight_decay=1e-4)
    elif config['training']['optimizer'] == 'adam':
        optimizer = AdamW(filter(lambda x: x.requires_grad, model.parameters()), lr=learning_rate)
    elif config['training']['optimizer'] == 'adadelta':
        optimizer = torch.optim.Adadelta(filter(lambda x: x.requires_grad, model.parameters()), lr=learning_rate, weight_decay=1e-4)
    elif config['training']['optimizer'] == 'rmsprop':
        optimizer = torch.optim.RMSprop(filter(lambda x: x.requires_grad, model.parameters()), lr=learning_rate, weight_decay=1e-4)
    print('Optimizer:', config['training']['optimizer'])
    print('Learning rate:', config['training']['learning_rate'])

    # log info
    train_log_string = '%s :: Epoch %i :: Iter %i / %i :: train loss: %0.4f'
    valid_log_string = '%s :: Epoch %i :: valid loss: %0.4f\n'

    # Restore saved model (if one exists).
    ckpt_path = os.path.join( config['ckpt_dir'], config['experiment_name']+'.pt' )
    if os.path.exists(ckpt_path):
        print('Loading checkpoint: %s' % ckpt_path)
        ckpt = torch.load( ckpt_path )
        model.load_state_dict( ckpt['siamese'], strict = False )
        optimizer.load_state_dict( ckpt['optimizer'] )
    else:
        print('Fresh start!\n')
    
    training_process( config = config,
                    model = model,
                    training_set = training_set,
                    test_set = test_set,
                    optimizer = optimizer,
                    criterion = criterion,
                    valid_log_string = valid_log_string,
                    device = device )
# -

if __name__ == '__main__':
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    # Clear all cache
    gc.collect()
    torch.cuda.empty_cache()
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # Run again
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") 
    print("Device: ", device)
    # Relation + Novelty classes
    classes_stage1 = ['None', 'Other']
    classes_stage2 = ['Other', 'Association', 'Positive_Correlation', 'Negative_Correlation']
    classes_stage3 = ['Bind', 'Cotreatment', 'Comparison', 'Drug_Interaction', 'Conversion']
    classes_all = ['None', 'Association', 'Positive_Correlation', 'Negative_Correlation', 'Bind', 'Cotreatment', 'Comparison', 'Drug_Interaction', 'Conversion']  # stage 4
    classes_novelty5 = ['None', 'No', 'Novel']
    classes_novelty6 = ['No', 'Novel']

    parser = argparse.ArgumentParser(description='Train or Evaluate models')
    subparsers = parser.add_subparsers(dest='command')

    # Subparser for training
    parser_train = subparsers.add_parser('train', help='Train models')
    parser_train.add_argument('--mode', type=str, choices=['biore', 'novelty'], required=True, help='Training mode')
    parser_train.add_argument('--model_name', type=str, default='microsoft/BiomedNLP-PubMedBERT-large-uncased-abstract', help='Model name')
    parser_train.add_argument('--train_file', type=str, required=True, help='Training file path')
    parser_train.add_argument('--test_file', type=str, required=True, help='Test file path')

    # Subparser for evaluation
    parser_eval = subparsers.add_parser('evaluate', help='Evaluate models')
    parser_eval.add_argument('--eval_file', type=str, required=True, help='Evaluation file path')
    parser_eval.add_argument('--path_relation', type=str, required=True, help='Path to relation model')
    parser_eval.add_argument('--path_novel', type=str, required=True, help='Path to novelty model')
    parser_eval.add_argument('--output_file', type=str, required=True, help='Output file path for results')

    args = parser.parse_args()

    if args.command == 'train':
        if args.mode == 'biore':
            train(args.model_name, args.train_file, args.test_file, classes_all, 4)
        elif args.mode == 'novelty':
            train_novelty(args.model_name, args.train_file, args.test_file, classes_novelty5, 5)
    elif args.command == 'evaluate':
        df_eval = pd.read_csv(args.eval_file, sep=',', keep_default_na=False)
        df_result = predict_relation_no_none_novel(
            path_relation=args.path_relation,
            path_novel=args.path_novel,
            df_eval=df_eval,
            relation_class=classes_all,
            novel_class=classes_novelty5,
            relation_stage=4,
            novel_stage=5
        )
        df_result.to_csv(args.output_file, index=False)
import math, random, datetime, argparse
import numpy as np
import pandas as pd
import torch

from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.optim import AdamW  # use torch’s AdamW

from transformers import AutoTokenizer, RobertaConfig, get_linear_schedule_with_warmup

from models.models import BiEncoderAttentionWithRationaleClassification
from evaluation_utils import *

parser = argparse.ArgumentParser("BiEncoder")
parser.add_argument("--lr", default=2e-5, type=float)
parser.add_argument("--lambda_EI", default=0.5, type=float)
parser.add_argument("--lambda_RE", default=0.5, type=float)
parser.add_argument("--dropout", default=0.1, type=float)
parser.add_argument("--max_len", default=64, type=int)
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--epochs", default=4, type=int)
parser.add_argument("--seed_val", default=12, type=int)
parser.add_argument("--train_path", type=str)
parser.add_argument("--dev_path", type=str)
parser.add_argument("--test_path", type=str)
parser.add_argument("--do_validation", action="store_true")
parser.add_argument("--do_test", action="store_true")
parser.add_argument("--save_model", action="store_true")
parser.add_argument("--save_model_path", type=str)
args = parser.parse_args()

print("=====================Args====================")
for k, v in vars(args).items(): print(f"{k} = {v}")
print("=============================================")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
if not torch.cuda.is_available():
    print('No GPU available, using the CPU instead.')

# Load data
if not args.train_path:
    print('No input training data specified.\nExiting...')
    exit(-1)

df = pd.read_csv(args.train_path, delimiter=',')
df['rationale_labels'] = df['rationale_labels'].apply(lambda s: torch.tensor(np.asarray([int(i) for i in s.split(',')]), dtype=torch.long))

if args.do_test:
    if not args.test_path:
        print('No input test data specified.\nExiting...')
        exit(-1)
    df_test = pd.read_csv(args.test_path, delimiter=',')
    df_test['rationale_labels'] = df_test['rationale_labels'].apply(lambda s: torch.tensor(np.asarray([int(i) for i in s.split(',')]), dtype=torch.long))

if args.do_validation:
    if not args.dev_path:
        print('No input validation data specified.\nExiting...')
        exit(-1)
    df_val = pd.read_csv(args.dev_path, delimiter=',')
    df_val['rationale_labels'] = df_val['rationale_labels'].apply(lambda s: torch.tensor(np.asarray([int(i) for i in s.split(',')]), dtype=torch.long))
    df_val['rationale_labels_trimmed'] = df_val['rationale_labels_trimmed'].astype(int)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained('roberta-base', use_fast=True)

def batch_encode(texts):
    enc = tokenizer(
        list(texts),
        add_special_tokens=True,
        padding="max_length",
        truncation=True,
        max_length=args.max_len,
        return_tensors="pt",
    )
    return enc["input_ids"], enc["attention_mask"]

input_ids_RP, attention_masks_RP = batch_encode(df.response_post)
input_ids_SP, attention_masks_SP = batch_encode(df.seeker_post)

labels = torch.tensor(df.level.values.astype(int))
rationales = torch.stack(df.rationale_labels.values.tolist(), dim=0)

if args.do_validation:
    val_input_ids_RP, val_attention_masks_RP = batch_encode(df_val.response_post)
    val_input_ids_SP, val_attention_masks_SP = batch_encode(df_val.seeker_post)
    val_labels = torch.tensor(df_val.level.values.astype(int))
    val_rationales = torch.stack(df_val['rationale_labels'].values.tolist(), dim=0)
    val_rationales_trimmed = torch.tensor(df_val['rationale_labels_trimmed'].values.astype(int))

if args.do_test:
    test_input_ids_RP, test_attention_masks_RP = batch_encode(df_test.response_post)
    test_input_ids_SP, test_attention_masks_SP = batch_encode(df_test.seeker_post)
    test_labels = torch.tensor(df_test.level.values.astype(int))
    test_rationales = torch.stack(df_test.rationale_labels.values.tolist(), dim=0)
    test_rationales_trimmed = torch.tensor(df_test.rationale_labels_trimmed.values.astype(int))

# Model
model = BiEncoderAttentionWithRationaleClassification(hidden_dropout_prob=args.dropout).to(device)

# Do not finetune seeker encoder
for p in model.seeker_encoder.parameters():
    p.requires_grad = False

optimizer = AdamW(model.parameters(), lr=args.lr, eps=1e-8)

# DataLoaders
train_dataset = TensorDataset(input_ids_SP, attention_masks_SP, input_ids_RP, attention_masks_RP, labels, rationales)
train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=args.batch_size)

if args.do_validation:
    val_dataset = TensorDataset(val_input_ids_SP, val_attention_masks_SP, val_input_ids_RP, val_attention_masks_RP, val_labels, val_rationales, val_rationales_trimmed)
    validation_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=args.batch_size)

if args.do_test:
    test_dataset = TensorDataset(test_input_ids_SP, test_attention_masks_SP, test_input_ids_RP, test_attention_masks_RP, test_labels, test_rationales, test_rationales_trimmed)
    test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=args.batch_size)

total_steps = len(train_dataloader) * args.epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

random.seed(args.seed_val); np.random.seed(args.seed_val); torch.manual_seed(args.seed_val); torch.cuda.manual_seed_all(args.seed_val)

# Training
for epoch_i in range(args.epochs):
    model.train()
    total_train_loss = total_train_empathy_loss = total_train_rationale_loss = 0
    pbar = tqdm(total=len(train_dataloader), desc="training")

    for step, batch in enumerate(train_dataloader):
        b_input_ids_SP = batch[0].to(device)
        b_input_mask_SP = batch[1].to(device)
        b_input_ids_RP = batch[2].to(device)
        b_input_mask_RP = batch[3].to(device)
        b_labels = batch[4].to(device)
        b_rationales = batch[5].to(device)

        model.zero_grad()
        loss, loss_empathy, loss_rationale, logits_empathy, logits_rationale = model(
            input_ids_SP=b_input_ids_SP, input_ids_RP=b_input_ids_RP,
            attention_mask_SP=b_input_mask_SP, attention_mask_RP=b_input_mask_RP,
            empathy_labels=b_labels, rationale_labels=b_rationales,
            lambda_EI=args.lambda_EI, lambda_RE=args.lambda_RE
        )

        total_train_loss += loss.item()
        total_train_empathy_loss += loss_empathy.item()
        total_train_rationale_loss += loss_rationale.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        pbar.set_postfix_str(f"loss: {float(total_train_loss/(step+1)):.4f} epoch: {epoch_i}")
        pbar.update(1)
    pbar.close()

    if args.do_validation:
        print('\n\nRunning validation...\n')
        model.eval()
        total_eval_accuracy_empathy = total_eval_accuracy_rationale = 0
        total_pos_f1_empathy = total_micro_f1_empathy = total_macro_f1_empathy = 0.0
        total_macro_f1_rationale = total_iou_rationale = 0.0

        for batch in validation_dataloader:
            b_input_ids_SP = batch[0].to(device)
            b_input_mask_SP = batch[1].to(device)
            b_input_ids_RP = batch[2].to(device)
            b_input_mask_RP = batch[3].to(device)
            b_labels = batch[4].to(device)
            b_rationales = batch[5].to(device)
            b_rationales_trimmed = batch[6].to(device)

            with torch.no_grad():
                loss, loss_empathy, loss_rationale, logits_empathy, logits_rationale = model(
                    input_ids_SP=b_input_ids_SP, input_ids_RP=b_input_ids_RP,
                    attention_mask_SP=b_input_mask_SP, attention_mask_RP=b_input_mask_RP,
                    empathy_labels=b_labels, rationale_labels=b_rationales,
                    lambda_EI=args.lambda_EI, lambda_RE=args.lambda_RE
                )

            logits_empathy = logits_empathy.detach().cpu().numpy()
            logits_rationale = logits_rationale.detach().cpu().numpy()
            label_empathy_ids = b_labels.to('cpu').numpy()
            label_rationale_ids = b_rationales.to('cpu').numpy()
            rationale_lens = b_rationales_trimmed.to('cpu').numpy()

            total_eval_accuracy_empathy += flat_accuracy(logits_empathy, label_empathy_ids, axis_=1)
            total_eval_accuracy_rationale += flat_accuracy_rationale(logits_rationale, label_rationale_ids, label_empathy_ids, rationale_lens, axis_=2)

            _, _, macro_f1_empathy = compute_f1(logits_empathy, label_empathy_ids, axis_=1)
            macro_f1_rationale = compute_f1_rationale(logits_rationale, label_rationale_ids, label_empathy_ids, rationale_lens, axis_=2)
            iou_f1_rationale = iou_f1(logits_rationale, label_rationale_ids, label_empathy_ids, rationale_lens, axis_=2)

            total_macro_f1_empathy += macro_f1_empathy
            total_macro_f1_rationale += macro_f1_rationale
            total_iou_rationale += iou_f1_rationale

        n_val = len(validation_dataloader)
        print(f"  Accuracy-Empathy: {total_eval_accuracy_empathy/n_val:.4f}")
        print(f"  macro_f1_empathy: {total_macro_f1_empathy/n_val:.4f}")
        print(f"  Accuracy-Rationale: {total_eval_accuracy_rationale/n_val:.4f}")
        print(f"  IOU-F1-Rationale: {total_iou_rationale/n_val:.4f}")
        print(f"  macro_f1_rationale: {total_macro_f1_rationale/n_val:.4f}")

# Test
if args.do_test:
    print("\n\nRunning test...\n")
    model.eval()
    total_eval_accuracy_empathy = total_eval_accuracy_rationale = 0
    total_macro_f1_empathy = total_macro_f1_rationale = total_iou_rationale = 0.0

    for batch in test_dataloader:
        b_input_ids_SP = batch[0].to(device)
        b_input_mask_SP = batch[1].to(device)
        b_input_ids_RP = batch[2].to(device)
        b_input_mask_RP = batch[3].to(device)
        b_labels = batch[4].to(device)
        b_rationales = batch[5].to(device)
        b_rationales_trimmed = batch[6].to(device)

        with torch.no_grad():
            loss, loss_empathy, loss_rationale, logits_empathy, logits_rationale = model(
                input_ids_SP=b_input_ids_SP, input_ids_RP=b_input_ids_RP,
                attention_mask_SP=b_input_mask_SP, attention_mask_RP=b_input_mask_RP,
                empathy_labels=b_labels, rationale_labels=b_rationales,
                lambda_EI=args.lambda_EI, lambda_RE=args.lambda_RE
            )

        logits_empathy = logits_empathy.detach().cpu().numpy()
        logits_rationale = logits_rationale.detach().cpu().numpy()
        label_empathy_ids = b_labels.to('cpu').numpy()
        label_rationale_ids = b_rationales.to('cpu').numpy()
        rationale_lens = b_rationales_trimmed.to('cpu').numpy()

        total_eval_accuracy_empathy += flat_accuracy(logits_empathy, label_empathy_ids, axis_=1)
        total_eval_accuracy_rationale += flat_accuracy_rationale(logits_rationale, label_rationale_ids, label_empathy_ids, rationale_lens, axis_=2)

        _, _, macro_f1_empathy = compute_f1(logits_empathy, label_empathy_ids, axis_=1)
        macro_f1_rationale = compute_f1_rationale(logits_rationale, label_rationale_ids, label_empathy_ids, rationale_lens, axis_=2)
        iou_f1_rationale = iou_f1(logits_rationale, label_rationale_ids, label_empathy_ids, rationale_lens, axis_=2)

        total_macro_f1_empathy += macro_f1_empathy
        total_macro_f1_rationale += macro_f1_rationale
        total_iou_rationale += iou_f1_rationale

    n_test = len(test_dataloader)
    print(f"  Accuracy-Empathy: {total_eval_accuracy_empathy/n_test:.4f}")
    print(f"  macro_f1_empathy: {total_macro_f1_empathy/n_test:.4f}")
    print(f"  Accuracy-Rationale: {total_eval_accuracy_rationale/n_test:.4f}")
    print(f"  IOU-F1-Rationale: {total_iou_rationale/n_test:.4f}")
    print(f"  macro_f1_rationale: {total_macro_f1_rationale/n_test:.4f}")

if args.save_model:
    torch.save(model.state_dict(), args.save_model_path)

import torch
import os, sys, argparse
import numpy as np 
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, XLMRobertaForSequenceClassification 

from sklearn.metrics import f1_score, precision_score, recall_score

from data_utils import OLID 

parser = argparse.ArgumentParser(description='XLM-R')
parser.add_argument("--logs_dir", default='english', type=str)

parser.add_argument("--language", default='english', type=str)
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--learning_rate", default=2e-5, type=float)
parser.add_argument("--weight_decay", default=5e-2, type=float)
parser.add_argument("--epochs", default=50, type=int)

args = parser.parse_args()

if not os.path.exists(args.logs_dir):
  os.mkdir(args.logs_dir)
  print(f'{args.logs_dir} created!')

logs_file = os.path.join(args.logs_dir, 'logs.txt')

def logprint(log):
    print(log, end='')
    with open(logs_file, 'a') as f:
        f.write(log) 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define the model repo
model_name = "xlm-roberta-base" 

# tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokens = tokenizer.tokenize('羅伯特 · 皮爾 斯 生於 1863年 , 在 英國 曼徹斯特 學習 而 成為 一 位 工程師 . 1933年 , 皮爾斯 在 直布羅陀去世 .')
# print(tokens)

# Download pytorch model
model = XLMRobertaForSequenceClassification.from_pretrained(model_name,
                                  num_labels = 2, # The number of output labels.   
                                  output_attentions = False, # Whether the model returns attentions weights.
                                  output_hidden_states = False, # Whether the model returns all hidden-states.
                                  )

model.cuda()

def warmp_up(lr, epoch, warmup_epochs):
  lr = lr * (epoch / warmup_epochs)
  return lr

def step_down(lr, epoch, step_down_epochs):
  lr = (1 - epoch / step_down_epochs) * lr 
  return lr 

def validate_model(dataloader):
  model.eval()

  running_loss = 0.
  matches = 0.
  samples = 0.
  running_labels = []
  running_preds = []

  for idx, item in enumerate(dataloader):

    inputs = item['input']
    labels = item['label']

    input_ids = tokenizer(inputs, padding=True, truncation=True, max_length=512, return_tensors='pt')

    model_input = input_ids['input_ids'].cuda()
    attention_masks = input_ids['attention_mask'].cuda()
    labels = labels.to(device) 

    with torch.no_grad():
      outputs = model(model_input, 
                      token_type_ids=None, 
                      attention_mask=attention_masks, 
                      labels=labels)
      loss = outputs.loss

    logits = outputs.logits
    running_loss += loss.item()

    preds = torch.argmax(logits, dim=1)

    running_labels.extend( list(labels.detach().cpu().numpy()) )
    running_preds.extend( list(preds.detach().cpu().numpy()) )

    running_loss += loss.item()

    matches += torch.sum(preds == labels).item()
    samples += labels.shape[0]

    torch.cuda.empty_cache()

    # break

  # print(running_labels)

  precision = precision_score(running_labels, running_preds)
  recall = recall_score(running_labels, running_preds)
  f1 = f1_score(running_labels, running_preds)

  running_loss /= len(dataloader)
  running_acc = matches / samples 

  return precision, recall, f1, running_loss, running_acc

dataroot = 'data/'
langs = ['english', 'danish', 'turkish', 'arabic', 'greek']
batch_size = args.batch_size

for lang in langs:

    logprint('==============================\n')
    logprint(f'XLM-R finetuned on - {lang}\n')
    logprint('==============================\n')

    model_path = f'models/{lang}/xlm_{lang}.pth'
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['best_model_wts'])

    for ev_lang in langs:

        if ev_lang == lang:
            continue

        logprint(f'Zero-Shot eval on {ev_lang}\n')

        dataset = OLID(dataroot, 'test', ev_lang)
        testDataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        precision, recall, f1, val_loss, val_acc = validate_model(testDataloader)

        logprint(f"Val Loss: {val_loss}: Val Acc: {val_acc}\n")
        logprint(f'Val Precision:{precision} Recall: {recall} F1: {f1}\n')


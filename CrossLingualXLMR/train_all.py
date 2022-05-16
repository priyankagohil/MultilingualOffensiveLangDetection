import torch
import os, sys, argparse
import numpy as np 
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, XLMRobertaForSequenceClassification 

from sklearn.metrics import f1_score, precision_score, recall_score

from data_utils import OLID, OLIDUniformSample, OLIDSOLID

parser = argparse.ArgumentParser(description='XLM-R')
parser.add_argument("--logs_dir", default='english', type=str)

parser.add_argument("--data_type", type=str, default='olid')

parser.add_argument("--language", default='all', type=str)

parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--learning_rate", default=2e-5, type=float)
parser.add_argument("--weight_decay", default=5e-2, type=float)
parser.add_argument("--epochs", default=50, type=int)
parser.add_argument("--prev_ckpt", default=None, type=str)

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

dataroot = 'data/'

lang = args.language
batch_size = args.batch_size

if args.data_type=='olid_uniform':
  dataset = OLIDUniformSample(dataroot, 'train')
  trainDataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

elif args.data_type=='olid':
  dataset = OLID(dataroot, 'train', 'all')
  trainDataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

elif args.data_type=='solid_olid':
  print('solid_olid datatype')
  dataset = OLIDSOLID(dataroot)
  trainDataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

dataset = OLID(dataroot, 'test', 'all')
testDataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True) 

langs = ['arabic', 'danish', 'english', 'greek', 'turkish']
test_loaders = {}
for test_lang in langs:
    dataset = OLID(dataroot, 'test', test_lang)
    test_loaders[test_lang] = DataLoader(dataset, batch_size=batch_size, shuffle=True)

weight_decay = args.weight_decay
learning_rate = args.learning_rate 

optimizer = torch.optim.AdamW(model.parameters(),
                  lr = learning_rate, 
                  eps = 1e-8,
                  weight_decay=weight_decay,
                )

epochs = args.epochs

lr_warmup_epochs = 0.1 * epochs 
init_lr = 0 

if not isinstance(args.prev_ckpt, type(None)):
  checkpoint = torch.load(args.prev_ckpt)
  model.load_state_dict(checkpoint['best_model_wts'])
  
  logprint(f'model loaded with best_f1: {checkpoint["best_f1"]}\n')
    

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

    precision = precision_score(running_labels, running_preds)
    recall = recall_score(running_labels, running_preds)
    f1 = f1_score(running_labels, running_preds)

    logprint(f'val : {idx}/{len(dataloader)} precision: {precision} recall: {recall} f1: {f1}\n')

    # break

  # print(running_labels)

  precision = precision_score(running_labels, running_preds)
  recall = recall_score(running_labels, running_preds)
  f1 = f1_score(running_labels, running_preds)

  running_loss /= len(dataloader)
  running_acc = matches / samples 

  return precision, recall, f1, running_loss, running_acc

def train_model(dataloader):

  model.train()

  running_loss = 0.
  matches = 0.
  samples = 0.
  running_labels = []
  running_preds = []

  for idx, item in enumerate(dataloader):

    inputs = item['input']
    labels = item['label']

    input_ids = tokenizer(inputs, padding=True, truncation=True, max_length=512, return_tensors='pt')

    model_input = input_ids['input_ids'].to(device)
    attention_masks = input_ids['attention_mask'].to(device)
    labels = labels.to(device) 

    model.zero_grad()

    outputs = model(model_input, 
                    token_type_ids=None, 
                    attention_mask=attention_masks, 
                    labels=labels)
    loss = outputs.loss

    loss.backward()
    optimizer.step()

    logits = outputs.logits
    running_loss += loss.item()

    preds = torch.argmax(logits, dim=1)

    matches += torch.sum(preds == labels).item()
    samples += preds.shape[0]

    torch.cuda.empty_cache()

    running_labels.extend( list(labels.detach().cpu().numpy()) )
    running_preds.extend( list(preds.detach().cpu().numpy()) )

    if idx % 100 == 0:
      precision = precision_score(running_labels, running_preds)
      recall = recall_score(running_labels, running_preds)
      f1 = f1_score(running_labels, running_preds)

      logprint(f'train: {idx}/{len(dataloader)} precision: {precision} recall: {recall} f1: {f1}\n')


    # break

  running_loss /= len(dataloader)
  running_acc = matches / samples

  return running_loss, running_acc  


best_val_f1 = 0. 

for epoch_i in range(epochs):

  logprint("\n")
  logprint('======== Epoch {:} / {:} ========\n'.format(epoch_i + 1, epochs))
  logprint('Training...\n')

  # if epoch_i < lr_warmup_epochs:
  #   lr = warmp_up(learning_rate, epoch_i, lr_warmup_epochs)
  # else:
  #   lr = step_down(learning_rate, epoch_i - lr_warmup_epochs, epochs - lr_warmup_epochs)
  lr = step_down(learning_rate, epoch_i, epochs)
  for g in optimizer.param_groups:
    g['lr'] = lr
  
  train_loss, train_acc = train_model(trainDataloader)

  logprint(f"Train loss: {train_loss} Train Acc: {train_acc}\n")

  for test_lang in langs:
      precision, recall, f1, val_loss, val_acc = validate_model(test_loaders[test_lang])
    #   logprint(f"Val Loss: {val_loss}: Val Acc: {val_acc}\n")
      logprint(f'Lang: {test_lang} Val Precision:{precision} Recall: {recall} F1: {f1}\n')


  precision, recall, f1, val_loss, val_acc = validate_model(testDataloader)

  logprint(f"overall Val Loss: {val_loss}: Val Acc: {val_acc}\n")
  logprint(f'overall Val Precision:{precision} Recall: {recall} F1: {f1}\n')

  if f1 > best_val_f1:
    best_val_f1 = f1 

    logprint(f'best epoch: {epoch_i+1} best overall f1: {f1}\n')

    save_dict = {
      'best_model_wts' : model.state_dict(),
      'best_f1' : best_val_f1,
      'precision' : precision,
      'recall' : recall 
    }

    save_model_path = os.path.join(args.logs_dir, f'xlm_{args.language}.pth')

    torch.save(save_dict, save_model_path)
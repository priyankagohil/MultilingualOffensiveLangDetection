import torch
import os, sys, argparse
import numpy as np 
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, XLMRobertaForSequenceClassification 
from sklearn.metrics import f1_score, precision_score, recall_score
import googletrans
from googletrans import Translator
import pandas as pd 
from data_utils import OLID 

parser = argparse.ArgumentParser(description='XLM-R')
parser.add_argument("--logs_dir", default='english', type=str)

parser.add_argument("--language", default='english', type=str)
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_name = "xlm-roberta-base" 

# tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokens = tokenizer.tokenize('羅伯特 · 皮爾 斯 生於 1863年 , 在 英國 曼徹斯特 學習 而 成為 一 位 工程師 . 1933年 , 皮爾斯 在 直布羅陀去世 .')

model = XLMRobertaForSequenceClassification.from_pretrained(model_name,
                                  num_labels = 2, # The number of output labels.   
                                  output_attentions = False, # Whether the model returns attentions weights.
                                  output_hidden_states = False, # Whether the model returns all hidden-states.
                                  )

model.cuda()


translation_keys = {'arabic' : 'ar', 'danish' : 'da', 'english' : 'en', 'greek' : 'el', 'turkish' : 'tr'}
translator = Translator()

ckpt_paths = {'lang-specific':'models/{}/xlm_{}.pth',
                'cross-lingual' : 'models/all/xlm_all.pth', 
                'solid_olid': 'models/solid_olid/xlm_all.pth', 
                'olid_solid_off' : 'models/olid_on_solid_off/xlm_all.pth'
                }

langs = ['arabic', 'danish', 'english', 'greek', 'turkish']

langs = ['english']
langs = ['arabic', 'danish', 'greek', 'turkish']

for lang in langs:
    
    df = pd.DataFrame(columns=['tweet', 'label', 'lang-specific', 'cross-lingual', 'olid_solid_off', 'solid_olid'])

    dataset = OLID('data', 'test', lang)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

    sents = []
    label = []
    preds = {'lang-specific' :[], 'cross-lingual': [], 'olid_solid_off': [], 'solid_olid': []}
    translations = []

    for idx, item in enumerate(dataloader):

        inputs = item['input']
        labels = item['label']

        input_ids = tokenizer(inputs, padding=True, truncation=True, max_length=512, return_tensors='pt')
        model_input = input_ids['input_ids'].cuda()
        attention_masks = input_ids['attention_mask'].cuda()
        labels = labels.to(device) 

        for key, path in ckpt_paths.items():

            model_path = path 
            if key == 'lang-specific':
                model_path = model_path.format(lang, lang)

            # print('model_path:', model_path)

            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['best_model_wts'])
            del checkpoint

            with torch.no_grad():
                outputs = model(model_input, 
                                token_type_ids=None, 
                                attention_mask=attention_masks, 
                                labels=labels)
            
            logits = outputs.logits
            batch_preds = torch.argmax(logits, dim=1)


            

            preds[key].extend(list(batch_preds.detach().cpu().numpy()))

        sents.extend(inputs)
        label.extend(list(labels.detach().cpu().numpy()))
        if lang != 'english':
            for s in inputs:
                translation = translator.translate(s, src=translation_keys[lang], dest='en')
                out = translation.text 
                translations.append(out)

    df['tweet'] = sents 
    df['label'] = label
    for key in preds.keys(): 
        df[key] = preds[key]

    if lang != 'english':
        df['translated'] = translations

    df.to_csv(f'error_analysis_{lang}.csv') 

        
        




# def validate_model(dataloader):
#   model.eval()

#   running_loss = 0.
#   matches = 0.
#   samples = 0.
#   running_labels = []
#   running_preds = []

#   for idx, item in enumerate(dataloader):

#     inputs = item['input']
#     labels = item['label']

#     input_ids = tokenizer(inputs, padding=True, truncation=True, max_length=512, return_tensors='pt')

#     model_input = input_ids['input_ids'].cuda()
#     attention_masks = input_ids['attention_mask'].cuda()
#     labels = labels.to(device) 

#     with torch.no_grad():
#       outputs = model(model_input, 
#                       token_type_ids=None, 
#                       attention_mask=attention_masks, 
#                       labels=labels)
#       loss = outputs.loss

#     logits = outputs.logits
#     running_loss += loss.item()

#     preds = torch.argmax(logits, dim=1)

#     running_labels.extend( list(labels.detach().cpu().numpy()) )
#     running_preds.extend( list(preds.detach().cpu().numpy()) )

#     running_loss += loss.item()

#     matches += torch.sum(preds == labels).item()
#     samples += labels.shape[0]

#     torch.cuda.empty_cache()

#     # break

#   # print(running_labels)

#   precision = precision_score(running_labels, running_preds)
#   recall = recall_score(running_labels, running_preds)
#   f1 = f1_score(running_labels, running_preds)

#   running_loss /= len(dataloader)
#   running_acc = matches / samples 

#   return precision, recall, f1, running_loss, running_acc

# dataroot = 'data/'
# langs = ['english', 'danish', 'turkish', 'arabic', 'greek']
# batch_size = args.batch_size

# for lang in langs:

#     logprint('==============================\n')
#     logprint(f'XLM-R finetuned on - {lang}\n')
#     logprint('==============================\n')

#     model_path = f'models/{lang}/xlm_{lang}.pth'
#     checkpoint = torch.load(model_path)
#     model.load_state_dict(checkpoint['best_model_wts'])

#     for ev_lang in langs:

#         if ev_lang == lang:
#             continue

#         logprint(f'Zero-Shot eval on {ev_lang}\n')

#         dataset = OLID(dataroot, 'test', ev_lang)
#         testDataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#         precision, recall, f1, val_loss, val_acc = validate_model(testDataloader)

#         logprint(f"Val Loss: {val_loss}: Val Acc: {val_acc}\n")
#         logprint(f'Val Precision:{precision} Recall: {recall} F1: {f1}\n')


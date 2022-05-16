from torch.utils import data
import pandas as pd
import os, argparse, random
import torch, sys
from torch.utils.data import Dataset, DataLoader

langs = ['arabic', 'danish', 'english', 'greek', 'turkish']
dataroot = 'data/'

class OLID(Dataset):
  def __init__(self, dataroot, mode='train', lang='english'):
    if lang!='all' and lang not in langs:
      print(f'language {lang} not supported. Supported all_langs + {",".join(langs)}')
      exit() 
    if mode not in ['train', 'test']:
      print(f'mode {mode} not supported. Supported train, test')
      exit() 

    self.dataroot = dataroot 
    self.mode = mode 
    self.lang = lang 

    self.sents = []

    if lang=='all':

      for lang in langs:
        tsv_anno = os.path.join(self.dataroot, lang, f'{mode}.tsv')
        self.df = pd.read_csv(tsv_anno, sep='\t')
        self.df = self.df[['tweet', 'subtask_a']]
        self.df = self.df.dropna()
        self.df['tweet'] = self.df['tweet'].map(lambda x : x.replace('@USER', ''))

        self.sents.extend(self.df.values.tolist())
    
    else:
      tsv_anno = os.path.join(self.dataroot, lang, f'{mode}.tsv')
      self.df = pd.read_csv(tsv_anno, sep='\t')
      self.df = self.df[['tweet', 'subtask_a']]
      self.df = self.df.dropna()
      self.df['tweet'] = self.df['tweet'].map(lambda x : x.replace('@USER', ''))
      
      self.sents.extend(self.df.values.tolist())

    random.shuffle(self.sents)

  def __len__(self):
    # return self.df.shape[0]
    return len(self.sents)

  def __getitem__(self, idx):
    # row = self.df.iloc[idx] 
    sent = self.sents[idx] 
    input = sent[0]
    label = sent[1]
    if label == 'OFF':
      label = 1
    else:
      label = 0 
    return {'input' : input, 'label' : label}

class OLIDUniformSample(Dataset):
  def __init__(self, dataroot, mode='train'):

    self.dataroot = dataroot 
    self.mode = mode 

    self.df = {}
    self.langs = ['arabic', 'danish', 'english', 'greek', 'turkish']
    self.categories = ['OFF', 'NOT']

    for lang in self.langs:
      tsv_anno = os.path.join(self.dataroot, lang, f'{mode}.tsv')
      ldf = pd.read_csv(tsv_anno, sep='\t')
      ldf = ldf[['tweet', 'subtask_a']]
      ldf = ldf.dropna()
      self.df[lang] = ldf


  def __len__(self):
    l = 0
    for lang in self.langs:
      l += len(self.df[lang].index)
    return l

  def __getitem__(self, idx):

    sample_lang = random.choice(self.langs)
    sample_cat = random.choice(self.categories)  

    df = self.df[sample_lang]

    df_subsample = df[df['subtask_a'] == sample_cat] 
    sent_samples = df_subsample.values.tolist()

    sent = random.choice(sent_samples)
    
    input = sent[0]
    label = sent[1]

    # print(f'lang:{sample_lang} cat:{sample_cat} sent:{input}')

    if label == 'OFF':
      label = 1
    else:
      label = 0 
    return {'input' : input, 'label' : label}

import googletrans
from googletrans import Translator
import numpy as np 

def process_solid(solid):
  solid.columns = ['id', 'tweet', 'score']
  solid = solid[['tweet', 'score']]
  solid = solid.astype({'score': 'float'})
  labels = ['OFF' for _ in range(len(solid.index))]
  solid['subtask_a'] = labels
  solid.loc[solid['score'] < 0.7, 'subtask_a'] = 'NOT'
  solid.loc[solid['score'] >= 0.7, 'subtask_a'] = 'OFF'
  return solid

class OLIDSOLID(Dataset):
  def __init__(self, dataroot, mode='train'):

    self.dataroot = dataroot 
    self.mode = mode 

    self.df = {}
    self.langs = ['arabic', 'danish', 'english', 'greek', 'turkish']
    self.categories = ['OFF', 'NOT']
    self.datasets = ['olid', 'solid']

    self.sents = []

    for lang in self.langs:
      tsv_anno = os.path.join(self.dataroot, lang, f'{mode}.tsv')
      ldf = pd.read_csv(tsv_anno, sep='\t')
      ldf = ldf[['tweet', 'subtask_a']]
      ldf = ldf.dropna()
      ldf['tweet'] = ldf['tweet'].map(lambda x : x.replace('@USER', ''))
      self.df[lang] = ldf

      self.sents.extend(ldf.values.tolist())

    tsv_anno = os.path.join(self.dataroot, 'solid/solid_offensive_only.tsv')
    self.solid = pd.read_csv(tsv_anno, sep='\t')

    for lang in self.langs:

      df = self.solid[[lang, 'label']]
      self.sents.extend(df.values.tolist())
    
    random.shuffle(self.sents)

  def __len__(self):
    return len(self.sents)

  def __getitem__(self, idx):

    sent = self.sents[idx] 
    input = sent[0]
    label = sent[1]

    if label == 'OFF':
      label = 1
    else:
      label = 0 
    return {'input' : input, 'label' : label}

# dataset = OLIDSOLID(dataroot)
# trainDataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# for idx, item in enumerate(trainDataloader):
#   inputs = item['input']
#   labels = item['label']
#   print('inputs:',inputs)
#   print('labels:',labels)
#   break

def translate_solid(dataroot):
  tsv_anno = os.path.join(dataroot, 'solid/SOLID_tweets.tsv')
  solid = pd.read_csv(tsv_anno, sep='\t')
  solid = process_solid(solid)
  solid['tweet'] = solid['tweet'].map(lambda x : x.replace('@USER', ''))

  df_off = solid[solid['subtask_a'] == 'OFF']
  df_not = solid[solid['subtask_a'] == 'NOT']

  print('off:', len(df_off.index))
  print('not:', len(df_not.index))

  translation_keys = {'arabic' : 'ar', 'danish' : 'da', 'english' : 'en', 'greek' : 'el', 'turkish' : 'tr'}
  translator = Translator()

  columns = [lang for lang in langs]
  columns.append('label')
  
  df = pd.DataFrame(columns = langs)

  entry = {lang: '' for lang in langs}
  entry = {'label' : ''} 

  off_labels = 0
  not_labels = 0

  # for idx, row in solid.iterrows():
  for idx, row in df_off.iterrows():
    # print('off:', off_labels, 'not:', not_labels)
    print('idx:', idx)

    tweet = row['tweet']
    label = row['subtask_a']

    # if label == 'OFF' and off_labels < 1000:
    #   off_labels += 1
    # elif label == 'NOT' and not_labels < 1000:
    #   not_labels += 1
    # elif off_labels == 1000 and not_labels == 1000:
    #   break
    # else:
    #   continue
    
    entry['label'] = label

    for lang in langs:
      if lang == 'english':
        output = tweet
        entry[lang] = output
      else:
        key = translation_keys[lang]
        # print('key:', key)
        translation = translator.translate(tweet, src='en', dest=key)
        output = translation.text
        entry[lang] = output
      # print(lang, output)

    df = df.append(entry, ignore_index=True)

  # df.to_csv('data/solid/solid_translated.tsv', sep='\t')
  df.to_csv('data/solid/solid_offensive_only.tsv', sep='\t')

# translate_solid(dataroot)

class SOLID(Dataset):
  def __init__(self, dataroot):

    self.dataroot = dataroot 

    self.df = {}
    self.langs = ['arabic', 'danish', 'english', 'greek', 'turkish']
    self.categories = ['OFF', 'NOT']

    self.sents = []

    tsv_anno = os.path.join(self.dataroot, 'solid/solid_translated.tsv')
    self.solid = pd.read_csv(tsv_anno, sep='\t')

    for lang in self.langs:
      df = self.solid[[lang, 'label']]
      self.sents.extend(df.values.tolist())
    
    random.shuffle(self.sents)

  def __len__(self):
    return len(self.sents)

  def __getitem__(self, idx):

    sent = self.sents[idx] 
    input = sent[0]
    label = sent[1]

    if label == 'OFF':
      label = 1
    else:
      label = 0 
    return {'input' : input, 'label' : label}

# dataset = SOLID(dataroot)
# trainDataloader = DataLoader(dataset, batch_size=64, shuffle=True)
# tsv_anno = os.path.join(dataroot, 'solid/solid_offensive_only.tsv')
# solid = pd.read_csv(tsv_anno, sep='\t')

# solid = solid[['english', 'label']]
# print(solid.head())
# print(f'#tweets = {len(solid.index)}')

# solid_off = solid[solid['label']=='OFF']
# print(f'#OFF = {len(solid_off.index)}')

# solid_not = solid[solid['label']=='NOT']
# print(f'#NOT = {len(solid_not.index)}')

# for idx, item in enumerate(trainDataloader):
#   inputs = item['input']
#   labels = item['label']
#   print('inputs:',inputs)
#   print('labels:',labels)
#   break

# parser = argparse.ArgumentParser(description='XLM-R')
# parser.add_argument("--lang", default='english', type=str)
# args = parser.parse_args()

# dataset = OLID(dataroot, 'train', args.lang)
# trainDataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# dataset = OLID(dataroot, 'test', args.lang)
# testDataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# dataset = OLIDUniformSample(dataroot, 'train')
# trainDataloader = DataLoader(dataset, batch_size=16, shuffle=True)


# for idx, item in enumerate(trainDataloader):
#   inputs = item['input']
#   labels = item['label']
#   print('inputs:',inputs)
#   print('labels:',labels)
#   break


# for lang in langs:

#   tsv_anno = os.path.join(dataroot, lang, 'train.tsv')
#   df = pd.read_csv(tsv_anno, sep='\t')

#   # print(f'lang: {lang} train: {len(df.index)}')

#   df_off = df[df['subtask_a'] == 'OFF']
#   print(f'lang: {lang} train OFF: {len(df_off.index)}')

#   df_not = df[df['subtask_a'] == 'NOT']
#   print(f'lang : {lang}train NOT: {len(df_not.index)}')


#   tsv_anno = os.path.join(dataroot, lang, 'test.tsv')
#   df2 = pd.read_csv(tsv_anno, sep='\t')

#   df_off = df2[df2['subtask_a'] == 'OFF']
#   print(f'lang: {lang} val OFF: {len(df_off.index)}')

#   df_not = df2[df2['subtask_a'] == 'NOT']
#   print(f'lang: {lang} NOT: {len(df_not.index)}')

  # print(f'test: {len(df2.index)}')

# df = df.append(df2)
# print(f'all: {len(df.index)}')

# df_off = df[df['subtask_a'] == 'OFF']
# print(f'OFF: {len(df_off.index)}')

# df_not = df[df['subtask_a'] == 'NOT']
# print(f'NOT: {len(df_not.index)}')

# total_len = len(df.index)
# off_len = len(df_off.index)
# not_len = len(df_not.index)

# train_off = int(off_len * 0.9)
# train_not = int(not_len * 0.9)

# df_train = df_off[:train_off]
# df_val = df_off[train_off:]

# df_train = df_train.append(df_not[:train_not])
# df_val = df_val.append(df_not[train_not:])

# l1 = len(df_train.index)
# l2 = len(df_val.index)

# print(f'l1: {l1} l2:{l2}')

# df_off = df_train[df_train['subtask_a'] == 'OFF']
# print(f'train OFF: {len(df_off.index)}')

# df_not = df_train[df_train['subtask_a'] == 'NOT']
# print(f'train NOT: {len(df_not.index)}')

# df_off = df_val[df_val['subtask_a'] == 'OFF']
# print(f'val OFF: {len(df_off.index)}')

# df_not = df_val[df_val['subtask_a'] == 'NOT']
# print(f'val NOT: {len(df_not.index)}')

# tsv_anno = os.path.join(dataroot, args.lang, 'train2.tsv')
# df_train.to_csv(tsv_anno, sep='\t')

# print(f'train: {len(df.index)}')

# tsv_anno = os.path.join(dataroot, args.lang, 'test2.tsv')
# df_val.to_csv(tsv_anno, sep='\t')

# dataset = OLID(dataroot, 'train', args.lang)
# trainDataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# dataset = OLID(dataroot, 'test', args.lang)
# testDataloader = DataLoader(dataset, batch_size=64, shuffle=True)

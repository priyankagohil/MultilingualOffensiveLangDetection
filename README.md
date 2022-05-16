# Multilingual Offensive Language Detection

In this work, we performed the following three
tasks:
### Task A: Offensive language identification:

Identifying whether the text is offensive or in-offensive (done for all 5 languages).

### Task B: Categorization of offensive language: 

Identifying whether the offensive text is targeted or untargeted. (English only)

### Task C: Offensive language target identification: 

Identifying whether the targeted offensive language is targeting a group, individual, or others. (English only)


## 1. Pipeline-based model: 
a language classifier (BERT base (uncased)) is trained to detect the language of the tweet and then, that tweet will be fed to a BERT model that is pre-trained on that language only.

There are five different BERT models for each of the languages:
1. English
2. Danish
3. Turkish
4. Arabic
5. Greek

One advantage of this method is that since, the BERT is pre-trained on a specific language, it is better able to process nuances associated with
that language. However, a single BERT can have around 100M parameters and keeping separate BERT models for each language is not scalable.

## 2. cross-lingual model : XLM-RoBERTa
To address the scalability issue, a cross-lingual model, called XLM-RoBERTa is implemented. This single model is capable of taking multilingual inputs. 
This XLM-R is trained on all languages together, and the results are compared with the pipeline-based model.




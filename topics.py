import math
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def sigmoid(x):
  return 1 / (1 + math.exp(-x))
  
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-dec2021-tweet-topic-multi-all")
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-dec2021-tweet-topic-multi-all", problem_type="multi_label_classification")
model.eval()
class_mapping = model.config.id2label

def extract_topic(sentence):
    with torch.no_grad():
        text = sentence
        tokens = tokenizer(text, return_tensors='pt')
        output = model(**tokens)
        flags = [sigmoid(s) > 0.5 for s in output[0][0].detach().tolist()]
        topic = [class_mapping[n] for n, i in enumerate(flags) if i]
    return(topic)

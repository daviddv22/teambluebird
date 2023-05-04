   
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, AutoModelForSequenceClassification, \
BertModel

from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from preprocess import split_dataset
import torch
from collections import OrderedDict
import torch.nn as nn
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np


def sentiment_scores(dataset):

    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

    model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    toreturn = []
    for i, sentence in enumerate(tqdm(dataset)):
        inputs = tokenizer(sentence, return_tensors="pt")
        outputs = model(**inputs)
        predicted_class = torch.argmax(outputs.logits)
        toreturn.append(predicted_class)

    return torch.tensor(toreturn)

def train_embedding_generator(dataset):
    # gets the tokenizer and pretrained model
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    toreturn = []
    for i, sentence in enumerate(tqdm(dataset)):
        toreturn.append(model.encode(sentence))
        
    return toreturn

def formality_loss(dataset):
    # gets the tokenizer and pretrained model
    tokenizer = AutoTokenizer.from_pretrained("s-nlp/roberta-base-formality-ranker")
    model = AutoModelForSequenceClassification.from_pretrained("s-nlp/roberta-base-formality-ranker")
    
    # gets the formality score for the sentence
    toreturn = []
    for i, sentence in enumerate(tqdm(dataset)):
        # print(i)
        inputs = tokenizer(sentence, return_tensors="pt")

        with torch.no_grad():
            logits = model(**inputs).logits

        predicted_class_id = logits.argmax().item()
        score = torch.nn.functional.softmax(logits, dim=1)[0, predicted_class_id].item()
        toreturn.append(score)
    return torch.tensor(toreturn)


dataset, train_x, train_y, test_x, test_y, raw_x, raw_y, raw_test_x, raw_test_y = split_dataset()

labels = list(map(lambda x: float(1) if x == float(1) or x == float(2) else float(0), list(map(float, raw_y))))
labels = torch.tensor(labels).reshape(len(labels), 1)

test_labels = list(map(lambda x: float(1) if x == float(1) or x == float(2) else float(0), list(map(float, raw_test_y))))
test_labels = torch.tensor(test_labels).reshape(len(test_labels), 1)

text = "After stealing money from the bank vault, the bank robber was seen " \
       "fishing on the Mississippi river bank."
# embedding = train_embedding_generator(raw_x)
# formality_score = formality_loss(raw_x) # Get from RoBERTa Model
# sentiment_score = sentiment_scores(raw_x)

# torch.save(sentiment_score, 'sentiment_score.pt')

# #  save the embedding and formality score
# torch.save(embedding, 'embedding.pt')
# torch.save(formality_score, 'formality_score.pt')

embedding = torch.load('embedding.pt')
formality_score = torch.load('formality_score.pt')
sentiment_score = torch.load('sentiment_score.pt')
# input_dim = len(embedding[0]) + formality_score.shape[0] + sentiment_score.shape[0]
                            
                            #385 #769

# print(sum(labels)/len(labels))  

labels = 1 - labels
test_labels = 1 - test_labels

# Accuracy of the model on the test images: 98.8557516737675 %
# Accuracy of the model on the test images: 53.79746835443038 %

# testing 
# test_embedding = train_embedding_generator(raw_test_x)
# test_formality_score = formality_loss(raw_test_x) # Get from RoBERTa Model

# labels = labels.detach().numpy().reshape(labels.shape[0], 1)
# formality_score = formality_score.detach().numpy().reshape(labels.shape[0], 1)

# plt.scatter(formality_score[labels == 1], labels[labels == 1], c='r')
# plt.scatter(formality_score[labels == 0], labels[labels == 0], c='b')
# plt.xlabel('Formality Score')
# plt.ylabel('Labels')
# plt.title('Formality Score vs Labels')
# plt.show()

# test_sentiment_score = sentiment_scores(raw_test_x)

# torch.save(test_sentiment_score, 'test_sentiment_score.pt')
# save the embedding and formality score
# torch.save(test_embedding, 'test_embedding.pt')
# torch.save(test_formality_score, 'test_formality_score.pt')

# load the embedding and formality score
test_embedding = torch.load('test_embedding.pt')
test_formality_score = torch.load('test_formality_score.pt')
test_sentiment_score = torch.load('test_sentiment_score.pt')


# reshape formality score
formality_score = formality_score.reshape(formality_score.shape[0], 1)
sentiment_score = sentiment_score.reshape(sentiment_score.shape[0], 1)
model_ip = torch.cat((torch.tensor(embedding) , formality_score), 1)
print(model_ip.shape)
model_ip = torch.cat((model_ip, sentiment_score), 1)
print(model_ip.shape)

# reshape formality score
test_formality_score = test_formality_score.reshape(test_formality_score.shape[0], 1)
test_sentiment_score = test_sentiment_score.reshape(test_sentiment_score.shape[0], 1)
test_model_ip = torch.cat((torch.tensor(test_embedding) , test_formality_score), 1)
test_model_ip = torch.cat((test_model_ip, test_sentiment_score), 1)

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.SiLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim1)
        self.relu2 = nn.SiLU()
        self.maxpool = nn.MaxPool1d(2)
        self.layernorm = nn.LayerNorm(hidden_dim1)
        self.fc3 = nn.Linear(hidden_dim1, hidden_dim2)
        self.dropout = nn.Dropout(0.2)
        self.relu3 = nn.SiLU()
        self.fc4 = nn.Linear(hidden_dim2, hidden_dim3)
        self.relu4 = nn.SiLU()
        self.fc5 = nn.Linear(hidden_dim3, output_dim)
        self.dropout2 = nn.Dropout(0.3)
        self.globalavgpool = nn.AdaptiveAvgPool1d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        # out = out.permute(0, 2, 1)
        # out = self.maxpool(out)
        # out = out.permute(0, 2, 1)
        out = self.layernorm(out)
        out = self.fc3(out)
        out = self.dropout(out)
        out = self.relu3(out)
        out = self.fc4(out)
        out = self.relu4(out)
        out = self.fc5(out)
        out = self.dropout2(out)
        # out = self.globalavgpool(out)
        # out = out.squeeze()
        out = self.sigmoid(out)
        return out

# Define your model hyperparameters
input_dim = 386
hidden_dim = 256
hidden_dim1 = 128
hidden_dim2 = 64
hidden_dim3 = 32
output_dim = 1

# Define your model and loss function
model = MLP(input_dim, hidden_dim, output_dim)
criterion = nn.BCELoss()

# Define your optimizer


num_epochs = 50
decayRate = 2.3
inital_learning_rate = 0.02

# for epoch in range(num_epochs):
#     learning_rate = (1/(1+decayRate*epoch))*inital_learning_rate
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#     for i, data in tqdm(enumerate(model_ip)):
#         inputs = data
#         label = labels[i]
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, label)
#         loss.backward()
#         optimizer.step()

# # save the model
# torch.save(model.state_dict(), 'model.ckpt2')

# load the model
model = MLP(input_dim, hidden_dim, output_dim)
model.load_state_dict(torch.load('model.ckpt2'))

# get the accuracy
results = []
with torch.no_grad():
    correct = 0
    total = 0
    for i, data in enumerate(test_model_ip):
        inputs = data
        label = test_labels[i]
        outputs = model(inputs)
        predicted = torch.round(outputs.data)
        results.append([i, label, predicted])
        correct += (predicted == label).sum().item()
    print('Accuracy of the model on the test images: {} %'.format(100 * correct / test_labels.size(0)))

for item in results[:10]:
    line = raw_test_x[item[0]]
    correct = 'Correct' if item[1] == item[2] else 'Incorrect'
    toprint = raw_test_x[item[0]] + '\t' + correct + '\t'
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

    inputs = tokenizer(line, return_tensors="pt")
    outputs = model(**inputs)
    predicted_class = torch.argmax(outputs.logits)
    if predicted_class == 0:
        toprint += 'Negative'
    elif predicted_class == 1:
        toprint += 'Neutral'
    elif predicted_class == 2:
        toprint += 'Positive'

    tokenizer = AutoTokenizer.from_pretrained("s-nlp/roberta-base-formality-ranker")
    model = AutoModelForSequenceClassification.from_pretrained("s-nlp/roberta-base-formality-ranker")
    
    inputs = tokenizer(line, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_class_id = logits.argmax().item()
    
    if predicted_class_id == 0:
        toprint += ' Informal'
    elif predicted_class_id == 1:
        toprint += ' Formal'

    print(toprint)
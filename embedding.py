   
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, AutoModelForSequenceClassification, \
BertModel
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from preprocess import split_dataset
import torch
from collections import OrderedDict
import torch.nn as nn
# import loading bar from tqdm
from tqdm import tqdm

# cache the results of the function
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


dataset, train_x, train_y, test_x, test_y, raw_x, raw_y = split_dataset()

labels = list(map(lambda x: float(1) if x == float(1) or x == float(2) else float(0), list(map(float, raw_y))))
labels = torch.tensor(labels).reshape(len(labels), 1)

text = "After stealing money from the bank vault, the bank robber was seen " \
       "fishing on the Mississippi river bank."
# embedding = train_embedding_generator(raw_x)
# formality_score = formality_loss(raw_x) # Get from RoBERTa Model

# #  save the embedding and formality score
# torch.save(embedding, 'embedding.pt')
# torch.save(formality_score, 'formality_score.pt')

# load the embedding and formality score
embedding = torch.load('embedding.pt')
formality_score = torch.load('formality_score.pt')

# reshape formality score
formality_score = formality_score.reshape(formality_score.shape[0], 1)
# print(embedding)
model_ip = torch.cat((torch.tensor(embedding) , formality_score), 1)







# this doesnt work
model = nn.Sequential(
    nn.Linear(385, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 16),
    nn.ReLU(),
    nn.Linear(16, 1),
    nn.Softmax(1)
)   
# mlp = nn.Sequential(OrderedDict([
#     ('dense1', nn.Linear(385, 256)),
#     ('act1', nn.ReLU()),
#     ('dense2', nn.Linear(256, 128)),
#     ('act2', nn.ReLU()),
#     ('dense4', nn.Linear(128, 64)),
#     ('act4', nn.ReLU()),
#     ('dense3', nn.Linear(64, 16)),
#     ('act3', nn.ReLU()),
#     ('output', nn.Linear(16, 1)),
#     ('outact', nn.Softmax(1)),
# ]))

# train the model
num_epochs = 100
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
for n in tqdm(range(num_epochs)):
    y_pred = model(model_ip)
    loss = loss_fn(y_pred, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("Epoch: ", n, "Loss: ", loss.item())

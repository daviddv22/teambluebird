   
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


def train_embedding_generator(dataset):
    # gets the tokenizer and pretrained model
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')


    # for fine tuning if we decide to do so
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    # Train the BERT model for Embeddings
    # training_args = TrainingArguments(
    #     output_dir="./bert-base-uncased",
    #     overwrite_output_dir=True,
    #     num_train_epochs=1,
    #     per_device_train_batch_size=64,
    #     save_steps=10_000,
    #     save_total_limit=2,
    #     # batch_size=64,
    # )

    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     data_collator=data_collator,
    #     train_dataset=dataset,
    # )

    # trainer.train()
    # text = "After stealing money from the bank vault, the bank robber was seen " \
    #    "fishing on the Mississippi river bank."

    # Add the special tokens.
    # marked_text = "[CLS] " + text + " [SEP]"

    # gets the embeddings for the sentence (work in progress)
    toreturn = []
    for sentence in dataset:
        toreturn.append(model.encode(sentence))
        # marked_text = "[CLS] " + sentence + " [SEP]"
        # tokenized_text = tokenizer.tokenize(marked_text)
        # indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        # segments_ids = [1] * len(tokenized_text)
        # tokens_tensor = torch.tensor([indexed_tokens])
        # segments_tensors = torch.tensor([segments_ids])

        # with torch.no_grad():
        #     outputs = model(tokens_tensor, segments_tensors)
        #     hidden_states = outputs[2]
        #     # layer_i = 0
        #     # batch_i = 0
        #     # token_i = 0

        # token_vecs = hidden_states[-2][0]
        # token_embeddings = torch.stack(hidden_states, dim=0)
        # token_embeddings = torch.squeeze(token_embeddings, dim=1)
        # token_embeddings = token_embeddings.permute(1,0,2)

        # sentence_embedding = torch.mean(token_vecs, dim=0).numpy()
        # toreturn.append(sentence_embedding)

    print(toreturn[0:2])
    return toreturn

def formality_loss(dataset):
    # gets the tokenizer and pretrained model
    tokenizer = AutoTokenizer.from_pretrained("s-nlp/roberta-base-formality-ranker")
    model = AutoModelForSequenceClassification.from_pretrained("s-nlp/roberta-base-formality-ranker")
    
    # gets the formality score for the sentence
    toreturn = []
    for i, sentence in enumerate(dataset):
        # print(i)
        inputs = tokenizer(sentence, return_tensors="pt")

        with torch.no_grad():
            logits = model(**inputs).logits

        predicted_class_id = logits.argmax().item()
        score = torch.nn.functional.softmax(logits, dim=1)[0, predicted_class_id].item()
        toreturn.append(score)
    return torch.tensor(toreturn)


# data = load_dataset("liar")
dataset, train_x, train_y, test_x, test_y, raw_x, raw_y = split_dataset()
# print(data[0])
text = "After stealing money from the bank vault, the bank robber was seen " \
       "fishing on the Mississippi river bank."
embedding = train_embedding_generator(raw_x[:100])
formality_score = formality_loss(raw_x[:100]) # Get from RoBERTa Model
# reshape formality score
formality_score = formality_score.reshape(100, 1)
# print(embedding)
model_ip = torch.cat((torch.tensor(embedding) , formality_score), 1)







# this doesnt work
mlp = nn.Sequential(OrderedDict([
    ('dense1', nn.Linear(385, 256)),
    ('act1', nn.ReLU()),
    ('dense2', nn.Linear(256, 128)),
    ('act2', nn.ReLU()),
    ('dense4', nn.Linear(128, 64)),
    ('act4', nn.ReLU()),
    ('dense3', nn.Linear(64, 16)),
    ('act3', nn.ReLU()),
    ('output', nn.Linear(16, 1)),
    ('outact', nn.Softmax(1)),
]))

loss_fn = nn.CrossEntropyLoss()
# loss = loss_fn(model_ip, torch.tensor([0]))

optimizer = torch.optim.Adam(mlp.parameters(), lr=0.05)

num_epochs = 100
for n in range(num_epochs):
    y_pred = mlp(model_ip)
    # print(y_pred.shape)
    labels = list(map(lambda x: float(1) if x == float(1) or x == float(2) else float(0), list(map(float, raw_y[:100]))))
    labels = torch.tensor(labels).reshape(100, 1)
    # for i in range(len(y_pred)):
    #     print(y_pred[i], labels[i])
    # print(y_pred)
    loss = loss_fn(y_pred, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("Epoch: ", n, "Loss: ", loss.item())

    # needs is a low bar, can still harm the future despite meeting our needs
    # how do we measure how off we are, what about remote places that we may like, if we continue to advance to be better off
    # we may rid the future of it. Economics like fudgibility
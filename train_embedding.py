   
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from sentence_transformers import SentenceTransformer
from preprocess import split_dataset
import torch
import torch.nn as nn
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
from tqdm import tqdm
from MLP import MLP


def sentiment_scores(dataset):
    """
    This function takes in a list of sentences and uses the twitter-roberta-base-sentiment pre-trained model
    from Hugging Face to predict the sentiment scores (positive, negative, or neutral) for each sentence
    
    Parameters:
    dataset (list): A list of sentences to analyze the sentiment of.
    
    Returns:
    torch.tensor: A tensor containing the predicted sentiment scores for each sentence in the dataset.
    """

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
    """
    generates sentence embeddings using a pretrained model called paraphrase-MiniLM-L6-v2.

    Args:

    dataset (list): A list of strings containing the sentences to be embedded.
    Returns:

    toreturn (list): A list of numpy arrays, where each array is a sentence embedding of the corresponding input sentence.
    """
    # gets the tokenizer and pretrained model
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    toreturn = []
    for i, sentence in enumerate(tqdm(dataset)):
        toreturn.append(model.encode(sentence))
        
    return toreturn

def formality_loss(dataset):
    """_summary_
    This function takes in a list of sentences and uses a pre-trained RoBERTa-based formality ranker model 
    from Hugging Face to calculate the formality score of each sentence

    Parameters:

    Args:
        dataset (List[str]): A list of strings where each string is a sentence. 

    Returns:
       a tensor containing formality scores for each of the input sentences.
    """
    # gets the tokenizer and pretrained model
    tokenizer = AutoTokenizer.from_pretrained("s-nlp/roberta-base-formality-ranker")
    model = AutoModelForSequenceClassification.from_pretrained("s-nlp/roberta-base-formality-ranker")
    
    # gets the formality score for the sentence
    toreturn = []
    for i, sentence in enumerate(tqdm(dataset)):
        inputs = tokenizer(sentence, return_tensors="pt")

        with torch.no_grad():
            logits = model(**inputs).logits

        predicted_class_id = logits.argmax().item()
        score = torch.nn.functional.softmax(logits, dim=1)[0, predicted_class_id].item()
        toreturn.append(score)
    return torch.tensor(toreturn)

# get our training and testing data
dataset, train_x, train_y, test_x, test_y, raw_x, raw_y, raw_test_x, raw_test_y = split_dataset()

# convert our labels into binary labels
labels = list(map(lambda x: float(1) if x == float(1) or x == float(2) else float(0), list(map(float, raw_y))))
labels = torch.tensor(labels).reshape(len(labels), 1)

test_labels = list(map(lambda x: float(1) if x == float(1) or x == float(2) else float(0), list(map(float, raw_test_y))))
test_labels = torch.tensor(test_labels).reshape(len(test_labels), 1)

# get the embedding and formality score for each sentence

# embedding = train_embedding_generator(raw_x)
# formality_score = formality_loss(raw_x) 
# sentiment_score = sentiment_scores(raw_x)

# save the embedding, formality score, and sentiment score

# torch.save(sentiment_score, 'sentiment_score.pt')
# torch.save(embedding, 'embedding.pt')
# torch.save(formality_score, 'formality_score.pt')

# load the embedding, formality score, and sentiment score
embedding = torch.load('embedding.pt')
formality_score = torch.load('formality_score.pt')
# sentiment_score = torch.load('sentiment_score.pt')

# normalize the sentiment score
# sentiment_score = (sentiment_score - sentiment_score.min())/(sentiment_score.max() - sentiment_score.min())

# flip the labels
labels = 1 - labels

# Accuracy of the model on the train images: 98.8557516737675 %
# Accuracy of the model on the test images: 60.79746835443038 %


# get the test embedding and formality score

# test_embedding = train_embedding_generator(raw_test_x)
# test_formality_score = formality_loss(raw_test_x) 
# test_sentiment_score = sentiment_scores(raw_test_x)

# plot the correlation between the formality score and the labels

# plt.scatter(formality_score[labels == 1], labels[labels == 1], c='r')
# plt.scatter(formality_score[labels == 0], labels[labels == 0], c='b')
# plt.xlabel('Formality Score')
# plt.ylabel('Labels')
# plt.title('Formality Score vs Labels')
# plt.show()

# save the test embedding and formality score
# torch.save(test_sentiment_score, 'test_sentiment_score.pt')
# torch.save(test_embedding, 'test_embedding.pt')
# torch.save(test_formality_score, 'test_formality_score.pt')


# reshape formality score
formality_score = formality_score.reshape(formality_score.shape[0], 1)
sentiment_score = sentiment_score.reshape(sentiment_score.shape[0], 1)
model_ip = torch.cat((torch.tensor(embedding) , formality_score), 1)

# add sentiment scores
# model_ip = torch.cat((model_ip, sentiment_score), 1)
# model_ip = torch.tensor(embedding)

input_dim = 385
hidden_dim = 256
hidden_dim1 = 128
hidden_dim2 = 64
hidden_dim3 = 32
output_dim = 1


def main():
    # Define your model and loss function
    model = MLP(input_dim, hidden_dim, hidden_dim1, hidden_dim2, hidden_dim3, output_dim)
    loss_function = nn.BCELoss()

    # set up learning rate decay calculation
    num_epochs = 25
    decayRate = 2.2
    inital_learning_rate = 0.025

    for epoch in range(num_epochs):
        learning_rate = (1/(1+decayRate*epoch))*inital_learning_rate
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        for i, data in tqdm(enumerate(model_ip)):
            inputs = data
            label = labels[i]
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, label)
            loss.backward()
            optimizer.step()

    # save the model
    torch.save(model.state_dict(), 'model.ckpt2')

if __name__ == '__main__':
    main()
### Project name: teambluebird

### Github link: https://github.com/daviddv22/teambluebird.git

## Team members (cs logins)

- Luoyun Wang (lwang105)
- David Doan(ddoan2)
- Indigo Funk (ifunk1)
- Muhiim Ali (mali37)

### Include the total estimated time it took to complete the project:

2 weeks

## Project purpose:

Being interested in natural language processing tasks and with the remaining prominence of fake news in the real world, we wanted to come up with a potential solution or tool to help distinguish real and fake news, specifically on Twitter. Intuitively, we would expect that there might be a pattern in the language used for real and fake news, and that there might exist a discrepancy between the two statements. dealing with the prominence of fake news in the real world. For our final project for Brown’s Deep Learning task, we wanted to test our theory and build a classification model to detect real news with a new heuristic function, the formality of the language used.
Our model is a supervised classification model, with news statements as our dataset and each statement corresponding to a truth value label. We then categorize our inputs into either fake or real based on the language used.

### Class functionalities

- `preprocess.py`: provides functionality for splitting the liar dataset into train and test sets, tokenizing the sentences, padding the tensors to the same length, and reshaping the tensors to be used in a model. The output is saved to a JSON file, and the function returns the processed dataset in addition to the train and test sets and their corresponding labels.
- `datasetClass.py`: represents a dataset of fake news examples that can be used for training and testing the model
- `embedding.py`: This class uses various Natural Language Processing (NLP) techniques and models to analyze the sentiment, formality and embeddings of text data. It uses transformers such as AutoTokenizer and AutoModelForSequenceClassification from the Hugging Face library to predict sentiment scores for each sentence in a given dataset. It also uses SentenceTransformer, which is a neural network-based model, to generate embeddings for each sentence. Additionally, it uses a pre-trained RoBERTa-based formality ranker model from Hugging Face to calculate the formality score of each sentence.
-

### Model Accuracy

- Embedding: 51.56%
- Embedding + sentiment + Formality: 54.9%
- Embeddig + formality: 60.43%

### Challenges

- Preprocessing was a bit of a challenge: we made a whole FakeNewsDataset class only to realize that we didn’t need it. Its functionality was handled for us in the pretrained models we downloaded and in the “datasets” package
  - In a similar way, we wrote an analogue to a train_step function which tokenized our data and then called our model, only to realize tokenization had to be handled in the preprocessing file, and that it was easier to run our “train_step” outside of a function as direct commands.
- We also thought about writing a separate method for our model class for encoding vs calling on the dense layers, but we realized encoding was also implemented in the BERT pretrained model.
- It took a bit of time to figure out the most efficient wrapper for the model—at first we tried to pass a dictionary into nn.Sequential, but since it wasn’t working we did without the names.

### Known Bugs

### Setup

Download the files below from https://huggingface.co/bert-base-uncased/tree/main and add them to your root directory

- pytorch_model.bin
- flax_model.msgpack
- tf_model.h5 from
  Then run pip3 install datasets

### How to run..

#### Tests

-

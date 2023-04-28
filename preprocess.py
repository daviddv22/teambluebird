from datasets import load_dataset
# import the tokenizer used in splitting the dataset
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from datasetClass import FakeNewsDataset
import json

# save the dataset to a file to be used in other files
# would be nice to save as something like a json file
def split_dataset():
    # load the dataset from the file
    
    dataset = load_dataset("liar")

    #select only the columns we need
    dataset = dataset.select_columns(["label", "statement"]) 

    # split the dataset into train and test manually
    data_split = dataset["train"].train_test_split(test_size=0.2, shuffle=True)

    # create a new dataset with the train and test datasets
    train_x = data_split["train"]["statement"]
    train_y_ful = data_split["train"]["label"]
    # label 0-1: 0 false 1 true
    train_y = [0 if label <= 2 else 1 for label in train_y_ful]
    
    test_x = data_split["test"]["statement"]
    test_y_ful = data_split["test"]["label"]
    test_y = [0 if label <= 2 else 1 for label in test_y_ful]

    # tokenize the sentences into tensors
    raw_x = train_x
    raw_y = train_y
    raw_test_x = test_x
    raw_test_y = test_y
    tokenizer = Tokenizer(num_words=100000, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_x)
    train_x = tokenizer.texts_to_sequences(train_x)
    test_x = tokenizer.texts_to_sequences(test_x)

    # pad the tensors to the same length
    train_x = pad_sequences(train_x, padding="post")
    test_x = pad_sequences(test_x, padding="post")

    # reshape the tensors to be used in the model
    train_x = train_x.reshape(train_x.shape[0], train_x.shape[1], 1)
    test_x = test_x.reshape(test_x.shape[0], test_x.shape[1], 1)

    # raw_text, train_y, test_x, test_y = split_dataset()
    # dataset = FakeNewsDataset(train_x, tokenizer)
     # save the data to a JSON file
    data = {
        "train_x": train_x.tolist(),
        "train_y": train_y,
        "test_x": test_x.tolist(),
        "test_y": test_y,
        "raw_x": raw_x,
        "raw_y": raw_y
    }
    with open("data.json", "w") as f:
        json.dump(data, f)

    # return the data
    return dataset, train_x, train_y, test_x, test_y, raw_x, raw_y, raw_test_x, raw_test_y




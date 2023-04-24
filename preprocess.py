from datasets import load_dataset
# import the tokenizer used in splitting the dataset
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from datasetClass import FakeNewsDataset

# save the dataset to a file to be used in other files
# would be nice to save as something like a json file
def split_dataset():
    # load the dataset from the file
    
    dataset = load_dataset("liar")

    #select only the columns we need
    dataset = dataset.select_columns(["label", "statement"]) 

    # split the dataset into train and test manually
    train = dataset["train"].train_test_split(test_size=0.2, shuffle=True)
    test = dataset["train"].train_test_split(test_size=0.2, shuffle=True)

    # create a new dataset with the train and test datasets
    train_x = train["train"]["statement"]
    train_y = train["train"]["label"]

    test_x = test["train"]["statement"]
    test_y = test["train"]["label"]

    # tokenize the sentences into tensors
    raw_x = train_x
    raw_y = train_y
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
    return dataset, train_x, train_y, raw_x, raw_y


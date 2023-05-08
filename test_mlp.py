
from MLP import MLP
import torch
from preprocess import split_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

input_dim = 385
hidden_dim = 256
hidden_dim1 = 128
hidden_dim2 = 64
hidden_dim3 = 32
output_dim = 1

dataset, train_x, train_y, test_x, test_y, raw_x, raw_y, raw_test_x, raw_test_y = split_dataset()

# convert our labels into binary labels
test_labels = list(map(lambda x: float(1) if x == float(1) or x == float(2) else float(0), list(map(float, raw_test_y))))
test_labels = torch.tensor(test_labels).reshape(len(test_labels), 1)
test_labels = 1 - test_labels

# load the embedding and formality score
test_embedding = torch.load('test_embedding.pt')
test_formality_score = torch.load('test_formality_score.pt')
test_sentiment_score = torch.load('test_sentiment_score.pt')

# reshape formality score
test_formality_score = test_formality_score.reshape(test_formality_score.shape[0], 1)
test_sentiment_score = test_sentiment_score.reshape(test_sentiment_score.shape[0], 1)
test_model_ip = torch.cat((torch.tensor(test_embedding) , test_formality_score), 1)

# add test sentiment scores
# test_model_ip = torch.cat((test_model_ip, test_sentiment_score), 1)
# test_model_ip = torch.tensor(test_embedding)

def print_results(results):
    # print the first 10 results
    for item in results[:10]:
        # print the line, whether it was correct or not, and the predicted label
        line = raw_test_x[item[0]]
        correct = 'Correct' if item[1] == item[2] else 'Incorrect'
        toprint = raw_test_x[item[0]] + '\t' + correct + '\t'
        # tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
        # model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

        # inputs = tokenizer(line, return_tensors="pt")
        # outputs = model(**inputs)
        # predicted_class = torch.argmax(outputs.logits)
        # if predicted_class == 0:
        #     toprint += 'Negative'
        # elif predicted_class == 1:
        #     toprint += 'Neutral'
        # elif predicted_class == 2:
        #     toprint += 'Positive'

        # print the formality score
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


def main():

    # load the model
    model = MLP(input_dim, hidden_dim, hidden_dim1, hidden_dim2, hidden_dim3, output_dim)
    model.load_state_dict(torch.load('model.ckpt2'))

    # get the accuracy
    results = []
    with torch.no_grad():
        correct = 0
        for i, data in enumerate(test_model_ip):
            inputs = data
            label = test_labels[i]
            outputs = model(inputs)
            predicted = torch.round(outputs.data)
            results.append([i, label, predicted])
            correct += (predicted == label).sum().item()
        print('Accuracy of the model on the test images: {} %'.format(100 * correct / test_labels.size(0)))
    
    # print the results
    # print_results(results)

if __name__ == '__main__':
    main()
import torch
# not needed I dont think
class FakeNewsDataset(torch.utils.data.Dataset):
    def __init__(self, raw_data, tokenizer):
        super(FakeNewsDataset, self).__init__()
        self.data = raw_data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_row = self.data[index]
        text = data_row['text']
        label = data_row['label']
        tokenized_text = self.tokenizer(text, return_tensors='pt')
        sequence = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        tensor_sequence = torch.tensor(sequence)
        return {'input_ids', tensor_sequence}
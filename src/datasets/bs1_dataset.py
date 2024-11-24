
# Source: https://github.com/eyalmazuz/MolGen/blob/master/MolGen/src/datasets/bs1_dataset.py

import pandas as pd
import torch
from torch.utils.data import Dataset



class BS1Dataset(Dataset):

    def __init__(self, data, tokenizer):

        self.data = data

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        smiles = self.data.loc[idx]['Smiles']
        label = self.data.loc[idx]['pChEMBL Value'] 

        smiles = '[CLS]' + smiles
        return smiles, label

    def collate_fn(self, batches):
        smiles = [b[0] for b in batches]
        labels = [b[1] for b in batches]
        labels = torch.tensor(labels, dtype=torch.float32)

        max_len = max(map(len, smiles))

        inputs = []
        masks = []
        for s in smiles:
            encodings = self.tokenizer(s, padding=True, max_length=max_len)
            inputs.append(encodings['input_ids'])
            masks.append(encodings['padding_mask'])

        inputs = torch.tensor(inputs, dtype=torch.long)
        masks = torch.tensor(masks, dtype=torch.long)
        encodings = {'input_ids': inputs, 'padding_mask': masks}

        return encodings, labels.view(-1, 1)

        

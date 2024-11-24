# Source : https://github.com/eyalmazuz/MolGen/blob/master/MolGen/src/datasets/smiles_dataset.py

from random import sample

from rdkit import Chem
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from  src.tokenizers import CharTokenizer


class SmilesDataset(Dataset):

    def __init__(self,
                 data_path,
                 tokenizer,
                 max_len=50):

        self.max_len = max_len
        self.data_path = data_path 
        self._molecules = self.load_molecules()
        self.tokenizer = tokenizer

    @property
    def molecules(self):
        return self._molecules
        
    def load_molecules(self):
        molecules = []
        with open(self.data_path, 'r') as f:
            molecules = f.readlines()
            molecules = [smiles.strip() for smiles in molecules]
        return molecules

    def __len__(self):
        return len(self._molecules)

    def __getitem__(self, idx):
        smiles = self._molecules[idx]
        
        smiles = '[BOS]' + smiles + '[EOS]'
        encodings = self.tokenizer(smiles, padding=True, max_length=self.max_len)
        encodings['labels'] = encodings['input_ids']

        encodings = {k: torch.tensor(v) for k, v in encodings.items()}
        return encodings
    
    def get_vocab_size(self):
        return self.tokenizer.vocab_size
    
    def get_block_size(self):
        return self.max_len

def main():

    dataset = SmilesDataset('/Users/veramazeeva/Library/CloudStorage/OneDrive-Personal/Carnegie Mellon/GenAI/project/code/data/gdb13/gdb13_rand1m.smi',
                                 '/Users/veramazeeva/Library/CloudStorage/OneDrive-Personal/Carnegie Mellon/GenAI/project/code/data/tokenizers/gdb13CharTokenizer.json')

    smiles = 'CCO'
    
    encoding = dataset.tokenizer.encode(smiles)
    print(encoding)
    rec_smiles = dataset.tokenizer.decode(encoding)
    print(rec_smiles)
    print(rec_smiles == smiles)

if __name__ == "__main__":
    main()

import argparse
from tqdm import tqdm
import os
import json

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-data_path', required=True, help='path to data file with smiles strings')
    parser.add_argument('-tokenizer_name', required=True, help='json filename for tokenizer')
    parser.add_argument('-tokenizer_path', default='./data/tokenizers', help='path to tokenizer json files')
    parser.add_argument('-type', default='char', help='type of tokenization (currently only char supported)') # TODO: BPE tokenizer
    
    args = parser.parse_args()

    with open(args.data_path, 'r') as f:
        molecules = f.readlines()
        molecules = [smiles.atrip() for smiles in molecules]

    tokens = set()

    print('Building tokenizer...')

    if args.type == 'char':
        for mol in molecules:
            tokens |= set(mol)
    
    id2token = {}
    for i, token in enumerate(tokens):
        id2token[i] = token
    
    print('Saving tokenizer...')
    
    with open(args.tokenizer_path, 'w') as f:
        json.dump(id2token, f)
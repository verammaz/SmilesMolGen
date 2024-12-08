import argparse
from tqdm import tqdm
import pandas as pd
import os
import json

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-data_path', required=True, help='path to data file with smiles strings')
    parser.add_argument('-tokenizer_name', required=True, help='json filename for tokenizer')
    parser.add_argument('-tokenizer_path', default='./data/tokenizers', help='path to tokenizer json files')
    #parser.add_argument('-type', default='char', help='type of tokenization (currently only char supported)') # TODO: BPE tokenizer
    
    args = parser.parse_args()

    df = pd.read_csv(args.data_path)

    smiles = df['smiles'].to_list()

    """with open(args.data_path, 'r') as f:
        molecules = f.readlines()
        molecules = [smiles.strip() for smiles in molecules]
"""
    tokens = set()

    print('Building tokenizer...')

    for mol in smiles:
        tokens |= set(mol.strip())
    
    id2token = {}
    for i, token in enumerate(tokens):
        id2token[i] = token
    
    print('Saving tokenizer...')
    
    with open(os.path.join(args.tokenizer_path, args.tokenizer_name), 'w') as f:
        json.dump(id2token, f)
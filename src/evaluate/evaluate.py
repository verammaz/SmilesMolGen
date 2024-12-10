import torch
import os
import pandas as pd
import numpy as np
import rdkit
import json
from tqdm import trange

from .metrics import *

def generate_smiles(model,
                    tokenizer,
                    temprature=1,
                    size=1000,
                    batch_size=100,
                    max_len=100,
                    device=torch.device('cuda'), verb=True):

    if verb: print(f'Evaluate {device}')
    model.to(device)
    model.eval()
    gen_smiles = []
    gen_tokens = []

    batches = 1 if size // batch_size == 0 else size // batch_size
    batch_size = batch_size if size > batch_size else size

    
    for batch in range(batches):
        tokens = model.sample([tokenizer.bos_token_id], batch_size, temprature, max_len, device, verb=verb)
        
        tokens = tokens.tolist()

        for mol in tokens:
            try:
                end_idx = mol.index(tokenizer.eos_token_id)
            except ValueError:
                end_idx = len(mol)
            mol = mol[:end_idx+1]
            gen_tokens.append(mol)
            smiles = tokenizer.decode(mol[1:-1])
            gen_smiles.append(smiles)

    model.train()
    return gen_smiles, gen_tokens


def get_statistics(generated_smiles, train_smiles, properties=['QED', 'SAS', 'LogP', 'MolWT'], save=True, save_path=None):
    stats = {}
    
    print('Filtering valid SMILES...')
    valid_smiles = filter_valid_molecules(generated_smiles)
    
    print('Calculating validity...')
    val = calc_validity(valid_smiles, generated_smiles)
    
    print('Calculating novelty...')
    nov = calc_novelty(valid_smiles, train_smiles)
    
    print('Calculating diversity')
    div = calc_diversity(valid_smiles)
    
    stats["Validity"] = val
    stats["Novelty"] = nov
    stats["Diversity"] = div

    molecules = convert_smiles_to_mol(valid_smiles)
    for property in properties:
        scores = []

        if property == 'QED':
            print('Calculating qed...')
            scores = calc_qed(molecules)
        
        elif property == 'LogP':
            print('Calculating logp...')
            scores = calc_logp(molecules, predictor=None)

        elif property == 'SAS':
            print('Calculating sas...')
            scores = calc_sas(molecules)
        
        elif property == 'MolWT':
            print('Calculating mol weights...')
            scores = calc_mw(molecules)

        if scores is not None:
            stats[property]= {"mean": np.mean(scores), "std": np.std(scores)}
    
    if save and save_path is not None:
        with open(save_path, 'w') as f:
            json.dump(stats, f, indent=4)


    return stats




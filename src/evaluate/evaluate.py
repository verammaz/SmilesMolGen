import torch
import os
import pandas as pd
import numpy as np
import rdkit
import json
from tqdm import trange

from metrics import *

def generate_smiles(model,
                    tokenizer,
                    temprature=1,
                    size=1000,
                    batch_size=100,
                    max_len=100,
                    device=torch.device('cuda')):

    print(f'Evaluate {device}')
    model.to(device)
    model.eval()
    gen_smiles = []

    batches = 1 if size // batch_size == 0 else size // batch_size
    batch_size = batch_size if size > batch_size else size

    
    for batch in trange(batches):
        tokens = model.sample([tokenizer.bos_token_id], batch_size, temprature, max_len, device)
        tokens = tokens.tolist()

        for mol in tokens:
            try:
                end_idx = mol.index(tokenizer.eos_token_id)
            except ValueError:
                end_idx = len(mol)
            mol = mol[:end_idx+1]
            smiles = tokenizer.decode(mol[1:-1])
            gen_smiles.append(smiles)

    return gen_smiles


def get_statistics(generated_smiles, train_smiles, properties=['QED', 'LogP', 'IC50'], save=True, save_path=None):
    stats = {}
    val = calc_validity(generated_smiles)
    nov = calc_novelty(generated_smiles, train_smiles)
    div = calc_diversity(generated_smiles, train_smiles)
    stats["Validity"] = {"mean": np.mean(val), "std": np.std(val), "max": np.max(val), "min": np.min(val)}
    stats["Novelty"] = nov
    stats["Diversity"] = div
    for property in properties:
        scores = []
        if property == 'QED':
            scores = calc_qed(generate_smiles)
        elif property == 'LogP':
            scores = calc_logp(generate_smiles, predictor=None)
        elif property == 'IC50':
            scores = calc_ic50(generate_smiles)
        stats[property] = {"mean": np.mean(scores), "std": np.std(scores), "max": np.max(scores), "min": np.min(scores)}
    
    if save and save_path is not None:
        with open(save_path, 'w') as f:
            json.dump(stats, f, indent=4)


    return stats




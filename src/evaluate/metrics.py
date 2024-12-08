import torch
import os
import sys
import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import QED, Crippen
from rdkit import RDConfig


def calc_validity(generated_smiles):
    valid_smiles = [smi for smi in generated_smiles if Chem.MolFromSmiles(smi) is not None]
    return len(valid_smiles) / len(generated_smiles)
    

def calc_novelty(generated_smiles, train_smiles):
    new_smiles = set(generated_smiles) - set(train_smiles)
    new_smiles = len(new_smiles)
    novelty_score = new_smiles / len(generated_smiles)
    return novelty_score

def calc_diversity(generated_smiles):
    unique_smiles = set(generated_smiles)
    return len(unique_smiles) / len(generated_smiles)

def calc_qed(generated_smiles):
    qed_values = [QED.qed(Chem.MolFromSmiles(smi)) for smi in generated_smiles]
    return qed_values

def calc_logp(generated_smiles, predictor=None):
    logp_values = []
    if predictor is None:  
        logp_values = [Crippen.MolLogP(Chem.MolFromSmiles(smi)) for smi in generated_smiles]

    return logp_values


def calc_ic50(generated_smiles):
    
    pass


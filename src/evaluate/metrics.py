import torch
import os
import sys
import pandas as pd
import numpy as np


from rdkit import Chem
from rdkit.Chem import QED, Crippen, Descriptors
from rdkit import RDConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

from rdkit import RDLogger
# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')

def filter_valid_molecules(generated_smiles):
  valid_smiles = [smi for smi in generated_smiles if Chem.MolFromSmiles(smi) is not None]
  return valid_smiles

def convert_smiles_to_mol(valid_smiles):
    molecules = [(Chem.MolFromSmiles(smi)) for smi in valid_smiles]
    return molecules


def calc_validity(valid_smiles, generated_smiles):
    return len(valid_smiles) / len(generated_smiles)

def calc_novelty(valid_smiles, train_smiles):
    new_smiles = set(valid_smiles) - set(train_smiles)
    new_smiles = len(new_smiles)
    novelty_score = new_smiles / len(valid_smiles)
    return novelty_score

def calc_diversity(valid_smiles):
    unique_smiles = set(valid_smiles)
    return len(unique_smiles) / len(valid_smiles)

def calc_qed(molecules):
    qed_values = [QED.qed(mol) for mol in molecules]
    return qed_values

def calc_logp(molecules, predictor=None):
    logp_values = []
    if predictor is None:  
        logp_values = [Crippen.MolLogP(mol) for mol in molecules]

    return logp_values

def calc_mw(molecules):
    weights = [Descriptors.MolWt(mol) for mol in molecules]
    return weights

def calc_sas(molecules):
    try:
        sascores = [sascorer.calculateScore(mol) for mol in molecules]
        return sascores
    except Exception:
        return None


def calc_ic50(molecules):
    pass

def get_top_molecules(molecules, smiles, k, property):
    scores = []
    if property == 'qed':
        scores = calc_qed(molecules)
    elif property == 'logp':
        scores = calc_logp(molecules)
    elif property == 'sas':
        scores = calc_sas(molecules)
    elif property == 'mwt':
        scores = calc_mw(molecules)
    else:
        [], [], []
    
    indices = np.argsort(scores)[-k:][::-1]

    top_mols, top_smiles, top_scores = [], [], []

    for id in indices:
        top_mols.append(molecules[id])
        top_smiles.append(smiles[id])
        top_scores.append(scores[id])

    return top_mols, top_smiles, top_scores



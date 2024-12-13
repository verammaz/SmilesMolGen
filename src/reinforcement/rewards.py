import torch
import rdkit
import numpy as np
from rdkit import Chem
from rdkit.Chem import QED, Crippen, Descriptors
from ..evaluate.metrics import *


def validity_reward(smiles_batch, device='cuda'):
    # Batch processing for validity
    mols = [Chem.MolFromSmiles(smiles_str) for smiles_str in smiles_batch]
    rewards = [1.0 if mol is not None else 0.0 for mol in mols]
    return torch.tensor(rewards, dtype=torch.float32, device=device)


def qed_reward(smiles_batch, device='cuda', **kwargs):
    mul = kwargs['qed_mul']  # Default multiplier is 1
    mols = [Chem.MolFromSmiles(smiles_str) for smiles_str in smiles_batch]
    rewards = [QED.qed(mol) * mul if mol is not None else 0.0 for mol in mols]
    return torch.tensor(rewards, dtype=torch.float32, device=device)


def logp_reward(smiles_batch, device='cuda', **kwargs):
    ideal_range = kwargs['ideal_range_logp']
    max_logp = kwargs['max_logp']
    lower_ideal, upper_ideal = ideal_range
    lower_penalty_scale = lower_ideal
    upper_penalty_scale = max_logp - upper_ideal

    mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_batch]
    rewards = []
    for mol in mols:
        if mol is None:
            rewards.append(0.0)
            continue

        logp = Descriptors.MolLogP(mol)
        if logp > max_logp:
            rewards.append(0.0)
        elif lower_ideal <= logp <= upper_ideal:
            rewards.append(1.0)
        else:
            if logp < lower_ideal:
                reward = 1.0 - (lower_ideal - logp) / lower_penalty_scale
            else:
                reward = 1.0 - (logp - upper_ideal) / upper_penalty_scale
            rewards.append(max(reward, 0.0))
    return torch.tensor(rewards, dtype=torch.float32, device=device)



def validity_qed_logp_reward(smiles_batch, device='cuda', **kwargs): 
    qed_mul = kwargs['qed_mul']
    qed_weight = kwargs['qed_weight']
    validity_weight = kwargs['validity_weight']
    logp_weight = kwargs['logp_weight']
    ideal_range = kwargs['ideal_range_logp']
    max_logp = kwargs['max_logp']

    # Batch process the SMILES
    mols = [Chem.MolFromSmiles(smiles_str) for smiles_str in smiles_batch]

    # Validity rewards
    validity_rewards = [1.0 if mol is not None else 0.0 for mol in mols]

    # QED rewards
    qed_rewards = [QED.qed(mol) * qed_mul if mol is not None else 0.0 for mol in mols]

    # logP rewards
    logp_rewards = []
    lower_ideal, upper_ideal = ideal_range
    lower_penalty_scale = lower_ideal  # Scale for values below the ideal range
    upper_penalty_scale = max_logp - upper_ideal  # Scale for values above the ideal range

    for mol in mols:
        if mol is None:
            logp_rewards.append(0.0)
            continue
        
        logp = Descriptors.MolLogP(mol)
        if logp > max_logp:
            logp_rewards.append(0.0)
        elif lower_ideal <= logp <= upper_ideal:
            logp_rewards.append(1.0)
        else:
            if logp < lower_ideal:
                reward = 1.0 - (lower_ideal - logp) / lower_penalty_scale
            else:  # logP > upper_ideal but <= max_logp
                reward = 1.0 - (logp - upper_ideal) / upper_penalty_scale
            logp_rewards.append(max(reward, 0.0))
    
    # Combine rewards with weights
    combined_rewards = [
        validity_weight * validity + qed_weight * qed + logp_weight * logp
        for validity, qed, logp in zip(validity_rewards, qed_rewards, logp_rewards)
    ]

    # Convert to PyTorch tensor
    return torch.tensor(combined_rewards, dtype=torch.float32, device=device)


def get_reward_fn(target, device, **kwargs):
    if target in ['QED', 'qed']:
        def qed_reward_with_kwargs(smiles_batch, device=device):
            return qed_reward(smiles_batch, device=device, **kwargs)
        return qed_reward_with_kwargs
    elif target in ['LogP', 'logp', 'logP']:
        def logp_reward_with_kwargs(smiles_batch, device=device):
            return logp_reward(smiles_batch, device=device, **kwargs)
        return logp_reward_with_kwargs
    elif target in ['Validity', 'val']:
        return validity_reward  # Assuming no kwargs needed for validity
    else:  # Default: validity + qed + logp
        def combined_reward_with_kwargs(smiles_batch, device=device):
            return validity_qed_logp_reward(smiles_batch, device=device, **kwargs)
        return combined_reward_with_kwargs

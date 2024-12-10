import torch
import rdkit
from rdkit import Chem
from rdkit.Chem import QED, Crippen, Descriptors
from ..evaluate.metrics import *


def validity_reward(smiles_batch, device='cuda'):
    # Batch processing for validity
    mols = [Chem.MolFromSmiles(smiles_str) for smiles_str in smiles_batch]
    rewards = [1.0 if mol is not None else 0.0 for mol in mols]
    return torch.tensor(rewards, dtype=torch.float32, device=device)


def qed_reward(smiles_batch, device='cuda'):
    # Batch processing for QED
    mols = [Chem.MolFromSmiles(smiles_str) for smiles_str in smiles_batch]
    rewards = [QED.qed(mol) if mol is not None else 0.0 for mol in mols]
    return torch.tensor(rewards, dtype=torch.float32, device=device)
    

def logp_reward(smiles_batch, device='cuda', ideal_range=(1.35, 1.8), max_logp=5.0):
    # Precompute the ideal range values for efficiency
    lower_ideal, upper_ideal = ideal_range
    lower_penalty_scale = lower_ideal  # Scale for values below the ideal range
    upper_penalty_scale = max_logp - upper_ideal  # Scale for values above the ideal range
    
    # Batch process the SMILES
    mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_batch]
    
    rewards = []
    for mol in mols:
        if mol is None:
            rewards.append(0.0)  # Invalid SMILES
            continue
        
        logp = Descriptors.MolLogP(mol)
        if logp > max_logp:
            rewards.append(0.0)  # Penalize logP > max_logp
        elif lower_ideal <= logp <= upper_ideal:
            rewards.append(1.0)  # Reward logP in the ideal range
        else:
            # Compute penalty based on deviation from the ideal range
            if logp < lower_ideal:
                reward = 1.0 - (lower_ideal - logp) / lower_penalty_scale
            else:  # logP > upper_ideal but <= max_logp
                reward = 1.0 - (logp - upper_ideal) / upper_penalty_scale
            rewards.append(max(reward, 0.0))  # Ensure rewards are non-negative
    
    # Convert rewards to a PyTorch tensor
    return torch.tensor(rewards, dtype=torch.float32, device=device)


def validity_qed_logp_reward(
    smiles_batch, device='cuda', qed_weight=1.0, validity_weight=1.0, logp_weight=1.0,
    ideal_range=(1.35, 1.8), max_logp=5.0
):
    # Batch process the SMILES
    mols = [Chem.MolFromSmiles(smiles_str) for smiles_str in smiles_batch]

    # Validity rewards
    validity_rewards = [1.0 if mol is not None else 0.0 for mol in mols]

    # QED rewards
    qed_rewards = [QED.qed(mol) if mol is not None else 0.0 for mol in mols]

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


def get_reward_fn(target):
    if target in ['QED', 'qed']: 
        return qed_reward
    elif target in ['LogP', 'logp', 'logP']:
        return logp_reward
    elif target in ['Validitiy', 'val']:
        return validity_reward
    else: # by default, return validity + qed reward
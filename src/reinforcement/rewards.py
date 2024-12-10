import torch
import rdkit
from rdkit import Chem


def validity_reward(smiles_batch, device='cude'):
    rewards = []
    for smiles_str in smiles_batch:
        try:
            mol = Chem.MolFromSmiles(smiles_str)
            rewards.append(1.0 if mol is not None else 0.0)
        except:
            rewards.append(0.0)
    return torch.tensor(rewards, dtype=torch.float32, device=device)
    


def get_reward_fn(properties):
    return validity_reward
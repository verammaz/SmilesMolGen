
import os
import sys
import json
import random

import numpy as np
import pandas as pd
import torch

def get_max_smiles_len(data_path):

    df = pd.read_csv(data_path)
    smiles = df['smiles'].to_list()
    max_len = len(max(smiles, key=len).strip())
    
    return max_len



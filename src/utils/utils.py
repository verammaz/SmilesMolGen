
import os
import sys
import json
import random
from ast import literal_eval

import numpy as np
import torch

# -----------------------------------------------------------------------------

def get_max_smiles_len(data_path: str) -> int:
    """
    Returns the length of the molecule which has the longest SMILES string.
    this is used for padding in the tokenizer when traning the model.  
    """
    if os.path.isdir(data_path):
        max_len = 0
        for path in os.listdir(data_path):
            full_path = os.path.join(data_path, path)
            file_max_len = len(max(open(full_path, 'r'), key=len))
            max_len = file_max_len if file_max_len > max_len else max_len
    else:
        max_len = len(max(open(data_path, 'r'), key=len).strip())
    
    return max_len



from torch.utils import data
from .smiles_dataset import SmilesDataset


def get_dataset(**kwargs):

	dataset = SmilesDataset(**kwargs)

	return dataset
from torch.utils.data import Dataset, random_split
from torch_geometric.loader import DataLoader as PyGDataLoader

import pickle
import os

class JSSPDataset(Dataset):
    def __init__(self, data_paths):
        """
        data_paths: list of strings (paths to .pkl files) or a single string (path to directory)
        """
        self.instances = []
        if isinstance(data_paths, str):
            if os.path.isdir(data_paths):
                # Load all .pkl in directory
                files = [os.path.join(data_paths, f) for f in os.listdir(data_paths) if f.endswith('.pkl')]
            else:
                files = [data_paths]
        else:
            files = data_paths

        for f_path in files:
            with open(f_path, 'rb') as f:
                batch_instances = pickle.load(f)
                # Ensure each instance knows its size
                for inst in batch_instances:
                    if 'size' not in inst:
                        # Infer from times shape: (N, M)
                        N, M = inst['times'].shape
                        inst['size'] = (N, M)
                self.instances.extend(batch_instances)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        return self.instances[idx]
    
    def filter_by_size(self, size_tuple):
        """Returns a subset of instances matching (N, M)."""
        return [inst for inst in self.instances if inst['size'] == size_tuple]

    def get_split(self, split_name):
        if split_name == 'train':
            return self.train_split
        elif split_name == 'val':
            return self.val_split
        else:
            raise ValueError(f"Unknown split: {split_name}")
        
        
def split_dataset(dataset, train_ratio=0.8):
    total = len(dataset)
    train_size = int(train_ratio * total)
    val_size = total - train_size
    train_subset, val_subset = random_split(dataset, [train_size, val_size])
    dataset.train_split = train_subset
    dataset.val_split = val_subset

def init_dataloaders(dataset, splits_to_use=('train', 'val'), batch_size=16, num_workers=0):
    loaders = {}
    for split_name in splits_to_use:
        subset = dataset.get_split(split_name)
        shuffle = split_name == 'train'
        loaders[split_name] = PyGDataLoader(subset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loaders

def get_dataloaders(dataset, splits_to_use=('train', 'val'), batch_size=16, num_workers=0):
    return init_dataloaders(dataset, splits_to_use, batch_size, num_workers)


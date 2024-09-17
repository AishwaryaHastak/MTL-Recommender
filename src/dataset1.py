import h5py
import pandas as pd
import os

import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import scipy.sparse as sp

from src.constants import batch_size, hdf5_file_path, csv_file_path

def convert_hdf5_to_csv(hdf5_file_path, csv_file_path):
    # Open the HDF5 file
    with h5py.File(hdf5_file_path, 'r') as hdf5_file:
        # Extract the data
        user_ids = hdf5_file['/user_id'][:]
        item_ids = hdf5_file['/item_id'][:]
        ratings = hdf5_file['/rating'][:]

    # Create a DataFrame
    df = pd.DataFrame({
        'user_id': user_ids,
        'item_id': item_ids,
        'rating': ratings
    })

    # Save the DataFrame to a CSV file
    df.to_csv(csv_file_path, index=False)
    
class MovieLensDataset(Dataset):
    def __init__(self):
        convert_hdf5_to_csv(hdf5_file_path, csv_file_path)
        self.data = pd.read_csv(csv_file_path, header = 0, nrows=1000)
        self.user_ids = torch.tensor(self.data['user_id'].values, dtype=torch.long)
        self.item_ids = torch.tensor(self.data['item_id'].values, dtype=torch.long)
        self.ratings = torch.tensor(self.data['rating'].values, dtype=torch.float)

        self.num_users = int(self.user_ids.max() + 1)
        self.num_items = int(self.item_ids.max() + 1)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return {
            'user_id': self.user_ids[idx],
            'item_id': self.item_ids[idx],
            'rating': self.ratings[idx]
        }
    
    def tocoo(self):
        """
        Transform to a scipy.sparse COO matrix.
        """
        row = self.user_ids
        col = self.item_ids
        data = self.ratings if self.ratings is not None else np.ones(len(self))

        return sp.coo_matrix((data, (row, col)),
                             shape=(self.num_users, self.num_items))

    def tocsr(self):
        """
        Transform to a scipy.sparse CSR matrix.
        """
        return self.tocoo().tocsr()


class MovieLensDataLoader():
    def __init__(self):
        self.dataset = MovieLensDataset() 
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None


    def load_data(self):
        # Shuffle and split the dataset into training and validation sets
        train_size = int(0.8 * len(self.dataset))
        test_val_size = len(self.dataset) - train_size
        val_size = int(0.9 * test_val_size)
        test_size = test_val_size - val_size
        
        train_dataset, test_val_dataset = random_split(self.dataset, [train_size, test_val_size])
        val_dataset, test_dataset = random_split(test_val_dataset, [val_size, test_size])

        # Create DataLoaders for batching
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
    def __getdataloader__(self, dataset = "train"):
        if dataset == "train":
            return self.train_loader
        elif dataset == "val":
            return self.val_loader
        elif dataset == "test":
            return self.test_loader

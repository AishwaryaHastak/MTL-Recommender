import h5py
import pandas as pd
import os

import pandas as pd
from sklearn.model_selection import train_test_split
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
    def __init__(self, dataset, num_users, num_items):
        self.user_ids = torch.tensor(dataset['user_id'].values, dtype=torch.long)
        self.item_ids = torch.tensor(dataset['item_id'].values, dtype=torch.long)
        self.ratings = torch.tensor(dataset['rating'].values, dtype=torch.float)
        
        self.num_users = num_users
        self.num_items = num_items

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
        row = self.user_ids.numpy()
        col = self.item_ids.numpy()
        data = self.ratings.numpy() if self.ratings is not None else np.ones(len(self))

        return sp.coo_matrix((data, (row, col)),
                            shape=(self.num_users, self.num_items))

    def tocsr(self):
        """
        Transform to a scipy.sparse CSR matrix.
        """
        return self.tocoo().tocsr()


class MovieLensDataLoader():
    def __init__(self):
        # self.dataset = MovieLensDataset() 
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.num_users = 0
        self.num_items = 0
        self.get_datasets()
        self.load_data()  

    def get_datasets(self):
        convert_hdf5_to_csv(hdf5_file_path, csv_file_path)
        data = pd.read_csv(csv_file_path, header = 0, nrows=1000)
        self.num_users = int(data['user_id'].values.max() + 1)
        self.num_items = int(data['item_id'].values.max() + 1)
        # Split data into training and test sets
        train_data, test_val_data = train_test_split(data, test_size=0.2, random_state=42)
        val_data, test_data = train_test_split(data, test_size=0.1, random_state=42)

        self.train_dataset = MovieLensDataset(train_data, self.num_users, self.num_items)
        self.val_dataset = MovieLensDataset(val_data, self.num_users, self.num_items)
        self.test_dataset = MovieLensDataset(test_data, self.num_users, self.num_items)
        # return train_dataset, val_dataset, test_dataset

    def load_data(self):
        # Shuffle and split the dataset into training and validation sets
        # train_size = int(0.8 * len(self.dataset))
        # test_val_size = len(self.dataset) - train_size
        # val_size = int(0.9 * test_val_size)
        # test_size = test_val_size - val_size
        # train_dataset, test_val_dataset = random_split(self.dataset, [train_size, test_val_size])
        # val_dataset, test_dataset = random_split(test_val_dataset, [val_size, test_size])
        # # Create DataLoaders for batching
        # self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        # self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        # self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        # self.train_dataset, test_val_dataset = random_split(self.dataset, [train_size, test_val_size])
        # self.val_dataset, self.test_dataset = random_split(test_val_dataset, [val_size, test_size])

        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)
        
    def __getdataset__(self, dataset_type = "train"):
        if dataset_type == "train":
            return self.train_dataset
        elif dataset_type == "val":
            return self.val_dataset
        elif dataset_type == "test":
            return self.test_dataset
        
    def __getdataloader__(self, dataset_type = "train"):
        if dataset_type == "train":
            return self.train_loader
        elif dataset_type == "val":
            return self.val_loader
        elif dataset_type == "test":
            return self.test_loader

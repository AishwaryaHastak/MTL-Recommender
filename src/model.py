# model.py

from src.loss import bpr_loss, regression_loss
from src.constants import batch_size
import torch.optim as optim
import torch
import numpy as np
from tqdm import tqdm

class MultitaskModel(object):
    def __init__(self,
                 net,
                 num_items,
                 factorization_weight = 0.5,
                 regression_weight = 0.5,
                 n_iter=1,
                 eval_steps=1,
                 batch_size=batch_size,
                 l2=0.0,
                 learning_rate=1e-3,):
        self._net = net
        self._num_items = num_items
        self._factorization_weight = factorization_weight
        self._regression_weight = regression_weight
        self._n_iter = n_iter
        self._eval_steps = eval_steps
        self._optimizer = optim.Adam(
                self._net.parameters(),
                weight_decay=l2,
                lr=learning_rate
            )
        self._factorization_loss_func = bpr_loss
        self._regression_loss_func = regression_loss
    
    def fit(self, dataloader_obj):

        for epoch in tqdm(range(self._n_iter)):
            epoch_factorization_loss = []
            epoch_regression_loss = []
            epoch_loss = []
            for batch in tqdm(dataloader_obj.train_loader):  
                user_ids, item_ids, ratings = batch['user_id'] , batch['item_id'], batch['rating'] 
                self._optimizer.zero_grad()

                positive_prediction, score = self._net(user_ids, item_ids)
                negative_prediction = self._get_negative_prediction(user_ids)

                # Compute loss
                factorization_loss = self._factorization_loss_func(positive_prediction, negative_prediction, ratings)
                regression_loss = self._regression_loss_func(score, ratings)
                loss = ( factorization_loss * self._factorization_weight +
                            regression_loss * self._regression_weight )
                
                epoch_factorization_loss.append(factorization_loss.item())
                epoch_regression_loss.append(regression_loss.item())
                epoch_loss.append(loss.item())

                loss.backward()
                self._optimizer.step()
        
            print(f"Epoch: {epoch}, Factorization Loss: {np.mean(epoch_factorization_loss)}, Regression Loss: {np.mean(epoch_regression_loss)}, Total Loss: {np.mean(epoch_loss)}")
        
            # Evaluate every eval_steps epochs
            if epoch % self._eval_steps == 0:
                self.evaluate(dataloader_obj.val_loader)

        return (np.mean(epoch_factorization_loss), 
                np.mean(epoch_regression_loss),
                np.mean(epoch_loss))
    
    def _get_negative_prediction(self, user_ids):
        """
        Generate negative predictions for user-item interactions, 
        corresponds to p_ij^- in the assignment.

        Parameters
        ----------

        user_ids: tensor
            A tensor of integer user IDs of shape (batch,) 
        Returns
        -------

        negative_prediction: tensor
            A tensor of user-item interaction log-probability
            of shape (batch,)
        """

        negative_items = torch.tensor(np.random.RandomState().randint(0, self._num_items, len(user_ids), dtype=np.int64))
        negative_prediction, _ = self._net(user_ids, negative_items)

        return negative_prediction

    def evaluate(self, data_loader, item_ids=None):  
        for batch in tqdm(data_loader): 
            with torch.no_grad():
                user_ids, item_ids, ratings = batch['user_id'] , batch['item_id'], batch['rating']
                positive_prediction, score = self._net(user_ids, item_ids)
                negative_prediction = self._get_negative_prediction(user_ids)

                # Compute loss
                factorization_loss = self._factorization_loss_func(positive_prediction, negative_prediction)
                regression_loss = self._regression_loss_func(score, ratings)
                loss = ( factorization_loss * self._factorization_weight +
                            regression_loss * self._regression_weight )

        print(f"Factorization Loss: {factorization_loss}, Regression Loss: {regression_loss}, Loss: {loss}")

        return (factorization_loss, regression_loss,loss)

    def predict(self, user_ids, item_ids=None): 

        if item_ids is None:
            item_ids = np.arange(self._num_items, dtype=np.int64)
 
        if np.isscalar(user_ids):
            user_ids = np.array(user_ids, dtype=np.int64)

        user_ids = torch.from_numpy(user_ids.reshape(-1, 1).astype(np.int64))
        item_ids = torch.from_numpy(item_ids.reshape(-1, 1).astype(np.int64))
        
        positive_prediction, score = self._net(user_ids, item_ids)
        return positive_prediction, score
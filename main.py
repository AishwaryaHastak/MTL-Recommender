# main.py
# from src.dataset import MovieLensDataLoader
from src.dataset_new import MovieLensDataLoader
from src.mtl_net import MultiTaskNet
from src.model import MultitaskModel
from src.evaluate import mse_score, mrr_score

if __name__ == '__main__':
    dataloader_obj = MovieLensDataLoader()
    # # dataloader_obj.load_data()
    # mlt_net = MultiTaskNet(dataloader_obj.dataset.num_users, dataloader_obj.dataset.num_items)

    mlt_net = MultiTaskNet(dataloader_obj.num_users, dataloader_obj.num_items)

    print("[DEBUG] Initializing the model...")
    mlt_model = MultitaskModel(mlt_net, 
                               dataloader_obj.num_items,
                               n_iter=10, 
                               eval_steps=5,
                               learning_rate=2e-3)
    # print("[DEBUG] Training the model...")
    # mlt_model.fit(dataloader_obj)
    # print(dataloader_obj.)
    print("[DEBUG] Evaluating the model...")
    mse = mse_score(mlt_model, dataloader_obj.test_loader)
    mrr = mrr_score(mlt_model, dataloader_obj.test_dataset, dataloader_obj.train_dataset)
    print(f"MSE: {mse}, MRR: {mrr}")
    
    
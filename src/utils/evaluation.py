import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader

class Dataset_from_matrix(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data_matrix):
        """
        Args: create a torch dataset from a tensor data_matrix with size n * p
        [treatment, features, outcome]
        """
        self.data_matrix = data_matrix
        self.num_data = data_matrix.shape[0]

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.data_matrix[idx, :]
        return (idx, (sample[0:-1], sample[-1]))


def get_iter(data_matrix, batch_size, shuffle=True):
    dataset = Dataset_from_matrix(data_matrix)
    iterator = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return iterator


def eval_outcome(model, data, data_eval):
    data = data.numpy()
   
    mse = np.zeros((data_eval.shape[0]*data_eval.shape[1], 1))
    arr_eval = np.zeros((data_eval.shape[0], data_eval.shape[1], model.dim_treat+2))
    i = 0
    j = 0
    loader = get_iter(data, batch_size=512, shuffle=False)
    for dataset_index, _ in loader:
        with torch.no_grad():
            # print(t_grid.shape, data.repeat(data_eval.shape[1], 0).shape)
            t_grid_i = data_eval[dataset_index, :, :-1].reshape(-1, model.dim_treat)
            # print(t_grid_i.shape, data[dataset_index].repeat(data_eval.shape[1], 0).shape)
            y_hat_i, hidden = model.forward(torch.tensor(t_grid_i, dtype=torch.float32), 
                    torch.tensor(data[dataset_index].repeat(data_eval.shape[1], 0), dtype=torch.float32))
        y_hat_i = y_hat_i.numpy().reshape(-1, 1)
        mse_i = ((data_eval[dataset_index, :, -1].reshape(-1, 1) - y_hat_i)**2)#.mean()
        arr_eval_i = np.hstack((data_eval[dataset_index].reshape(-1, model.dim_treat+1), y_hat_i)).reshape(-1, data_eval.shape[1], model.dim_treat+2)
    
        mse[i:i+len(mse_i), :] = mse_i
        arr_eval[j:j+len(arr_eval_i), :, :] = arr_eval_i
        i += len(mse_i)
        j += len(arr_eval_i)
    
    return arr_eval, mse.mean()


def eval_policy(model, data, model_true, y_optimal):
    t_hat = model(data)
    y_t_hat = model_true(t_hat)
    mse = ((y_t_hat - y_optimal)**2).mean()
    return mse

    
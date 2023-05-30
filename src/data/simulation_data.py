import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path


def simulate_t(x, v, dim_treat, param_shift, param_concentration):

    pred_x = (x[:, None,None,:] * v[None,...]).sum(3)
    pred_x = pred_x[..., 0] / (2*pred_x[..., 1])
    pred_x = 1/(1+np.exp(-pred_x))

    pred_x_adj = pred_x / 20 +0.2

    print(pred_x_adj.mean(0), pred_x_adj.std(0))

    t = np.zeros((x.shape[0], dim_treat))
    for i in range(x.shape[0]):
        alpha_b = param_concentration + 1
        # select beta s.t. modes of the beta distribution
        beta_b = (alpha_b-1) / (pred_x_adj[i, :]) - alpha_b + 2
        t[i, :] = np.random.beta(alpha_b, beta_b)

    return t


def simulate_y(t, x, v, param_interaction, noise=0.0):
    
    pred_x = (np.float32(x[:, None,None,:]) * np.float32(v[None,...])).sum(3) # reducing float for big tcga dataset
    pred_x = pred_x[..., 0] / (2*pred_x[..., 1])
    pred_x = 1/(1+np.exp(-pred_x))
    pred_x_adj = pred_x / 20 +0.2

    y = 2 + 2 * (pred_x.mean(1)+0.5) * (np.cos((t - pred_x_adj)*3*np.pi) - 0.01*((t - pred_x_adj)**2)).mean(axis=1)\
        - param_interaction* 0.1*(((t-pred_x_adj)**2).prod(axis=1))
    noise = np.random.normal(0, noise, size=len(y))
    return y + noise


def simulate_data(x, v, dim_treat, param_shift=0, param_concentration=1, param_interaction=1, noise=0.0):
    t = simulate_t(x, v, dim_treat, param_shift=param_shift, param_concentration=param_concentration)
    y = simulate_y(t, x, v, param_interaction=param_interaction, noise=noise)

    data = np.hstack((t, x, y.reshape(-1, 1)))
    return data


@hydra.main(version_base=None, config_path="../../conf", config_name="config.yaml")
def main(args: DictConfig):
    
    print(OmegaConf.to_yaml(args))
    df_features = pd.read_pickle(os.path.join(to_absolute_path(os.path.join(args.data_dir, args.dataset)), 'df_'+args.dataset+'_x.pkl'))

    dim_treat = args.dim_treat
    random_seed = args.random_seed

    data_args = args.data
    n_grid_1dim = data_args.n_grid_1dim
    param_shift = data_args.param_shift
    param_concentration = data_args.param_concentration
    param_interaction = data_args.param_interaction
    noise = data_args.noise
    dim_cov = df_features.shape[1]

    path = to_absolute_path(os.path.join(args.data_dir, args.dataset, str(args.dim_treat)+'dim', str(data_args.param_concentration)+'conc'))

    if not os.path.exists(path):
        os.makedirs(path)

    np.random.seed(random_seed)

    v = np.random.normal(loc=0.0, scale=1.0, size=(dim_treat, 2, dim_cov))
    v = v/(np.linalg.norm(v, 1, axis=2).reshape(dim_treat, 2, -1))
    

    idx_train, idx_test = train_test_split(list(range(len(df_features))), test_size=0.2, shuffle=True)

    simulate_train = df_features.iloc[idx_train, :].values
    simulate_test = df_features.iloc[idx_test, :].values

    data_train = simulate_data(simulate_train, v, dim_treat=dim_treat, param_shift=param_shift, 
                param_concentration=param_concentration, param_interaction=param_interaction, noise=noise)
    data_test = simulate_data(simulate_test, v, dim_treat=dim_treat, param_shift=param_shift, 
                param_concentration=param_concentration, param_interaction=param_interaction, noise=noise)

    if dim_treat > 1:	
        t_cartesian_prod = [torch.linspace(0, 1, n_grid_1dim)]*dim_treat
        t_grid = torch.cartesian_prod(*t_cartesian_prod).numpy()
    else:
        t_grid = np.expand_dims(np.linspace(0, 1, n_grid_1dim), 1)

    # y_eval_train = simulate_y(np.tile(t_grid, (simulate_train.shape[0], 1)), simulate_train.repeat(t_grid.shape[0], 0), v, param_interaction=1, noise=0.0)
    y_eval_test = simulate_y(np.tile(t_grid, (simulate_test.shape[0], 1)), simulate_test.repeat(t_grid.shape[0], 0), v, param_interaction=1, noise=0.0)

    # data_eval_train = np.hstack((np.tile(t_grid, (simulate_train.shape[0], 1)), 
                        # y_eval_train.reshape(-1, 1))).reshape(-1, t_grid.shape[0], dim_treat+1)
    data_eval_test = np.hstack((np.tile(t_grid, (simulate_test.shape[0], 1)), 
                        y_eval_test.reshape(-1, 1))).reshape(-1, t_grid.shape[0], dim_treat+1)

    dict_info = {'random_seed': random_seed,
                'n_grid_1dim': n_grid_1dim,
                'dim_treat': dim_treat,
                'param_shift': param_shift,
                'param_concentration': param_concentration,
                'param_interaction': param_interaction
                }
    pd.to_pickle(dict_info, os.path.join(path,'info.pkl'))
    np.save(os.path.join(path, 'train.npy'), data_train)
    np.save(os.path.join(path, 'test.npy'), data_test)
    # np.save(os.path.join(path, 'eval_train.npy'), data_eval_train)
    np.save(os.path.join(path, 'eval_test.npy'), data_eval_test)
    np.save(os.path.join(path, 'v_vector.npy'), v)

if __name__ == "__main__":
    main()
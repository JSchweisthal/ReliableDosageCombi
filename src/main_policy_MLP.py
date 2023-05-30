import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro
import tqdm
from torch.utils.tensorboard import SummaryWriter

from src.models.normflow import ConditionalNormalizingFlow

from src.data.simulation_data import simulate_y


from torch.utils.data import Dataset, DataLoader

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import get_original_cwd, to_absolute_path

from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter


# advanced loader for getting global sample index
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

folder_log = 'runs/policy/'+str(pd.Timestamp.now().round(freq='S')).replace(' ', '_').replace(':', '-')
@hydra.main(version_base=None, config_path="../conf", config_name="config.yaml")
def main(args: DictConfig):
    writer = SummaryWriter(folder_log)
    print(OmegaConf.to_yaml(args))

    dataset = args.dataset
    data_dir = args.data_dir
    save_dir = args.save_dir

    random_seed = args.random_seed
    args.policy.model_args.dim_treat = args.dim_treat
    args.density.model_args.dim_treat = args.dim_treat
    args.outcome.model_args.dim_treat = args.dim_treat

#########################################################
    policy_args = args.policy
    outcome_args = args.outcome
    density_args = args.density

    n_epochs = policy_args.n_epochs

    batch_size = policy_args.batch_size
    init_lr = policy_args.init_lr

    model_args = policy_args.model_args

    dim_treat = model_args.dim_treat

    load_path = to_absolute_path(os.path.join(data_dir, dataset, str(model_args.dim_treat)+'dim', str(args.data.param_concentration)+'conc'))
    save_path = to_absolute_path(os.path.join(save_dir, dataset, str(model_args.dim_treat)+'dim', str(args.data.param_concentration)+'conc', str(args.random_seed)))

    init_lr_penalty = policy_args.init_lr_penalty
    # epsilon = policy_args.epsilon # replaced by x percent quantile of train data
    lambda_start = policy_args.lambda_start

    np.random.seed(random_seed)
    torch.manual_seed(random_seed)


    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # load data set info and vector:
    data_info = pd.read_pickle(os.path.join(load_path, 'info.pkl'))
    v = np.load(os.path.join(load_path, 'v_vector.npy'))

    data = np.load(load_path + '/train.npy')
    train_matrix_ = torch.from_numpy(data).float()
    data = np.load(load_path + '/test.npy')
    test_matrix = torch.from_numpy(data).float()
    
    idx_train, idx_test = train_test_split(list(range(train_matrix_.shape[0])), test_size=0.2, shuffle=True)

    train_matrix = train_matrix_[idx_train]
    validation_matrix = train_matrix_[idx_test]

    train_loader = get_iter(train_matrix, batch_size=batch_size, shuffle=True)
    validation_loader = get_iter(validation_matrix, batch_size=validation_matrix.shape[0], shuffle=True)
    # test_loader = get_iter(test_matrix, batch_size=test_matrix.shape[0], shuffle=False)

    model_args.dim_cov = train_matrix.shape[1]-model_args.dim_treat-1 # minus treatments and outcome
    density_args.model_args.dim_cov = model_args.dim_cov
    outcome_args.model_args.dim_cov = model_args.dim_cov

    # set loss for maximizing predicted outcome with penalization in low density areas
    def criterion(out, gps, penalty, dataset_index, epsilon):
        loss = (-out - ((penalty[dataset_index, :]*torch.sign(penalty[dataset_index, :]))*(gps - epsilon))).mean()
        return loss


    ####################################################
    runs_val_out = np.array([])
    runs_test_out = np.array([])
    runs_best_model = None
    k_val_out = -999
     ################################
    # start looping through k_runs
    for k in range(policy_args.k_runs):
        hidden_dim = 50
        model_outcome = nn.Sequential(nn.Linear(train_matrix.shape[1]-1, hidden_dim),
                                nn.ReLU(),
                                nn.Linear(hidden_dim, hidden_dim),
                                nn.ReLU(),
                                nn.Linear(hidden_dim, hidden_dim),
                                nn.ReLU(),
                                nn.Linear(hidden_dim, hidden_dim),
                                nn.ReLU(),
                                nn.Linear(hidden_dim, 1))
        model_outcome.load_state_dict(torch.load(os.path.join(save_path, 'model_outcome_MLP.pth.tar'))['state_dict'])
        model_outcome.eval()

        model_density = ConditionalNormalizingFlow(input_dim=dim_treat, split_dim=dim_treat-1, context_dim=density_args.model_args.dim_cov, hidden_dim=density_args.model_args.hidden_dim, 
            num_layers=density_args.model_args.num_layers, flow_length=density_args.model_args.flow_length, count_bins=density_args.model_args.count_bins,
            order=density_args.model_args.order, bound=density_args.model_args.bound, use_cuda=False)
        model_density.load_state_dict(torch.load(os.path.join(save_path, 'model_density.pth.tar')))
        model_density.eval()

        # define epsilon and lambda
        if policy_args.unadjusted:
            epsilon = 0
            lambda_start = 0
        else:
            tar_eps = model_density.log_prob(train_matrix[:, :dim_treat]-0.5, train_matrix[:, dim_treat:-1]).exp().detach()
            tar_eps = torch.quantile(tar_eps.detach(), 0.05)
            print(tar_eps)
            epsilon = tar_eps
        # define policy network
        model_policy = nn.Sequential(nn.Linear(model_args.dim_cov, model_args.hidden_dim),
                            nn.ReLU(),
                            nn.Linear(model_args.hidden_dim, model_args.hidden_dim),
                            nn.ReLU(),
                            nn.Linear(model_args.hidden_dim, model_args.dim_treat))
        

        optimizer_model = torch.optim.Adam(model_policy.parameters(), lr=init_lr)

        # Penalty
        penalty_param = torch.nn.Parameter(torch.ones(len(train_matrix), 1)*lambda_start)
        optimizer_penalty = torch.optim.Adam([penalty_param], lr=init_lr_penalty)

        # train
        epochs = tqdm.trange(n_epochs)
        best_val_out = -1* np.inf
        patience = 20
        p = 0
        penalty_epoch = 0
        penalty_last_epoch = 0
        best_model = model_policy
        for epoch in epochs:
            model_policy.train()
            running_loss = 0
            val_out_epoch = 0
            val_out_true_epoch = 0
            out_epoch = 0
            gps_epoch = 0
    
            for idx, (dataset_index, (inputs, y)) in enumerate(train_loader):
                x = inputs[:, model_args.dim_treat:]

                optimizer_model.zero_grad()
                optimizer_penalty.zero_grad()

                t_param = model_policy(x)
                # scale for CNF
                t_param = torch.clip(t_param, -0.5, 0.5)
                gps = model_density.log_prob(t_param, x).exp()
                out = model_outcome.forward(torch.cat((t_param+0.5, x), dim=1))
                loss = criterion(out, gps, penalty_param.detach(), dataset_index, epsilon) #, penalty=penalty_param
                # print(loss)
                loss.backward(retain_graph =True) # retain_graph =True
                optimizer_model.step()
                
                # add adversarial gradient ascent for learning individual penalty
                loss = -1 * criterion(out.detach(), gps.detach(), penalty_param, dataset_index, epsilon) #, penalty=penalty_param
                loss.backward() #retain_graph =True    
                optimizer_penalty.step()

                running_loss += loss.detach()* (-1)
                out_epoch += out.detach().mean()
                gps_epoch += gps.detach().mean()
                penalty_epoch += penalty_param.detach().mean()
            running_loss /= len(train_loader)
            out_epoch /= len(train_loader)
            gps_epoch /= len(train_loader)
            penalty_epoch /= len(train_loader)

            model_policy.eval()
            with torch.no_grad():
                for idx, (dataset_index, (inputs, y))  in enumerate(validation_loader):
                    x = inputs[:, model_args.dim_treat:]
                    t_param = model_policy(x)
                    # print(t_param.mean(), t_param.std(), t_param.min(), t_param.max())
                    t_param = torch.clip(t_param, -0.5, 0.5)
                    gps = model_density.log_prob(t_param, x).exp()
                    out = model_outcome.forward(torch.cat((t_param+0.5, x), dim=1))
                    out = torch.where(gps > epsilon, out.reshape(-1), torch.tensor(0.0)).sum() / inputs.shape[0]
                    val_out_epoch += out.detach()
                    # true outcome
                    val_out_true_epoch += simulate_y(t_param.numpy() + 0.5, x.numpy(), v, param_interaction=data_info['param_interaction'], noise=0).mean()
                val_out_epoch /= len(validation_loader)
                val_out_true_epoch /= len(validation_loader)


            epochs.set_description("Train: [ PolicyLoss: {:.3f} , AvgOutcome: {:.3f}, AvgDensity: {:.3f}] --- Val: [AvgOutcome: {:.3f}]".format(running_loss, out_epoch, gps_epoch, val_out_epoch))

            writer.add_scalar(f"{dataset}/{model_args.dim_treat}dim/1_train_loss", running_loss, epoch)
            writer.add_scalar(f"{dataset}/{model_args.dim_treat}dim/2_train_penalty", penalty_epoch, epoch)
            writer.add_scalar(f"{dataset}/{model_args.dim_treat}dim/3_train_density", gps_epoch, epoch)
            writer.add_scalar(f"{dataset}/{model_args.dim_treat}dim/4_train_outcome", out_epoch, epoch)
            writer.add_scalar(f"{dataset}/{model_args.dim_treat}dim/5_validation_outcome", val_out_epoch, epoch)
            writer.add_scalar(f"{dataset}/{model_args.dim_treat}dim/6_validation_outcome_true", val_out_true_epoch, epoch)

            if (epoch >= 10) and (val_out_epoch > (best_val_out+0.01)):
                best_val_out = val_out_epoch
                best_model = model_policy
                p = 0
            elif  (not policy_args.unadjusted) and (penalty_epoch < (penalty_last_epoch-0.01)):
                p = 0
            else:
                p += 1
            if (epoch > 50) and (p >= patience):
                print(f"Early stopping at epoch: {epoch}, patience: {patience}")
                break
            penalty_last_epoch = penalty_epoch.clone()
        writer.flush()

        t_estimated_test = model_policy(test_matrix[:, dim_treat:-1]).detach()
        out_test = simulate_y(t_estimated_test.numpy() + 0.5, test_matrix[:, dim_treat:-1].numpy(), v, \
                            param_interaction=data_info['param_interaction'], noise=0).mean()
        
        print(f"Test outcome: {out_test}")
    
        runs_val_out = np.append(runs_val_out, best_val_out)
        runs_test_out = np.append(runs_test_out, out_test)

        if best_val_out > k_val_out:
            runs_best_model = best_model
    dict_runs = {'val_out': runs_val_out, 'test_out': runs_test_out}

    # save dict to pickle
    df_runs = pd.DataFrame(dict_runs)
    if policy_args.unadjusted:
        df_runs.to_pickle(os.path.join(save_path, 'runs_MLP_unadjusted.pkl'))
        torch.save(runs_best_model.state_dict(), os.path.join(save_path, f'model_policy_MLP_unadjusted.pth.tar'))
    else:
        df_runs.to_pickle(os.path.join(save_path, 'runs_MLP.pkl'))
        torch.save(runs_best_model.state_dict(), os.path.join(save_path, f'model_policy_MLP.pth.tar'))


if __name__ == "__main__":
    main()

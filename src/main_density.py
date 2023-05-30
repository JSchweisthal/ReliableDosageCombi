import os
import numpy as np
import pandas as pd
import torch
import pyro
import tqdm

from src.models.normflow import ConditionalNormalizingFlow
from src.data.data import get_iter

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import get_original_cwd, to_absolute_path

from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter

folder_log = 'runs/density/'+str(pd.Timestamp.now().round(freq='S')).replace(' ', '_').replace(':', '-')
@hydra.main(version_base=None, config_path="../conf", config_name="config.yaml")
def main(args: DictConfig):
    writer = SummaryWriter(folder_log)
    print(OmegaConf.to_yaml(args))

    dataset = args.dataset
    data_dir = args.data_dir
    save_dir = args.save_dir

    random_seed = args.random_seed
    args.density.model_args.dim_treat = args.dim_treat

#########################################################
    density_args = args.density


    n_epochs = density_args.n_epochs

    batch_size = density_args.batch_size
    init_lr = density_args.init_lr

    model_args = density_args.model_args

    dim_treat = model_args.dim_treat

    load_path = to_absolute_path(os.path.join(data_dir, dataset, str(model_args.dim_treat)+'dim', str(args.data.param_concentration)+'conc'))
    save_path = to_absolute_path(os.path.join(save_dir, dataset, str(model_args.dim_treat)+'dim', str(args.data.param_concentration)+'conc', str(args.random_seed)))

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    data = np.load(load_path + '/train.npy')
    train_matrix_ = torch.from_numpy(data).float()

    idx_train, idx_test = train_test_split(list(range(train_matrix_.shape[0])), test_size=0.2, shuffle=True)

    train_matrix = train_matrix_[idx_train]
    validation_matrix = train_matrix_[idx_test]

    train_loader = get_iter(train_matrix, batch_size=batch_size, shuffle=True)
    validation_loader = get_iter(validation_matrix, batch_size=validation_matrix.shape[0], shuffle=True)

    
    model_args.dim_cov = train_matrix.shape[1]-dim_treat-1

####################################################

    # Build NF model
    model = ConditionalNormalizingFlow(input_dim=dim_treat, split_dim=dim_treat-1, context_dim=model_args.dim_cov, hidden_dim=model_args.hidden_dim, 
        num_layers=model_args.num_layers, flow_length=model_args.flow_length, count_bins=model_args.count_bins,
        order=model_args.order, bound=model_args.bound, use_cuda=False)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr) 
    print("number of params: ", sum(p.numel() for p in model.parameters())) 

    # train
    pyro.clear_param_store()
    epochs = tqdm.trange(n_epochs)
    best_val_loss = np.inf
    patience = 50
    p = 0
    best_model = model
    for epoch in epochs:
        model.train()
        running_loss = 0
        val_loss = 0
        

        for idx, (inputs, y) in enumerate(train_loader):
            t =  inputs[:, :dim_treat]
            x = inputs[:, dim_treat:]
            # center t
            t -=0.5
            # add noise regularization
            t += torch.randn_like(t) * 0.1
            t = torch.clip(t, -0.5, 0.5)

            optimizer.zero_grad()
            if model.use_cuda:
                t, x = t.cuda(), x.cuda()
            loss = -model.log_prob(t, x).mean()
            loss.backward()
            optimizer.step()
            model.flow_dist.clear_cache() 
            running_loss += float(loss)
        running_loss /= len(train_loader)

        model.eval()
        with torch.no_grad():
            for idx, (inputs, y) in enumerate(validation_loader):
                t =  inputs[:, :dim_treat]
                x = inputs[:, dim_treat:]
                # center t
                t -=0.5

                if model.use_cuda:
                    t, x = t.cuda(), x.cuda()
                loss = -model.log_prob(t, x).mean()
                model.flow_dist.clear_cache() 
                val_loss += float(loss) 
            val_loss /= len(validation_loader)

        epochs.set_description("Train Loss: {:.3f} --- Validation Loss: {:.3f}".format(running_loss, val_loss))
        writer.add_scalar(f"{dataset}/{model_args.dim_treat}dim/1_train_loss", running_loss, epoch)
        writer.add_scalar(f"{dataset}/{model_args.dim_treat}dim/2_validation_loss", val_loss, epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model
            p = 0
        else:
            p += 1
        if p >= patience:
            print(f"Early stopping at epoch: {epoch}, patience: {patience}")
            break
       
    torch.save(best_model.state_dict(), os.path.join(save_path, f'model_density.pth.tar'))
    print(os.path.join(save_path, f'model_density.pth.tar'))

if __name__ == "__main__":
    main()
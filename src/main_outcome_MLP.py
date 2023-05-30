import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import tqdm

from src.data.data import get_iter
from src.utils.evaluation import eval_outcome

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path

from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter


folder_log = 'runs/outcome/'+str(pd.Timestamp.now().round(freq='S')).replace(' ', '_').replace(':', '-')

@hydra.main(version_base=None, config_path="../conf", config_name="config.yaml")
def main(args: DictConfig):
    
    writer = SummaryWriter(folder_log)
    print(OmegaConf.to_yaml(args))

    dataset = args.dataset
    data_dir = args.data_dir
    save_dir = args.save_dir

    random_seed = args.random_seed
    args.outcome.model_args.dim_treat = args.dim_treat

#########################################################
    outcome_args = args.outcome

    n_epochs = outcome_args.n_epochs
    verbose = outcome_args.verbose

    batch_size = outcome_args.batch_size
    init_lr = outcome_args.init_lr

    model_args = outcome_args.model_args

    load_path = to_absolute_path(os.path.join(data_dir, dataset, str(model_args.dim_treat)+'dim', str(args.data.param_concentration)+'conc'))
    save_path = to_absolute_path(os.path.join(save_dir, dataset, str(model_args.dim_treat)+'dim', str(args.data.param_concentration)+'conc', str(args.random_seed)))

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    with open(os.path.join(save_path, 'config.yaml'), 'w') as fp:
        OmegaConf.save(config=args, f=fp.name)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    data = np.load(load_path + '/train.npy')
    train_matrix_ = torch.from_numpy(data).float()
    data = np.load(load_path + '/test.npy')
    test_matrix = torch.from_numpy(data).float()
    # data_eval_test = np.load(load_path + '/eval_test.npy')

    idx_train, idx_test = train_test_split(list(range(train_matrix_.shape[0])), test_size=0.2, shuffle=True)

    train_matrix = train_matrix_[idx_train]
    validation_matrix = train_matrix_[idx_test]
    test_matrix = test_matrix

    train_loader = get_iter(train_matrix, batch_size=batch_size, shuffle=True)
    validation_loader = get_iter(validation_matrix, batch_size=validation_matrix.shape[0], shuffle=True)
    # test_loader = get_iter(test_matrix, batch_size=test_matrix.shape[0], shuffle=False)

    def save_checkpoint(state, model_name, checkpoint_dir='.'):
        filename = os.path.join(checkpoint_dir, model_name + '.pth.tar')
        print('=> Saving checkpoint to {}'.format(filename))
        torch.save(state, filename)

    # criterion
    def criterion(out, y):
        return ((out.squeeze() - y.squeeze())**2).mean()
    
    hidden_dim = 50
    model = nn.Sequential(nn.Linear(train_matrix.shape[1]-1, hidden_dim),
                            nn.ReLU(),
                            nn.Linear(hidden_dim, hidden_dim),
                            nn.ReLU(),
                            nn.Linear(hidden_dim, hidden_dim),
                            nn.ReLU(),
                            nn.Linear(hidden_dim, hidden_dim),
                            nn.ReLU(),
                            nn.Linear(hidden_dim, 1))

    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)

    epochs = tqdm.trange(n_epochs)
    best_val_loss = np.inf
    patience = 20
    p = 0
    best_model = model
    for epoch in epochs:
        model.train()
        running_loss = 0
        val_loss = 0

        for idx, (inputs, y) in enumerate(train_loader):   
            optimizer.zero_grad()
            out = model.forward(inputs)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.data
        running_loss /= len(train_loader)

        model.eval()
        with torch.no_grad():
            for idx, (inputs, y) in enumerate(validation_loader):
                out = model.forward(inputs)
                loss = criterion(out, y)
                val_loss += loss.data
            val_loss /= len(validation_loader)
        
        epochs.set_description("Train loss: {:.3f} --- Validation loss: {:.3f}".format(running_loss, val_loss) )

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
    save_checkpoint({
                'model_name': 'MLP',
                # 'MISE_out': mse_out,
                'state_dict': best_model.state_dict()
            }, 'model_outcome_MLP', checkpoint_dir=save_path)
    print(f"Save final model to: {os.path.join(save_path, f'model_outcome_MLP')}")

if __name__ == "__main__":
    main()
from typing import Any
from py4cast.datasets import get_datasets
import torch
from torch.utils.data import DataLoader
import os

#loading the dataset - code from oscar

dataset_configuration: dict[str, Any] = {
    "periods": {
      "train": {
        "start": 20200101,
        "end": 20221231,
        "obs_step": 3600,
      },
      "valid": {
        "start": 20230101,
        "end": 20231231,
        "obs_step": 3600,
        "obs_step_btw_t0": 10800,
      },
      "test": {
        "start": 20240101,
        "end": 20240831,
        "obs_step": 3600,
        "obs_step_btw_t0": 10800,
      },
    },
    "grid": {
      "name": "PAAROME_1S40",
      "border_size": 0,
      "subdomain": [100, 612, 240, 880],
      "proj_name": "PlateCarree",
      "projection_kwargs": {},
    },
    "settings": {
      "standardize": True,
      "file_format": "npy",
    },
    "params": {
      "aro_r2": {
        "levels": [2],
        "kind": "input_output",
        },
      "aro_tp": {
        "levels": [0],
        "kind": "input_output",
        },
      "aro_u10": {
        "levels": [10],
        "kind": "input_output",
        },
      "aro_v10": {
        "levels": [10],
        "kind": "input_output",
        },
      "aro_t": {
        "levels": [250, 500, 700, 850],
        "kind": "input_output",
        },
      "aro_u": {
        "levels": [250, 500, 700, 850],
        "kind": "input_output",
        },
      "aro_v": {
        "levels": [250, 500, 700, 850],
        "kind": "input_output",
        },
      "aro_z": {
        "levels": [250, 500, 700, 850],
        "kind": "input_output",
        },
    },
}

#extracting data from the dataset

train, test, val = get_datasets(
    name="titan_aro_arp",
    num_input_steps=1,
    num_pred_steps_train=1,
    num_pred_steps_val_test=1,
    dataset_conf=dataset_configuration,
)

print(train)
print(type(train))

# choose workers based on CPU count
num_workers = min(4, os.cpu_count())
batch_size = 32 # adjust for GPU/CPU memory

def collate_items(batch):
    # batch is a list of Item objects
    inputs = [item.inputs.tensor.squeeze(dim=0) for item in batch]
    targets = [item.outputs.tensor.squeeze(dim=0) for item in batch]
    
    # Stack into tensors for batching
    inputs = torch.stack(inputs, dim=0)
    targets = torch.stack(targets, dim=0)
    
    return inputs, targets

train_loader = DataLoader(
    train,
    batch_size=batch_size,            
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,   # keeps workers alive between epochs
    collate_fn=collate_items
)

val_loader = DataLoader(
    val,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
    collate_fn=collate_items
)

test_loader = DataLoader(
    test,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
    collate_fn=collate_items
)

#loading the model

from mfai.pytorch.models.gaussian_diffusion import GaussianDiffusionSettings, GaussianDiffusion

settings = GaussianDiffusionSettings(
    timesteps = 100,
    sampling_timesteps = None,
    objective = "pred_v",
    beta_schedule = "sigmoid",
    schedule_fn_kwargs = {},
    ddim_sampling_eta = 0.0,
    auto_normalize = True,
    offset_noise_strength = (0.0,),
    min_snr_loss_weight = False,
    min_snr_gamma = 5,
    immiscible = False
)

model = GaussianDiffusion(
  in_channels = len(train.params), #correspond bien au nombre de features
  out_channels = len(train.params),
  input_shape = (train.grid.x,train.grid.y), #on part du principe que les datasets auront tous la même grille
  settings = settings
)

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

print("cuda available? ",torch.cuda.is_available())

for input, target in train_loader:
    input_tensor = input.permute(0,3,1,2) #reshape to fit format
    target_tensor = target.permute(0,3,1,2)
    # Maintenant tu peux appeler ton modèle avec input pour générer et target pour avoir une loss

    # Zero your gradients for every batch!
    optimizer.zero_grad()
    output = model(input_tensor)

    # Compute the loss and its gradients
    loss = loss_fn(output, target_tensor)
    loss.backward()

    # Adjust learning weights
    optimizer.step()
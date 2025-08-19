import csv
import cartopy
import json
import numpy as np
from typing import Any
from py4cast.datasets import get_datasets
from py4cast.plots import plot_prediction, DomainInfo
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_DIR = "output_data/"
GRAPH_STEP = 1000 #pas d'enregistrement des graphiques de comparaison.
EPOCH_COUNT = 100 #nombre d'Ã©poch
TIMESTEPS = 2 #nombre de timesteps
BATCH_SIZE = 4
NUM_WORKERS = 9
FEATURE_PLOT_WHITELIST = ["aro_t2m_2m","aro_tp_0m","aro_r2_2m","aro_u10_10m"]

#loading the dataset - code from oscar

dataset_configuration: dict[str, Any] = {
    "periods": {
      "train": {
        "start": 20200101,
        "end": 20200101, #20240815,
        "obs_step": 3600,
      },
      "valid": {
        "start": 20240816,
        "end": 20240817,
        "obs_step": 3600,
        "obs_step_btw_t0": 10800,
      },
      "test": {
        "start": 20240818,
        "end": 20240819,
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
      "aro_t2m": {
        "levels": [2],
        "kind": "input_output",
        },
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
      "arp_t": {
        "levels": [250, 500, 700, 850],
        "kind": "input",
      },
      "arp_u":{
        "levels": [250, 500, 700, 850],
        "kind": "input",
      },
      "arp_v":{
        "levels": [250, 500, 700, 850],
        "kind": "input",
      },
      "arp_z":{
        "levels": [250, 500, 700, 850],
        "kind": "input",
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

#make DataLoaders

def collate_items(batch):
    # batch is a list of Item objects
    inputs = [item.forcing.tensor.squeeze(dim=0) for item in batch]
    targets = [item.outputs.tensor.squeeze(dim=0) for item in batch]
    
    # Stack into tensors for batching
    inputs = torch.stack(inputs, dim=0)
    targets = torch.stack(targets, dim=0)
    
    return inputs, targets

train_loader = DataLoader(
    train,
    batch_size=BATCH_SIZE,            
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True,   # keeps workers alive between epochs
    collate_fn=collate_items
)

#loading the model

from mfai.pytorch.models.gaussian_diffusion import GaussianDiffusionSettings, GaussianDiffusion

settings = GaussianDiffusionSettings(
    timesteps = TIMESTEPS,
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
  in_channels = 21, #correspond bien au nombre de features
  out_channels = 21,
  input_shape = (train.grid.x,train.grid.y), #on part du principe que les datasets auront tous la mÃªme grille
  settings = settings
)
model.to(device)

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

loss_scores = []

def save_loss_scores_entry():
  global loss_scores
  loss_to_save = loss_scores.pop(0)
  with open(OUTPUT_DIR+"losses/loss_scores.csv","a") as lossfile:
    wr = csv.writer(lossfile)
    wr.writerow(loss_to_save)

domain_info = train.domain_info
interior_mask = torch.zeros_like(train[0].forcing.tensor.permute(0,3,1,2).detach()[:,0,:,:].squeeze(0))
loss_mask = torch.ones_like(train[0].outputs.tensor)

feature_whitelist = []

for feature in FEATURE_PLOT_WHITELIST:
  j = train[0].outputs.feature_names.index(feature)
  feature_whitelist.append((j,feature))

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR+"figures/", exist_ok=True)
os.makedirs(OUTPUT_DIR+"models/", exist_ok=True)
os.makedirs(OUTPUT_DIR+"losses/", exist_ok=True)

with open(OUTPUT_DIR+"losses/loss_scores.csv","w") as lossfile:
  wr = csv.writer(lossfile)
  wr.writerow(["epoch","train_i","loss"])

for epoch in range(EPOCH_COUNT):
  i = 0
  for input_tensor,target_tensor in train_loader:
    i+=1
    input_tensor = input.permute(0,3,1,2) #reshape to fit format
    target_tensor = target.permute(0,3,1,2)
    input_tensor.to(device)
    target_tensor.to(device)

    # Maintenant tu peux appeler ton modÃ¨le avec input pour gÃ©nÃ©rer et target pour avoir une loss

    # Zero your gradients for every batch!
    optimizer.zero_grad()
    output = model(input_tensor)

    # Compute the loss and its gradients
    loss = loss_fn(output, target_tensor)
    loss.backward()
    loss_scores.append([epoch,i,loss.item()])
    print(f"new loss score: {loss_scores[-1][-1]}")
    is_better_loss = len(loss_scores)>1 and loss_scores[-1][-1] < loss_scores[-2][-1]
    if is_better_loss:
      print(f"latest loss is better ({loss_scores[-1][-1]} < {loss_scores[-2][-1]}), saving model number {epoch}-{i} as best_loss")
      torch.save({"epoch":epoch,"global_step":i,"state_dict":model.state_dict()}, OUTPUT_DIR+f"models/model_weights_best_loss.pth")
    if len(loss_scores)>2:
      save_loss_scores_entry()
    #plotting output against target
    if i%GRAPH_STEP == 0 or is_better_loss:
      for j,feature_name in feature_whitelist:
        pred = output.detach()[:,j,:,:].squeeze(0)
        plot_target = target_tensor.detach()[:,j,:,:].squeeze(0)
        fig = plot_prediction(pred=pred,target=plot_target,interior_mask=interior_mask,domain_info=domain_info,title=feature_name+" "+str(i),vrange=None)
        if i%GRAPH_STEP == 0:
          fig.savefig(OUTPUT_DIR+f"figures/target_output_{feature_name}_{epoch}-{i}.png")
        if is_better_loss:
          fig.savefig(OUTPUT_DIR+f"figures/target_output_{feature_name}_best_loss.png")
        plt.close(fig)
    # Adjust learning weights
    optimizer.step()
    print(f"{len(train)-i} left in training, epoch {epoch}/{EPOCH_COUNT-1}")

#save remaining loss
while len(loss_scores)>0:
  save_loss_scores_entry()

torch.save({"epoch":epoch,"global_step":i,"state_dict":model.state_dict()}, OUTPUT_DIR+f"models/model_weights_final.pth")
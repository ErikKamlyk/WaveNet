import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as utils
import numpy as np
import pandas as pd
import random
import os
import time
from IPython import display
from dataclasses import dataclass
import sys
import math

import torch
from torch import nn

import torchaudio

import librosa
from matplotlib import pyplot as plt

from melspectrogram import MelSpectrogram, MelSpectrogramConfig

featurizer = MelSpectrogram(MelSpectrogramConfig())

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
BATCH_SIZE = 1

torch.backends.cudnn.deterministic = True
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

from dataset import load_dataset
dataloader_train, dataloader_val = load_dataset(featurizer, BATCH_SIZE)

from model import WaveNet

generator = WaveNet(n_mels=80, n_blocks=20).to(device)

from math import exp, log

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import wandb
run = wandb.init(
  project="DLA_HW5",
  config={
    "n_mels": 80,
    "n_blocks": 20,
    "learn_rate": 0.001,
    "batch_size": 1}
)
config = wandb.config
api = wandb.Api()

def mu_encode(x, mu=255.):
    return torch.floor((torch.sign(x)*torch.log(1 + mu*torch.abs(x))/math.log(1 + mu) + 1)*128)


def mu_decode(x, mu=255.):
    x = ((x/128) - 1)
    sgn = torch.sign(x)
    x = torch.abs(x)
    return sgn*(torch.exp(x*math.log(1 + mu)) - 1)/mu


optimizer = optim.Adam(generator.parameters(), lr=0.001, eps=1e-8, weight_decay=0)
lambda1 = lambda step: exp(log(0.01)*min(75000, step)/75000)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda1])

criterion = nn.CrossEntropyLoss()

full_model = torch.load('model_final')
generator.load_state_dict(full_model['weights'])
optimizer.load_state_dict(full_model['optimizer'])

def inference(mels, real_audio = None, i=0, epoch=0):
    generator.eval()
    val_losses = []
    audio = torch.zeros(1, 1, 1).to(device)
    with torch.no_grad():
        for length in range(1, mels.shape[2] + 1):
            if length % 100 == 0:
                print(length)
            res = generator(audio[:, :, -1:], mels[:, :, length-1:length])
            #res = generator(audio[:, :, -2048:], mels[:, :, max(0, length-2048):length])
            audio = torch.cat([audio, mu_decode(torch.argmax(res[:, :, -1:], dim=1)).unsqueeze(1)], dim=2)
        name = "example " + str(i) + "_no_tf"
        wandb_gen = wandb.Audio(audio[0].squeeze().detach().cpu().numpy(), caption="Inference", sample_rate=22050)
        wandb_audios = [wandb_gen]
        if real_audio != None:
            wandb_real = wandb.Audio(real_audio[0].squeeze().detach().cpu().numpy(), caption="Real", sample_rate=22050)
            wandb_audios.append(wandb_real)
        wandb.log({name: wandb_audios}, step=epoch)

def inference_fast(mels, real_audio = None, i=0, epoch=0):
    generator.eval()
    val_losses = []
    audio = torch.zeros(mels.shape[0], 1, 1).to(device)
    with torch.no_grad():
        for length in range(1, mels.shape[2] + 1):
            if length % 1000 == 0:
                print(length)
            res = generator(audio[:, :, -1:], mels[:, :, length-1:length], fast_generation=True)
            audio = torch.cat([audio, mu_decode(torch.argmax(res[:, :, -1:], dim=1)).unsqueeze(1)], dim=2)
        name = "example " + str(i) + "_no_tf"
        wandb_gen = wandb.Audio(audio[0].squeeze().detach().cpu().numpy(), caption="Inference_1", sample_rate=22050)
        wandb_audios = [wandb_gen]
        if real_audio != None:
            wandb_real = wandb.Audio(real_audio[0].squeeze().detach().cpu().numpy(), caption="Real_1", sample_rate=22050)
            wandb_audios.append(wandb_real)
        wandb_gen2 = wandb.Audio(audio[1].squeeze().detach().cpu().numpy(), caption="Inference_2", sample_rate=22050)
        wandb_audios.append(wandb_gen2)
        if real_audio != None:
            wandb_real2 = wandb.Audio(real_audio[1].squeeze().detach().cpu().numpy(), caption="Real_2", sample_rate=22050)
            wandb_audios.append(wandb_real2)
        wandb.log({name: wandb_audios}, step=epoch)

def val(epoch):
    generator.eval()
    val_losses = []
    with torch.no_grad():
        i = 0
        for audio_b, mels_b, text_b in dataloader_val:
            if i%20 == 0:
                for param_group in optimizer.param_groups:
                    print(param_group['lr'])
            audio_b, mels_b, text_b = audio_b.to(device), mels_b.to(device), text_b.to(device)
            pad_mask = (audio_b != 0).to(device)
            res = generator(F.pad(audio_b[:, :, :-1], (1, 0)), mels_b)
            loss = criterion(res, mu_encode(audio_b).squeeze(1).type(torch.long))
            res = torch.argmax(res, dim=1)
            
            val_losses.append(loss.item())
            if i < 8:
                name = "example " + str(i)
                text = [chr(c) for c in text_b[0]]
                text = ''.join(text)
                wandb_gen = wandb.Audio(mu_decode(res[0]).detach().cpu().numpy(), caption="Generated", sample_rate=22050)
                wandb_real = wandb.Audio(audio_b[0].squeeze().detach().cpu().numpy(), caption="Real", sample_rate=22050)
                wandb_audios = [wandb_gen, wandb_real]
                wandb.log({name: wandb_audios}, step=epoch)
            if i > 500:
                break
            i += 1
    print('Val loss', np.mean(val_losses))
    wandb.log({"Val loss": np.mean(val_losses)}, step=epoch)

def run_inference():
    for audio_b, mels_b, text_b in dataloader_val:
        if i%20 == 0:
            for param_group in optimizer.param_groups:
                print(param_group['lr'])
        audio_b, mels_b, text_b = audio_b.to(device), mels_b.to(device), text_b.to(device)
        inference_fast(mels_b[:1], audio_b[:1], epoch=0)
        break

run_inference()
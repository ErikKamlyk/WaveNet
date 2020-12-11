import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import pandas as pd
import numpy as np
import random

class LJSpeech_Dataset(Dataset):
    def __init__(self, folder, csv, transform = None):
        super().__init__()
        self.csv = pd.read_csv(csv, sep='|', header=None)
        self.path = []
        self.labels = []
        for index, row in self.csv.iterrows():
            self.path.append(row[0])
            self.labels.append(str(row[2]))
        self.folder = folder
        self.transform = transform
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, i):
        original_wav, sr = torchaudio.load(self.folder + '/wavs/' + self.path[i] + '.wav')
        start_point = random.randrange(original_wav.shape[1] - 20000)
        original_wav = original_wav[:, start_point:start_point + 20000]
        text = self.labels[i]
        text = [ord(c) for c in text if ord(c) < 256]
        mels = self.transform(original_wav)
        mels = F.interpolate(mels, original_wav.shape[1], mode='linear')

        return original_wav, mels, torch.tensor(text)

def my_collate(batch):
    padded_audio = nn.utils.rnn.pad_sequence([item[0].view(1, -1).T for item in batch], batch_first=True)
    padded_mels = nn.utils.rnn.pad_sequence([item[1].view(80, -1).T for item in batch], batch_first=True)
    padded_text = nn.utils.rnn.pad_sequence([item[2] for item in batch], batch_first=True)
    return padded_audio.permute(0, 2, 1), padded_mels.permute(0, 2, 1), padded_text

def load_dataset(featurizer, BATCH_SIZE):
    train_data_path = './LJSpeech-1.1'
    test_data_path = './simple_image_classification/test'

    transform = featurizer
    dataset = LJSpeech_Dataset(train_data_path, train_data_path + '/metadata.csv', transform=transform)

    indices = list(range(len(dataset)))
    split = int(np.floor(len(dataset)*0.1))
    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    dataloader_train = DataLoader(dataset, batch_size=BATCH_SIZE, pin_memory=True, sampler=train_sampler, collate_fn=my_collate)
    dataloader_val = DataLoader(dataset, batch_size=BATCH_SIZE, pin_memory=True, sampler=val_sampler, collate_fn=my_collate)
    return dataloader_train, dataloader_val
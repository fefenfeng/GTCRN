import os
import toml
import random
import torch
import librosa

import pandas as pd
import soundfile as sf

from torch.utils import data


class MyDataset(data.Dataset):
    def __init__(self, train_folder, shuffle, num_tot, wav_len=0, n_fft=512, hop_length=256, win_length=512):
        super().__init__()

        # 从.txt文件中读取noisy和clean的filename pairs
        self.clean_list = []
        self.noisy_list = []

        with open(train_folder, 'r') as file:
            for line in file:
                # 去除每行的首尾空白字符（包括换行符）
                noisy_name, clean_name = line.strip().split()
                self.clean_list.append(clean_name)
                self.noisy_list.append(noisy_name)
        
        if shuffle:
            random.seed(7)
            indices = list(range(len(self.clean_list)))
            random.shuffle(indices)
            self.clean_list = [self.clean_list[i] for i in indices]
            self.noisy_list = [self.noisy_list[i] for i in indices]

        if num_tot != 0:
            self.clean_list = self.clean_list[: num_tot]
            self.noisy_list = self.noisy_list[: num_tot]
        
        self.train_folder = train_folder
        self.wav_len = wav_len

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

    def __getitem__(self, idx):
        clean, fs = librosa.load(self.clean_list[idx], sr=16000)
        noisy, fs = librosa.load(self.noisy_list[idx], sr=16000)

        noisy = torch.tensor(noisy)
        clean = torch.tensor(clean)

        if self.wav_len != 0:
            start = random.randint(0, max(0, len(clean) - int(self.wav_len * fs)))
            noisy = noisy[start: start + self.wav_len*fs]
            clean = clean[start: start + self.wav_len*fs]

        noisy = torch.stft(noisy, self.n_fft, self.hop_length, self.win_length, torch.hann_window(self.win_length).pow(0.5), return_complex=False)
        clean = torch.stft(clean, self.n_fft, self.hop_length, self.win_length, torch.hann_window(self.win_length).pow(0.5), return_complex=False)
        return noisy, clean
    
    def __len__(self):
        return len(self.clean_list)


if __name__ == '__main__':
    from tqdm import tqdm

    config = toml.load('config.toml')

    device = torch.device('cuda')

    train_dataset = MyDataset(**config['train_dataset'], **config['FFT'])
    train_dataloader = data.DataLoader(train_dataset, **config['train_dataloader'])
    
    validation_dataset = MyDataset(**config['validation_dataset'], **config['FFT'])
    validation_dataloader = data.DataLoader(validation_dataset, **config['validation_dataloader'])

    print(len(train_dataloader), len(validation_dataloader))

    for noisy, clean in tqdm(train_dataloader):
        print(noisy.shape, clean.shape)
        break

    for noisy, clean in tqdm(validation_dataloader):
        print(noisy.shape, clean.shape)
        break

    # train_bar = tqdm(train_dataloader, ncols=123)
    # for step, (mixture, target) in enumerate(train_bar, 1):
    #     print(step)
    #     print(mixture.shape, target.shape)
    #     if step == 10:
    #         break
    #
    # validation_bar = tqdm(validation_dataloader, ncols=123)
    # for step, (mixture, target) in enumerate(validation_bar, 1):
    #     print(step)
    #     print(mixture.shape, target.shape)
    #     if step == 10:
    #         break
 



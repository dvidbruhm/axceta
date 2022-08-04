from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from scipy import signal
from torchvision import transforms
from rich import print


class SiloFillDatasetv2(Dataset):
    def __init__(self):
        self.silo_data = 



class SiloFillDataset(Dataset):
    def __init__(self, xls_file, root_dir, transform=None, small_dataset=False):
        print(f"Loading dataset from : [bold green]{xls_file}[/bold green]")
        self.silo_data = pd.DataFrame(pd.read_excel(xls_file, sheet_name="data"))[:2534].drop([47])
        self.root_dir = root_dir
        self.silo_data = self.silo_data[self.silo_data.apply(lambda x: len(pd.DataFrame(pd.read_csv(Path(self.root_dir, x["filename"].split("/")[-1])))) > 1, axis=1)].reset_index()
        self.min_len = min([l for l in self.silo_data.apply(lambda x: len(pd.DataFrame(pd.read_csv(Path(self.root_dir, x["filename"].split("/")[-1])))), axis=1)])
        self.silo_data = self.silo_data[self.silo_data["TOF_ManuealReading"] < 30000].reset_index()
        if small_dataset:
            self.silo_data = self.silo_data[:int(len(self.silo_data) / 10)]
        self.transform = transform
        print(f"Done. There are [bold green]{len(self.silo_data)}[/bold green] data points.")


    def __len__(self):
        return len(self.silo_data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.to_list()

        raw_file_path = Path(self.silo_data.loc[idx, "filename"].split("/")[-1])
        raw_ultrasound = pd.DataFrame(pd.read_csv(Path(self.root_dir, raw_file_path))).values.squeeze()[:30600]#self.min_len]
        target_TOF = self.silo_data.loc[idx, "TOF_ManuealReading"] / len(raw_ultrasound)

        sample = raw_ultrasound, target_TOF
        if self.transform:
            sample = self.transform(sample)
        return sample

class Downsample(object):
    def __init__(self, downsample_factor):
        assert isinstance(downsample_factor, int)
        self.downsample_factor = downsample_factor

    def __call__(self, sample):
        ultrasound, target_TOF = sample[0], sample[1]
        downsampled_raw = signal.resample_poly(ultrasound.astype(float), 1, self.downsample_factor)
        return downsampled_raw, target_TOF

class ToTensor(object):
    def __call__(self, sample):
        ultrasound, target_TOF = sample[0], sample[1]
        ultrasound = torch.from_numpy(np.expand_dims(ultrasound, axis=0)).float()
        target_TOF = torch.tensor([target_TOF]).float()
        return ultrasound, target_TOF

def get_train_val_dataset(xls_path, data_path, transforms, small_dataset, train_size):
        full_silo_dataset = SiloFillDataset(xls_file=xls_path, root_dir=data_path, transform=transforms, small_dataset=small_dataset)
        train_len = int(len(full_silo_dataset) * train_size)
        val_len = len(full_silo_dataset) - train_len
        silo_train, silo_val = Subset(full_silo_dataset, range(0, train_len)), Subset(full_silo_dataset, range(train_len, len(full_silo_dataset)))#random_split(full_silo_dataset, [train_len, val_len])
        print(f"Splitting dataset. There are : \n\t[bold green]{len(silo_train)}[/bold green] training data and \n\t[bold green]{len(silo_val)}[/bold green] validation data.")
        return full_silo_dataset, silo_train, silo_val

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    trans = transforms.Compose([Downsample(10), ToTensor()])
    silo_dataset = SiloFillDataset(xls_file="data/LAFONTAINE-001acq_info.xlsx", root_dir="data/LAFONTAINE-001/", transform=trans)

    tofs = []
    for i in range(len(silo_dataset)):
        d, tof = silo_dataset[i]
        tofs.append(tof)
    plt.plot(tofs)
    plt.show()

    lens = []
    targets = []
    idxs = []

    for i in range(len(silo_dataset)):
        try:
            l = len(silo_dataset[i][0])
            t = silo_dataset[i][1]
            lens.append(l)
            targets.append(t)
            if l < 3000:
                print(l)
                idxs.append(i)
        except FileNotFoundError:
            print(i)
    print(idxs)
    print(f"Max len : {np.max(lens)}")
    print(f"Min len : {np.min(lens)}")
    print(f"Max target : {np.max(targets)}")
    print(f"Max target arg : {np.argmax(targets)}")
    print(f"Min target : {np.min(targets)}")
        
    dataloader = DataLoader(silo_dataset, batch_size=16, shuffle=True, num_workers=0)
    #for i_batch, sample_batched in enumerate(dataloader):
    #    print(i_batch, sample_batched['ultrasound'].size(), sample_batched['target_TOF'].size())


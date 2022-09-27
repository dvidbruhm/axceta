from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from scipy import signal
from torchvision import transforms
from rich import print
import us.data
import us.algos


class SiloFillDatasetParquet(Dataset):
    def __init__(self, parquet_files: List[Path], transform=None, small_dataset=False, target_name="LC_ToF"):
        print(f"\nLoading from these files: [bold green]{parquet_files}[/bold green]")
        dfs = []
        for parquet_file in parquet_files:
            print(f"Loading dataset from : [bold green]{parquet_file}[/bold green]")
            df = us.data.load_raw_LC_data(parquet_file)
            df = pd.DataFrame(df.sort_values(by="LC_AcquisitionTime"))
            df = pd.DataFrame(df.sort_values(by="AcquisitionTime")).reset_index()
            dfs.append(df)
        self.silo_data = pd.concat(dfs)

        # Keep only raw ultrasounds that are of high enough quality
        self.silo_data["raw_data_quality"] = self.silo_data.apply(lambda x: us.algos.raw_data_quality(x["sensor_raw_data"]), axis=1)
        self.silo_data = self.silo_data[self.silo_data["raw_data_quality"] > 2.5].reset_index()

        self.target_name = target_name
        self.small_dataset = small_dataset
        self.transform = transform

        if small_dataset:
            self.silo_data = self.silo_data[:int(len(self.silo_data) / 10)]

        self.min_len = min([l for l in self.silo_data.apply(lambda x: len(x["sensor_raw_data"]), axis=1)])
        self.min_len = 30600 #TEMP to make it fit with other dataset
        print(f"Done. There are [bold green]{len(self.silo_data)}[/bold green] data points.\n")

    def __len__(self):
        return len(self.silo_data)

    def __getitem__(self, idx):
        series = self.silo_data.loc[idx]
        raw_ultrasound = series["sensor_raw_data"][:self.min_len]
        target = series[self.target_name] / 2.0

        if "ToF" in self.target_name:
            target = target / len(raw_ultrasound)

        sample = raw_ultrasound, target

        if self.transform:
            sample = self.transform(sample)

        return sample


class SiloFillDatasetExcel(Dataset):
    def __init__(self, xls_file, root_dir, transform=None, small_dataset=False, target_name="TOF_ManuealReading"):
        print(f"\nLoading dataset from : [bold green]{xls_file}[/bold green]")
        self.silo_data = pd.DataFrame(pd.read_excel(xls_file, sheet_name="data"))[:2534].drop([47])
        self.root_dir = root_dir
        self.target_name = target_name
        self.silo_data = self.silo_data[self.silo_data.apply(lambda x: len(pd.DataFrame(pd.read_csv(Path(self.root_dir, x["filename"].split("/")[-1])))) > 1, axis=1)].reset_index()
        self.min_len = min([l for l in self.silo_data.apply(lambda x: len(pd.DataFrame(pd.read_csv(Path(self.root_dir, x["filename"].split("/")[-1])))), axis=1)])
        self.min_len = 30600 #TEMP to make it fit with other dataset
        self.silo_data = self.silo_data[self.silo_data["TOF_ManuealReading"] < 30000].reset_index()

        if small_dataset:
            self.silo_data = self.silo_data[:int(len(self.silo_data) / 10)]

        self.transform = transform
        print(f"Done. There are [bold green]{len(self.silo_data)}[/bold green] data points.\n")

    def __len__(self):
        return len(self.silo_data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.to_list()

        raw_file_path = Path(self.silo_data.loc[idx, "filename"].split("/")[-1])
        raw_ultrasound = pd.DataFrame(pd.read_csv(Path(self.root_dir, raw_file_path))).values.squeeze()[:self.min_len]
        target_TOF = self.silo_data.loc[idx, self.target_name]
        if self.target_name == "TOF_ManuealReading":
            target_TOF = target_TOF / len(raw_ultrasound)

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


def get_train_val_dataset(dataset, train_size):
        train_len = int(len(dataset) * train_size)
        val_len = len(dataset) - train_len
        silo_train, silo_val = Subset(dataset, range(0, train_len)), Subset(dataset, range(train_len, len(dataset)))#random_split(full_silo_dataset, [train_len, val_len])
        print(f"Splitting dataset. There are : \n\t[bold green]{len(silo_train)}[/bold green] training data and \n\t[bold green]{len(silo_val)}[/bold green] validation data.")
        return dataset, silo_train, silo_val

# v = d / t
# t = d / v
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    trans = transforms.Compose([Downsample(10), ToTensor()])
    silo_dataset_parquet = SiloFillDatasetParquet(parquet_files=[Path("data/Dashboards/1483A.csv.parquet")], transform=trans)

    steps = 0
    for i in range(0, len(silo_dataset_parquet), 100):
        steps += 1
        if steps > 10:
            exit()
        plt.plot(silo_dataset_parquet[i][0].view(-1))
        simple_max = np.argmax(silo_dataset_parquet[i][0].detach().numpy().squeeze()[400:]) + 400
        plt.axvline(simple_max)
        plt.show()

    exit()

    silo_dataset = SiloFillDatasetExcel(xls_file="data/LAFONTAINE-001acq_info.xlsx", root_dir="data/LAFONTAINE-001/", transform=trans)

    tofs = []
    for i in range(len(silo_dataset)):
        print(i)
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


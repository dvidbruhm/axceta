from pathlib import Path
from typing import List
from pytorch_lightning import LightningModule, Trainer, seed_everything
import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader, random_split, Subset, Dataset
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBar
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import matplotlib.pyplot as plt
from rich import print

import us.ml.data as data
import us.ml.utils as utils

import warnings
warnings.filterwarnings("ignore", ".*does not have many workers.*")


class SiloFillRegressor(LightningModule):
    def __init__(self, dataset: Dataset, epochs: int, learning_rate: float, batch_size: int, kernels, train_size: float = 0.5, paths: List[Path] = []):
        super().__init__()

        self.dataset = dataset
        self.train_size = train_size

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.kernels = kernels
        self.paths = paths

        self.save_hyperparameters(logger=True, ignore=["dataset"])
        print(f"\nHyperparameters : \n[bold green]{self.hparams}[/bold green]\n")
        self.conv1 = nn.Conv1d(in_channels=self.kernels[0], out_channels=self.kernels[1], kernel_size=5, stride=2)
        self.leaky1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv1d(in_channels=self.kernels[1], out_channels=self.kernels[2], kernel_size=5, stride=2)
        self.batch1 = nn.BatchNorm1d(self.kernels[2])
        self.leaky2 = nn.LeakyReLU(0.2, inplace=True)
        self.conv3 = nn.Conv1d(in_channels=self.kernels[2], out_channels=self.kernels[3], kernel_size=5, stride=2)
        self.leaky3 = nn.LeakyReLU(0.2, inplace=True)
        self.conv4 = nn.Conv1d(in_channels=self.kernels[3], out_channels=self.kernels[4], kernel_size=5, stride=2)
        self.batch2 = nn.BatchNorm1d(self.kernels[4])
        self.leaky4 = nn.LeakyReLU(0.2, inplace=True)

        self.linear1 = nn.Linear(self.kernels[4] * 188, 1)
        self.sigmoid1 = nn.Sigmoid()

        self.main = nn.Sequential(
            self.conv1,
            self.leaky1,
            self.conv2,
            self.batch1,
            self.leaky2,
            self.conv3,
            self.leaky3,
            self.conv4,
            self.batch2,
            self.leaky4
        )

    def forward(self, inputs):
        x = self.main(inputs)
        x = torch.flatten(x, 1)
        x = self.sigmoid1(self.linear1(x))
        return x

    def loss_fn(self, y_true, y_pred):
        mse_loss = nn.MSELoss()
        return mse_loss(y_pred, y_true)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.view(-1, 1)
        outputs = self(x)
        loss = self.loss_fn(y, outputs)
        self.log("train_loss", loss, logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.view(-1, 1)
        outputs = self(x)
        loss = self.loss_fn(y, outputs)

        self.log("val_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("hp_metric", loss)

    def on_fit_end(self):
        print("Creating training and validation TOF visualisation.")
        fig = viz_TOF(self, self.silo_train, "Train", return_fig=True)
        self.logger.experiment.add_figure("TOF visualisation Train", fig)
        fig = viz_TOF(self, self.silo_val, "Validation", return_fig=True)
        self.logger.experiment.add_figure("TOF visualisation Validation", fig)
        print("Done.")


    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            _, self.silo_train, self.silo_val = data.get_train_val_dataset(self.dataset, self.train_size)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def train_dataloader(self):
        return DataLoader(self.silo_train, batch_size=self.batch_size, shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.silo_val, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        pass


def train_excel(data_path: Path, xls_path: Path, epochs: int, learning_rate: float, batch_size: int, kernels: List[int], train_size: float, small_dataset: bool):
    seed_everything(1234)

    trans = transforms.Compose([data.Downsample(10), data.ToTensor()])
    silo_dataset = data.SiloFillDatasetExcel(xls_file=xls_path, root_dir=data_path, transform=trans, small_dataset=small_dataset)
    paths = [data_path, xls_path]
    train(silo_dataset, epochs, learning_rate, batch_size, kernels, train_size, paths, "silo_tof_excel")


def train_parquet(data_path: Path, epochs: int, learning_rate: float, batch_size: int, kernels: List[int], train_size: float, small_dataset: bool):
    seed_everything(1234)
    trans = transforms.Compose([data.Downsample(10), data.ToTensor()])
    silo_dataset = data.SiloFillDatasetParquet(data_path, trans, small_dataset)
    paths = [data_path]
    train(silo_dataset, epochs, learning_rate, batch_size, kernels, train_size, paths, "silo_tof_parquet")


def train(dataset: Dataset, epochs: int, learning_rate: float, batch_size: int, kernels: List[int], train_size: float, paths: List[Path], logger_name: str):

    model = SiloFillRegressor(
        dataset,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        kernels=kernels,
        train_size=train_size,
        paths=paths
    )

    early_stop_callback = EarlyStopping(monitor="val_loss", mode="min", patience=5)

    trainer = Trainer(
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
        max_epochs=epochs,
        callbacks=[RichProgressBar()],
        logger=TensorBoardLogger(str(Path("us", "ml", "logs")), name=logger_name, default_hp_metric=False),
        log_every_n_steps=1
    )

    trainer.fit(model)


def viz_excel(model_path: Path):
    from us.plot import plot_full_excel
    from us.data import load_excel_data
    import pandas as pd

    if model_path == Path():
        model_path = utils.find_last_checkpoint()

    print(f"Loading model from : [bold green]{model_path}[/bold green]")
    model = SiloFillRegressor.load_from_checkpoint(str(model_path))
    print("Done.")

    silo_dataset, silo_train, silo_val = data.get_train_val_dataset(model.xls_path, model.data_path, model.trans, model.small_dataset, model.train_size)

    df_excel = pd.DataFrame(pd.read_excel(model.xls_path, sheet_name="data"))[:2534].drop([47])
    root_dir = model.data_path
    df_excel = df_excel[df_excel.apply(lambda x: len(pd.DataFrame(pd.read_csv(Path(root_dir, x["filename"].split("/")[-1])))) > 1, axis=1)].reset_index()
    df_excel = df_excel[df_excel["TOF_ManuealReading"] < 30000].reset_index()


    viz_TOF(model, silo_dataset, title="LAFONTAINE-001", return_fig=True)
    cm_index = df_excel["measured_distance_in_mm"] * 2* 1000000 / 1000 / df_excel["sound_speed"]
    wf_index = df_excel["wavefront_distance_in_mm"] *2*  1000000 / 1000 / df_excel["sound_speed"]
    plt.plot(cm_index, label="CM TOF", color="red")
    plt.plot(wf_index, label="WF TOF", color="blue")
    plt.legend(loc="best")
    plt.xlabel("Time")
    plt.ylabel("Time of flight index")
    plt.show()
    exit()

    viz_single_ultrasounds(model, silo_val, [0, 5, 10, 15, 20])

    viz_TOF(model, silo_train, title="Train")
    viz_TOF(model, silo_val, title="Val")


def viz_TOF(model, dataset, title, return_fig=False):
    targets = []
    preds = []
    for i in range(len(dataset)):
        input_data, target = dataset[i]
        target = target.detach().numpy()[0] * input_data.shape[1] * 10
        input_data = input_data.view(1, 1, -1)
        pred = model(input_data).detach().numpy()[0] * input_data.shape[2] * 10

        preds.append(pred[0])
        targets.append(target)

    plt.title(title)
    plt.plot(targets, color="green", label="True TOF")
    plt.plot(utils.moving_average(preds, 15), color="magenta", label="Machine Learning")
    plt.legend(loc="best")
    if return_fig:
        return plt.gcf()
    else:
        plt.show()


def viz_single_ultrasounds(model, dataset, idxs):
    for i, idx in enumerate(idxs):
        plt.subplot(len(idxs), 1, i + 1)
        input_data = dataset[idx]
        target = input_data[1].detach()
        input_data = input_data[0].view(1, 1, -1)
        output = model(input_data).detach()

        input_data = input_data.detach().numpy().squeeze()

        plt.plot(input_data)
        plt.axvline(output.numpy()[0] * len(input_data), color="orange", label="Prediction")
        plt.axvline(target.numpy()[0] * len(input_data), color="green", label="Target")
        plt.legend(loc="best")
    plt.show()

from pathlib import Path
from pytorch_lightning import LightningModule, Trainer, seed_everything
import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader, random_split, Subset
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBar
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import CSVLogger
import matplotlib.pyplot as plt
from rich import print

import us.ml.data as data
import us.ml.utils as utils

import warnings
warnings.filterwarnings("ignore", ".*does not have many workers.*")


class SiloFillRegressor(LightningModule):
    def __init__(self, data_path: Path, xls_path: Path, small_dataset: bool = False, train_size: float = 0.5):
        super().__init__()
        self.save_hyperparameters()

        self.data_path = data_path
        self.xls_path = xls_path
        self.small_dataset = small_dataset
        self.trans = transforms.Compose([data.Downsample(10), data.ToTensor()])
        self.train_size = train_size

        self.batch_size = 16
        self.learning_rate = 0.00001

        kernels = [1, 8, 16, 24, 32]

        self.conv1 = nn.Conv1d(in_channels=kernels[0], out_channels=kernels[1], kernel_size=5, stride=2)
        self.leaky1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv1d(in_channels=kernels[1], out_channels=kernels[2], kernel_size=5, stride=2)
        self.batch1 = nn.BatchNorm1d(kernels[2])
        self.leaky2 = nn.LeakyReLU(0.2, inplace=True)
        self.conv3 = nn.Conv1d(in_channels=kernels[2], out_channels=kernels[3], kernel_size=5, stride=2)
        self.leaky3 = nn.LeakyReLU(0.2, inplace=True)
        self.conv4 = nn.Conv1d(in_channels=kernels[3], out_channels=kernels[4], kernel_size=5, stride=2)
        self.batch2 = nn.BatchNorm1d(kernels[4])
        self.leaky4 = nn.LeakyReLU(0.2, inplace=True)

        self.linear1 = nn.Linear(kernels[4] * 188, 1)
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
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.view(-1, 1)
        outputs = self(x)
        loss = self.loss_fn(y, outputs)

        self.log("val_loss", loss, prog_bar=True)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            _, self.silo_train, self.silo_val = data.get_train_val_dataset(self.xls_path, self.data_path, self.trans, self.small_dataset, self.train_size)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def train_dataloader(self):
        return DataLoader(self.silo_train, batch_size=self.batch_size, shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.silo_val, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        pass


def train(data_path: Path, xls_path: Path, epochs: int, train_size: float, small_dataset: bool):
    seed_everything(1234)
    model = SiloFillRegressor(data_path, xls_path, small_dataset=small_dataset, train_size=train_size)

    early_stop_callback = EarlyStopping(monitor="val_loss", mode="min", patience=5)
    trainer = Trainer(
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
        max_epochs=epochs,
        callbacks=[RichProgressBar()],
        logger=CSVLogger(save_dir="us/ml/logs/")
    )

    trainer.fit(model)


def viz(model_path: Path):
    if model_path == Path():
        model_path = utils.find_last_checkpoint()

    print(f"Loading model from : [bold green]{model_path}[/bold green]")
    model = SiloFillRegressor.load_from_checkpoint(str(model_path))
    print(f"Hyperparameters : \n[bold green]{model.hparams}[/bold green]")
    print("Done.")

    silo_dataset, silo_train, silo_val = data.get_train_val_dataset(model.xls_path, model.data_path, model.trans, model.small_dataset, model.train_size)

    viz_single_ultrasounds(model, silo_val, [0, 5, 10, 15, 20])

    viz_TOF(model, silo_train, title="Train")
    viz_TOF(model, silo_val, title="Val")


def viz_TOF(model, dataset, title):
    targets = []
    preds = []
    for i in range(len(dataset)):
        input_data, target = dataset[i]
        input_data = input_data.view(1, 1, -1)
        pred = model(input_data).detach().numpy()[0]
        target = target.detach().numpy()[0]

        preds.append(pred)
        targets.append(target)

    plt.title(title)
    plt.plot(targets, color="green", label="Targets")
    plt.plot(preds, color="orange", label="Preds")
    plt.legend(loc="best")
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

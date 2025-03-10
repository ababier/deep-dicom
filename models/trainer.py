import torch
from prediction.create_data_splits import DataSplit
from prediction.sample_config import SampleConfig
from torch import Tensor, optim
from torch.amp import GradScaler, autocast
from torch.nn import MSELoss
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.dataset import LargeImageDataset
from models.tensorboard_logger import TensorBoardLogger
from models.transformations import RotateAndFlip
from models.unet import UNet


class Trainer:
    def __init__(self, plan_type: str, batch_size: int = 2, learning_rate: float = 1e-4):
        """Initialize Trainer."""
        self.batch_size = batch_size
        self.plan_type = plan_type
        self.learning_rate = learning_rate

        self._device = self._setup_device()
        self._scaler = GradScaler()
        self.tb_logger = TensorBoardLogger(f"runs/unet_experiment_{plan_type}")

        data_splits = DataSplit.read(self.plan_type)
        self.data_config = SampleConfig.read(self.plan_type)
        training_data = LargeImageDataset(self.data_config, data_splits.train_ids, transform=RotateAndFlip())
        self._training_data_loader = DataLoader(training_data, batch_size=self.batch_size, shuffle=True, num_workers=4)
        validation_data = LargeImageDataset(self.data_config, data_splits.validation_ids)
        self._validation_data_loader = DataLoader(validation_data, batch_size=self.batch_size, shuffle=False, num_workers=4)
        self.model, self.loss_fn, self.optimizer, self.scheduler = self._initialize_model()

    @staticmethod
    def _setup_device() -> torch.device:
        """Setup device (GPU if available)."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("Using GPU:", torch.cuda.get_device_name(0))
        else:
            device = torch.device("cpu")
            print("Using CPU")
        return device

    def _initialize_model(self):
        """Init UNet, loss, optimizer, and LR scheduler."""
        model = UNet(num_inputs=len(self.data_config.input_names), num_outputs=len(self.data_config.output_names), base_channels=32)
        model.initialize_weights()
        model = model.to(self._device)
        print(f"Model params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

        loss_fn = MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        scheduler = ExponentialLR(optimizer=optimizer, gamma=0.98)
        return model, loss_fn, optimizer, scheduler

    def train_epochs(self, num_epochs: int = 100, save_path: str = "unet_model.pth"):
        """Train for given epochs."""
        for epoch in range(1, num_epochs + 1):
            self._train_epoch(epoch)
            self._validate_epoch(epoch)
            torch.cuda.empty_cache()
            self.scheduler.step()

            if epoch % 10 == 0:
                torch.save(self.model.state_dict(), f"{save_path}_epoch_{epoch}.pth")

        self.tb_logger.close()
        torch.save(self.model.state_dict(), save_path)

    def _train_epoch(self, epoch: int):
        """Train one epoch."""
        self.model.train()
        running_loss = 0.0

        for batch_idx, batch in tqdm(enumerate(self._training_data_loader), desc=f"Epoch {epoch}", unit=" batch"):
            inputs = self.get_inputs(batch)
            targets = self.get_targets(batch)

            self.optimizer.zero_grad()
            with autocast(device_type=self._device.type, dtype=torch.bfloat16):
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)

            self._scaler.scale(loss).backward()
            self._scaler.unscale_(self.optimizer)
            self._scaler.step(self.optimizer)
            self._scaler.update()

            running_loss += loss.item()
            self.tb_logger.add_scalar("Loss/train", loss.item(), epoch * len(self._training_data_loader) + batch_idx)
            if batch_idx % 10 == 0:
                self.log_images_to_tensorboard(batch, outputs, epoch, batch_idx, len(self._training_data_loader), "Training")

        print(f"Epoch {epoch}, Training Loss: {running_loss / len(self._training_data_loader)}")

    def _validate_epoch(self, epoch: int):
        """Validate one epoch."""
        self.model.eval()
        val_loss = 0.0

        original_checkpoint_mode = self.model.use_checkpoint
        self.model.use_checkpoint = False

        with torch.no_grad():
            for batch_idx, batch in enumerate(self._validation_data_loader):
                inputs = self.get_inputs(batch)
                targets = self.get_targets(batch)
                with autocast(device_type=self._device.type, dtype=torch.bfloat16):
                    outputs = self.model(inputs)
                    loss = self.loss_fn(outputs, targets)
                val_loss += loss.item()

                if batch_idx % 3 == 0:
                    self.log_images_to_tensorboard(batch, outputs, epoch, batch_idx, len(self._validation_data_loader))

            val_loss /= len(self._validation_data_loader)
            self.tb_logger.add_scalar("Loss/val", val_loss, epoch)
            print(f"Epoch {epoch}, Validation Loss: {val_loss}")

        self.model.use_checkpoint = original_checkpoint_mode

    def get_inputs(self, batch: dict[str, Tensor]) -> Tensor:
        """Return concatenated input tensor."""
        return torch.cat([batch[name].to(torch.bfloat16) for name in self.data_config.input_names], dim=1).to(self._device, non_blocking=True)

    def get_targets(self, batch: dict[str, Tensor]) -> Tensor:
        """Return concatenated target tensor."""
        return torch.cat([batch[name].to(torch.bfloat16) for name in self.data_config.output_names], dim=1).to(self._device, non_blocking=True)

    def log_images_to_tensorboard(self, batch, outputs, epoch, batch_idx, data_size, name: str = "Validation"):
        """Log images using the TensorBoardLogger."""
        self.tb_logger.log_images(batch, outputs, epoch, batch_idx, data_size, name)

import matplotlib.cm as cm
import torch
import torchvision.utils as vutils
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter


class TensorBoardLogger:
    def __init__(self, log_dir: str):
        self.writer = SummaryWriter(log_dir)

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        self.writer.add_scalar(tag, value, step)

    def log_images(self, batch: dict[str, Tensor], outputs: Tensor, epoch: int, batch_idx: int, data_size: int, name: str = "Validation") -> None:
        idx = 0
        ct_image = batch["ct"][idx]
        target_dose = batch["dose"][idx]

        dose_sum = target_dose.sum(dim=(-2, -1))
        index_with_most_dose = torch.argmax(dose_sum).squeeze()

        ct_slice = ct_image[0, index_with_most_dose, ...].squeeze()
        target_dose_slice = target_dose[0, index_with_most_dose, ...].squeeze()
        predicted_dose_slice = outputs[idx, 0, index_with_most_dose, ...].squeeze()
        ct_rgb = ct_slice.expand(3, -1, -1)

        target_dose_rgb = self.convert_to_heatmap(target_dose_slice)
        predicted_dose_rgb = self.convert_to_heatmap(predicted_dose_slice)

        combined_image = vutils.make_grid([ct_rgb, target_dose_rgb.squeeze(0), predicted_dose_rgb.squeeze(0)], nrow=3)
        self.writer.add_image(f"{name} (CT | Target Dose | Predicted Dose)", combined_image, epoch * data_size + batch_idx)

    @staticmethod
    def convert_to_heatmap(tensor: Tensor) -> Tensor:
        """Convert a tensor to an RGB heatmap."""
        colormap = cm.get_cmap("jet")
        tensor_np = tensor.detach().cpu().to(torch.float32).numpy()
        heatmap_np = colormap(tensor_np)[..., :3]  # Keep only RGB channels.
        heatmap_tensor = torch.tensor(heatmap_np, dtype=torch.float32).unsqueeze(0)
        heatmap_tensor = heatmap_tensor.permute(0, 3, 1, 2)
        return heatmap_tensor

    def close(self) -> None:
        self.writer.close()

"""Optional PyTorch image models on the shared partition manifest."""

import logging
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .evaluate import score_predictions
from .preprocessing import preprocess

LOG = logging.getLogger(__name__)


def build_network(name: str, pretrained: bool = False) -> nn.Module:
    if name == "resnet18":
        from torchvision.models import ResNet18_Weights, resnet18

        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, 2)
        return model
    if name != "cnn":
        raise ValueError(f"Unsupported network: {name}")
    return nn.Sequential(
        nn.Conv2d(3, 24, 5, stride=2, padding=2),
        nn.BatchNorm2d(24),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(24, 48, 3, padding=1),
        nn.BatchNorm2d(48),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(48, 96, 3, padding=1),
        nn.BatchNorm2d(96),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(96, 128, 3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((2, 2)),
        nn.Flatten(),
        nn.Dropout(0.20),
        nn.Linear(512, 2),
    )


def image_tensor(rgb: np.ndarray, name: str) -> torch.Tensor:
    pixels = rgb.astype(np.float32) / 255
    if name == "resnet18":
        pixels -= np.array([0.485, 0.456, 0.406], dtype=np.float32)
        pixels /= np.array([0.229, 0.224, 0.225], dtype=np.float32)
    else:
        pixels = pixels * 2 - 1
    return torch.from_numpy(pixels.transpose(2, 0, 1).copy())


class Leaves(Dataset):
    def __init__(self, root, samples, size, name, augment=False):
        self.root, self.samples, self.size, self.name = root, samples, size, name
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        rgb = preprocess(self.root / sample.path, self.size)
        pixels = image_tensor(rgb, self.name)
        if self.augment and torch.rand(()).item() < 0.5:
            pixels = pixels.flip(-1)
        return pixels, sample.label


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if (requested == "mps" and not torch.backends.mps.is_available()) or (
        requested == "cuda" and not torch.cuda.is_available()
    ):
        raise ValueError(
            f"Requested {requested} device is unavailable. Use --device cpu."
        )
    return torch.device(requested)


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    model.train(optimizer is not None)
    loss_sum, weight_sum, labels, scores = 0.0, 0.0, [], []
    with torch.set_grad_enabled(optimizer is not None):
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            loss = loss_fn(logits, targets)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                labels.extend(targets.cpu().tolist())
                scores.extend(logits.softmax(dim=1)[:, 1].cpu().tolist())
            weight = loss_fn.weight[targets].sum().item()
            loss_sum += loss.item() * weight
            weight_sum += weight
    return loss_sum / weight_sum, np.asarray(labels), np.asarray(scores)


def train_network(root, splits, cfg, name, requested_device, checkpoint: Path):
    """Save the epoch with the highest primary validation metric."""
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = resolve_device(requested_device)
    if device.type == "cpu":
        torch.set_num_threads(min(torch.get_num_threads(), 4))
    train = Leaves(root, splits["train"], cfg.image_size, name, augment=True)
    val = Leaves(root, splits["validation"], cfg.image_size, name)
    training = DataLoader(
        train,
        batch_size=cfg.cnn_batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    validation = DataLoader(val, batch_size=cfg.cnn_batch_size)
    counts = Counter(s.label for s in train.samples)
    weights = torch.tensor(
        [len(train) / (2 * counts[i]) for i in (0, 1)], device=device
    )
    model = build_network(name, pretrained=name == "resnet18").to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.cnn_learning_rate)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    history = {
        "train_loss": [],
        "validation_loss": [],
        "validation_primary_metric": [],
        "best_epoch": None,
    }
    best, stale = None, 0
    for epoch in range(cfg.cnn_epochs):
        train_loss, _, _ = run_epoch(model, training, loss_fn, device, optimizer)
        val_loss, labels, scores = run_epoch(model, validation, loss_fn, device)
        metrics = score_predictions(labels, (scores > 0.5).astype(int), scores)
        metric = cfg.primary_metric
        history["train_loss"].append(train_loss)
        history["validation_loss"].append(val_loss)
        history["validation_primary_metric"].append(metrics[metric])
        LOG.info(
            "%s epoch %d/%d | loss %.3f / %.3f | %s %.4f",
            name,
            epoch + 1,
            cfg.cnn_epochs,
            train_loss,
            val_loss,
            metric,
            metrics[metric],
        )
        if best is None or metrics[metric] > best[metric]:
            best, stale = metrics, 0
            history["best_epoch"] = epoch + 1
            weights_on_cpu = {
                key: value.detach().cpu().contiguous()
                for key, value in model.state_dict().items()
            }
            save_file(weights_on_cpu, checkpoint)
            checkpoint.chmod(0o644)
        else:
            stale += 1
            if stale >= cfg.cnn_patience:
                break
    return best, history, str(device)


def restore_model(path: Path, name: str) -> nn.Module:
    model = build_network(name)
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    model.load_state_dict(load_file(path))
    return model.eval()


@torch.inference_mode()
def infer_rgb(model, rgb, name):
    return model(image_tensor(rgb, name).unsqueeze(0)).softmax(dim=1)[0].numpy()


@torch.inference_mode()
def predict_samples(model, root, samples, size, name):
    loader = DataLoader(Leaves(root, samples, size, name), batch_size=64)
    scores = np.concatenate([model(x).softmax(dim=1)[:, 1].numpy() for x, _ in loader])
    return (scores > 0.5).astype(int), scores

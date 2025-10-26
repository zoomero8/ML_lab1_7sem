import copy
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm
import os
from torchvision.utils import save_image


# --- 1. Конфигурация проекта и воспроизводимость ---

PROJECT_DIR = Path(__file__).parent.parent
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def set_seed(seed_value=42):
    """Устанавливает seed для всех генераторов случайных чисел для воспроизводимости."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# --- 2. Дата-классы для параметров ---

@dataclass
class DataConfig:
    data_path: Path = PROJECT_DIR / "data" / "raw"
    img_size: int = 224
    batch_size: int = 16
    val_split: float = 0.2
    num_workers: int = 2


@dataclass
class TrainConfig:
    model_name: str = "resnet18"
    # Наша лучшая модель по итогам экспериментов
    num_classes: int = 3
    learning_rate: float = 0.001
    num_epochs: int = 20
    output_model_path: Path = PROJECT_DIR / "models" / "best_model.onnx"


# --- 3. Функции для данных и обучения ---

def get_dataloaders(config: DataConfig):
    """Создает и возвращает обучающий и валидационный загрузчики данных."""
    data_transforms = {
        'train': transforms.Compose([
                 transforms.RandomResizedCrop(config.img_size, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
                 transforms.RandomHorizontalFlip(),
                 transforms.ToTensor(),
                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(config.img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    full_dataset = datasets.ImageFolder(config.data_path)

    num_data = len(full_dataset)
    num_val = int(num_data * config.val_split)
    num_train = num_data - num_val

    train_dataset, val_dataset = random_split(full_dataset, [num_train, num_val])

    # Динамически применяем трансформации к нужным подвыборкам
    train_dataset.dataset = copy.copy(full_dataset)
    train_dataset.dataset.transform = data_transforms['train']
    val_dataset.dataset.transform = data_transforms['val']

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers),
        'val': DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    }

    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}
    class_names = full_dataset.classes

    print(f"Классы найдены: {class_names}")
    print(f"Размер обучающей выборки: {dataset_sizes['train']}")
    print(f"Размер валидационной выборки: {dataset_sizes['val']}")

    return dataloaders, dataset_sizes, class_names


def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=20):
    """Основной цикл обучения модели."""
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(dataloaders[phase], desc=f"{phase.capitalize()} Phase"):
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        print()

    time_elapsed = time.time() - since
    print(f'Обучение завершено за {time_elapsed // 60:.0f}м {time_elapsed % 60:.0f}с')
    print(f'Лучшая точность на валидации: {best_acc:4f}')

    model.load_state_dict(best_model_wts)
    return model


def export_to_onnx(model, config: TrainConfig, image_size: int):
    """Экспортирует обученную модель в формат ONNX."""
    # Убеждаемся, что папка для сохранения модели существует
    config.output_model_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    dummy_input = torch.randn(1, 3, image_size, image_size, device=DEVICE)

    torch.onnx.export(
        model,
        dummy_input,
        config.output_model_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"\nМодель успешно экспортирована в {config.output_model_path}")


def save_augmented_samples(dataloader, save_dir, num_batches=1):
    """
    Сохраняет примеры изображений после трансформаций.
    save_dir — папка, куда будут сохраняться картинки.
    num_batches — сколько батчей сохранить (по умолчанию 1).
    """
    os.makedirs(save_dir, exist_ok=True)
    print(f"Сохраняю аугментированные изображения в {save_dir}")

    batch_count = 0
    for inputs, labels in dataloader:
        for i, img in enumerate(inputs):
            filename = f"batch{batch_count}_img{i}_label{labels[i].item()}.jpg"
            filepath = os.path.join(save_dir, filename)
            # изображения нормализованы, поэтому "разнормализуем" их для просмотра
            img_show = img * torch.tensor([0.229, 0.224, 0.225]).view(3,1,1) + torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
            save_image(img_show, filepath)
        batch_count += 1
        if batch_count >= num_batches:
            break

    print("Сохранено")

# --- 4. Основная функция ---

def main():
    """Главный пайплайн: настройка, обучение, экспорт."""
    set_seed()
    data_cfg = DataConfig()
    train_cfg = TrainConfig()

    print(f"Используемое устройство: {DEVICE}")

    # 1. Загрузка данных
    dataloaders, dataset_sizes, class_names = get_dataloaders(data_cfg)
    save_augmented_samples(dataloaders['train'], PROJECT_DIR / "data" / "augmented_samples", num_batches=2)
    if len(class_names) != train_cfg.num_classes:
        raise ValueError(f"Ошибка! Найдено {len(class_names)} классов, а в конфиге {train_cfg.num_classes}")

    # 2. Настройка модели
    model = timm.create_model(train_cfg.model_name, pretrained=True, num_classes=train_cfg.num_classes)
    model = model.to(DEVICE)

    # Замораживаем все слои
    for param in model.parameters():
        param.requires_grad = False

    # Размораживаем только "голову" (для ResNet это 'fc')
    for param in model.fc.parameters():
        param.requires_grad = True

    # 3. Настройка обучения
    criterion = nn.CrossEntropyLoss()
    params_to_update = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params_to_update, lr=train_cfg.learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # 4. Обучение
    best_model = train_model(
        model, criterion, optimizer, scheduler,
        dataloaders, dataset_sizes, num_epochs=train_cfg.num_epochs
    )

    # 5. Экспорт
    export_to_onnx(best_model, train_cfg, data_cfg.img_size)


if __name__ == '__main__':
    main()
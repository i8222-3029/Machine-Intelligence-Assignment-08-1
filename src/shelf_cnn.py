
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import copy
import os

# --- 데이터 로드 및 numpy 기반 split (재현성 보장) ---
data = np.load('shelf_images.npz')
images = data['images']  # (900, 64, 64)
labels = data['labels']  # (900,)
class_names = list(data['class_names'])

# numpy로 shuffle 후 split
rng = np.random.default_rng(42)
idx = rng.permutation(len(images))
n_test = int(0.15 * len(images))
n_val = int(0.15 * len(images))
n_train = len(images) - n_test - n_val
train_idx = idx[:n_train]
val_idx = idx[n_train:n_train + n_val]
test_idx = idx[n_train + n_val:]

X_train, y_train = images[train_idx], labels[train_idx]
X_val, y_val = images[val_idx], labels[val_idx]
X_test, y_test = images[test_idx], labels[test_idx]

# --- Dataset 정의 ---
class ShelfDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        img = self.images[idx]
        img = np.expand_dims(img, 0)  # (1, 64, 64)
        img = torch.from_numpy(img).float()
        if self.transform:
            img = self.transform(img)
        label = int(self.labels[idx])
        return img, label

# --- Data Augmentation (train만 적용) ---
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
])
val_test_transform = None

train_set = ShelfDataset(X_train, y_train, transform=train_transform)
val_set = ShelfDataset(X_val, y_val, transform=val_test_transform)
test_set = ShelfDataset(X_test, y_test, transform=val_test_transform)

def get_loader(ds, batch_size=64, shuffle=False):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

train_loader = get_loader(train_set, shuffle=True)
val_loader = get_loader(val_set)
test_loader = get_loader(test_set)

def get_loader(ds, batch_size=64, shuffle=False):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

train_loader = get_loader(train_set, shuffle=True)
val_loader = get_loader(val_set)
test_loader = get_loader(test_set)

# 2. CNN 아키텍처 정의 (3 conv, BN, Dropout, FC)
# 2. CNN 아키텍처 정의 (3 conv, BN, Dropout, FC)
class ShelfCNN(nn.Module):
    def __init__(self, dropout_p=0.5, use_bn=True, n_conv=3, filters=[16,32,64]):
        super().__init__()
        layers = []
        in_c = 1
        for i in range(n_conv):
            out_c = filters[i]
            layers.append(nn.Conv2d(in_c, out_c, 3, padding=1))
            if use_bn:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2))
            in_c = out_c
        self.conv = nn.Sequential(*layers)
        self.flatten = nn.Flatten()
        # 64x64 -> 32x32 -> 16x16 -> 8x8 (3 conv)
        fc_in = filters[n_conv-1] * (64 // (2**n_conv))**2
        self.fc1 = nn.Linear(fc_in, 128)
        self.dropout = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# --- Fully Connected Baseline ---
class ShelfFC(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
        )
    def forward(self, x):
        return self.net(x)

# 3. 학습/평가 루프


# --- 학습/평가 루프 (epoch 0: 평가만, early stopping, best 복원) ---
def train_model(model, train_loader, val_loader, epochs=100, lr=1e-3, weight_decay=0, patience=15, device='cpu'):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    best_model = copy.deepcopy(model.state_dict())
    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    patience_counter = 0

    # epoch 0: 평가만 (chance level)
    model.eval()
    total, correct, running_loss = 0, 0, 0.0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            running_loss += loss.item() * x.size(0)
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += x.size(0)
    val_loss = running_loss / total
    val_acc = correct / total
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    train_losses.append(val_loss)  # epoch 0은 train loss 없음, val loss로 대체

    for epoch in range(1, epochs+1):
        # --- Training ---
        model.train()
        total, correct, running_loss = 0, 0, 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += x.size(0)
        train_loss = running_loss / total
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # --- Validation ---
        model.eval()
        total, correct, running_loss = 0, 0, 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                running_loss += loss.item() * x.size(0)
                pred = out.argmax(1)
                correct += (pred == y).sum().item()
                total += x.size(0)
        val_loss = running_loss / total
        val_acc = correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Epoch {epoch:3d}: train loss {train_loss:.4f}, val loss {val_loss:.4f}, train acc {train_acc:.3f}, val acc {val_acc:.3f}")

        # Early stopping (val_loss 기준)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    model.load_state_dict(best_model)
    return model, (train_losses, val_losses, train_accs, val_accs)

# 4. 실험: 아키텍처 변형 (2,3,4 conv, 필터 수)
def experiment_architectures():
    configs = [
        {'n_conv':2, 'filters':[16,32]},
        {'n_conv':3, 'filters':[16,32,64]},
        {'n_conv':4, 'filters':[8,16,32,64]},
        {'n_conv':3, 'filters':[8,16,32]},
        {'n_conv':3, 'filters':[32,64,128]},
    ]
    results = []
    for cfg in configs:
        print(f"\nConfig: {cfg}")
        model = ShelfCNN(n_conv=cfg['n_conv'], filters=cfg['filters'])
        model, (train_losses, val_losses, train_accs, val_accs) = train_model(model, train_loader, val_loader, epochs=30, lr=1e-3, patience=5)
        results.append({'config':cfg, 'val_acc':max(val_accs)})
    print("\nArchitecture comparison:")
    for r in results:
        print(f"{r['config']}: val acc={r['val_acc']:.3f}")
    return results

# 5. 정규화 툴킷 적용 (Dropout, Weight Decay, Data Aug, Early Stopping)
def run_with_regularization():
    model = ShelfCNN(dropout_p=0.5, use_bn=True, n_conv=3, filters=[16,32,64])
    model, (train_losses, val_losses, train_accs, val_accs) = train_model(
        model, train_loader, val_loader, epochs=100, lr=1e-3, weight_decay=1e-4, patience=15)
    return model, (train_losses, val_losses, train_accs, val_accs)

# 6. 결과 시각화
def plot_curves(train_losses, val_losses, train_accs, val_accs, title=''):    
    fig, ax1 = plt.subplots()
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend(loc='upper left')
    ax2 = ax1.twinx()
    ax2.plot(train_accs, label='Train Acc', color='g', linestyle='--')
    ax2.plot(val_accs, label='Val Acc', color='r', linestyle='--')
    ax2.set_ylabel('Accuracy')
    ax2.legend(loc='upper right')
    plt.title(title)
    plt.show()

# 7. 테스트셋 평가
def evaluate(model, loader, device='cpu'):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x)
            pred = out.argmax(1).cpu().numpy()
            y_pred.extend(pred)
            y_true.extend(y.numpy())
    return np.array(y_true), np.array(y_pred)

def show_confusion(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
    print(classification_report(y_true, y_pred, target_names=class_names))

def show_examples(images, y_true, y_pred, class_names, correct=True, max_show=5):
    idxs = np.where((y_true == y_pred) if correct else (y_true != y_pred))[0]
    np.random.shuffle(idxs)
    idxs = idxs[:max_show]
    plt.figure(figsize=(10,2))
    for i, idx in enumerate(idxs):
        plt.subplot(1, max_show, i+1)
        plt.imshow(images[idx], cmap='gray')
        plt.title(f"P:{class_names[y_pred[idx]]}\nT:{class_names[y_true[idx]]}")
        plt.axis('off')
    plt.suptitle('Correct' if correct else 'Misclassified')
    plt.show()

# 8. 첫 conv 필터 시각화
def visualize_first_layer(model):
    w = model.conv[0].weight.data.cpu().numpy()  # (out_c, 1, 3, 3)
    n = w.shape[0]
    plt.figure(figsize=(n,1))
    for i in range(n):
        plt.subplot(1, n, i+1)
        plt.imshow(w[i,0], cmap='gray')
        plt.axis('off')
    plt.suptitle('First Conv Filters')
    plt.show()

# 9. (보너스) 전이학습

def transfer_learning():
    model = models.resnet18(weights='IMAGENET1K_V1')
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, 3)
    model = model.to('cpu')
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    best_model = copy.deepcopy(model.state_dict())
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    for epoch in range(31):
        model.train()
        total, correct, running_loss = 0, 0, 0.0
        for x, y in train_loader:
            x, y = x, y
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            if epoch > 0:
                loss.backward()
                optimizer.step()
            running_loss += loss.item() * x.size(0)
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += x.size(0)
        train_loss = running_loss / total
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        # Validation
        model.eval()
        total, correct, running_loss = 0, 0, 0.0
        with torch.no_grad():
            for x, y in val_loader:
                out = model(x)
                loss = criterion(out, y)
                running_loss += loss.item() * x.size(0)
                pred = out.argmax(1)
                correct += (pred == y).sum().item()
                total += x.size(0)
        val_loss = running_loss / total
        val_acc = correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        print(f"[ResNet] Epoch {epoch:3d}: train loss {train_loss:.4f}, val loss {val_loss:.4f}, train acc {train_acc:.3f}, val acc {val_acc:.3f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 10:
                print(f"Early stopping at epoch {epoch}")
                break
    model.load_state_dict(best_model)
    return model, (train_losses, val_losses, train_accs, val_accs)


if __name__ == '__main__':
    # 1. 아키텍처 실험
    print("\n[1] Architecture experiments...")
    arch_results = experiment_architectures()

    # 2. FC baseline
    print("\n[2] FC baseline training...")
    fc_model = ShelfFC()
    fc_model, (fc_train_losses, fc_val_losses, fc_train_accs, fc_val_accs) = train_model(
        fc_model, train_loader, val_loader, epochs=30, lr=1e-3, patience=5)

    # 3. 정규화 적용
    print("\n[3] Training with full regularization...")
    model, (train_losses, val_losses, train_accs, val_accs) = run_with_regularization()
    plot_curves(train_losses, val_losses, train_accs, val_accs, title='With Regularization')

    # 4. 테스트셋 평가 (best model 복원 후)
    y_true, y_pred = evaluate(model, test_loader)
    show_confusion(y_true, y_pred, class_names)
    show_examples(X_test, y_true, y_pred, class_names, correct=True)
    show_examples(X_test, y_true, y_pred, class_names, correct=False)

    # 5. 필터 시각화
    visualize_first_layer(model)

    # 6. (보너스) 전이학습
    print("\n[4] Transfer learning (ResNet18)...")
    tmodel, (tlosses, vlosses, taccs, vaccs) = transfer_learning()
    plot_curves(tlosses, vlosses, taccs, vaccs, title='Transfer Learning (ResNet18)')
    y_true_t, y_pred_t = evaluate(tmodel, test_loader)
    show_confusion(y_true_t, y_pred_t, class_names)

"""
train.py  （语法修正版）
零配置 LSTM 训练脚本
"""
import os, glob, json, tqdm, argparse, numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
import joblib


# ---------- 数据集 ----------
class NpyDataset(Dataset):
    def __init__(self, root):
        # 只扫 train 子目录
        self.files = sorted(glob.glob(os.path.join(root, '*', 'train', '*.npy')))

    def __getitem__(self, idx):
        path = self.files[idx]
        label = 1 if 'pos' in path else 0
        x = torch.from_numpy(np.load(path)).float()          # (F, 99)
        return x, label

    def __len__(self):
        return len(self.files)


# ---------- 模型 ----------
class SimpleLSTM(nn.Module):
    def __init__(self, c=99, hidden=128, layers=2, n_classes=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(c, hidden, layers, batch_first=True, dropout=dropout)
        self.fc   = nn.Linear(hidden, n_classes)

    def forward(self, x):
        out, _ = self.lstm(x)          # (B,T,H)
        out = self.fc(out[:, -1, :])   # (B,2)
        # ===== 调试：输出范围 =====
        print('[OUT] min=', out.min().item(), 'max=', out.max().item(), 'shape=', out.shape)
        return out


# ---------- 训练 ----------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, running_acc = 0., 0.
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        print('[LOSS]', loss.item())                       # 调试：损失值
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)
        running_acc  += (out.argmax(1) == y).sum().item()
    return running_loss / len(loader.dataset), running_acc / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss, running_acc = 0., 0.
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)
        running_loss += loss.item() * x.size(0)
        running_acc  += (out.argmax(1) == y).sum().item()
    return running_loss / len(loader.dataset), running_acc / len(loader.dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='norm', help='数据根目录（含 pos/neg 子目录）')
    parser.add_argument('--out',  default='result', help='输出目录')
    parser.add_argument('--epoch', type=int, default=30, help='总 epoch')
    parser.add_argument('--batch', type=int, default=32, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--hidden', type=int, default=128, help='LSTM 隐藏层')
    parser.add_argument('--layers', type=int, default=2, help='LSTM 层数')
    parser.add_argument('--drop', type=float, default=0.2, help='dropout')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 数据（直接扫子目录）
    train_files = sorted(glob.glob(os.path.join(args.data, '*', 'train', '*.npy')))
    val_files   = sorted(glob.glob(os.path.join(args.data, '*', 'val', '*.npy')))

    def load_set(file_list):
        xs, ys = [], []
        for f in tqdm.tqdm(file_list, desc='Load'):
            x = np.load(f)  # (F, 99)
            label = 1 if 'pos' in f else 0  # 文件夹名 = 标签
            xs.append(torch.from_numpy(x).float())
            ys.append(torch.tensor(label).long())
        return TensorDataset(torch.stack(xs), torch.stack(ys))

    train_set = load_set(train_files)
    val_set   = load_set(val_files)
    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_set,   batch_size=args.batch, shuffle=False, drop_last=False)

    # 2. 模型 & 早停 & 日志（零改动）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleLSTM(hidden=args.hidden, layers=args.layers, dropout=args.drop).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_acc, patience = 0, 10
    log = []
    for epoch in range(args.epoch):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss,   val_acc   = validate(model, val_loader, criterion, device)
        log.append({'epoch': epoch, 'train_loss': train_loss, 'train_acc': train_acc,
                    'val_loss': val_loss, 'val_acc': val_acc})
        print(f'Epoch {epoch:02d} | Train Loss {train_loss:.4f} Acc {train_acc:.4f} '
              f'| Val Loss {val_loss:.4f} Acc {val_acc:.4f}')

        # 早停
        if val_acc > best_acc:
            best_acc = val_acc
            patience = 10
            torch.save(model.state_dict(), os.path.join(args.out, 'best.pt'))
        else:
            patience -= 1
            if patience == 0:
                print('>>> 早停触发')
                break

    torch.save(model.state_dict(), os.path.join(args.out, 'final.pt'))
    with open(os.path.join(args.out, 'log.json'), 'w', encoding='utf-8') as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    print(f'✔ 训练完成，最佳验证 Acc = {best_acc:.4f}')


if __name__ == '__main__':
    main()
import sys
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import shutil

# =========================
# CONFIG
# =========================
LR=0.05
EPOCHS=50
NUM_WORKERS=2
NUM_CLASSES=60
BATCH_SIZE=16

MODEL_PATH = "/content/drive/MyDrive/Deep Learning Project/models/ntu60_hrnet.pkl"
output_dir = os.path.dirname(MODEL_PATH)
saved_model_path = os.path.join(output_dir, 'ctr_gcn_checkpoint.pth')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# =========================
# GRAPH CONFIGURATION
# =========================
def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    return np.dot(A, Dn)

class Graph():
    def __init__(self, num_node=17, max_hop=3):
        self.num_node = num_node
        self.max_hop = max_hop
        # COCO 17 joints inward edges
        self.inward = [(1, 0), (2, 0), (3, 1), (4, 2), (5, 0), (6, 0), (7, 5), (8, 6), (9, 7), (10, 8), (11, 5), (12, 6), (13, 11), (14, 12), (15, 13), (16, 14)]
        self.edge = self.inward + [(j, i) for (i, j) in self.inward]
        self.A = self.get_adjacency()

    def get_adjacency(self):
        adjacency = np.zeros((self.num_node, self.num_node))
        for i, j in self.edge:
            adjacency[j, i] = 1
            adjacency[i, j] = 1
            
        A_pow = []
        A_k = np.eye(self.num_node)
        adj_with_self = normalize_digraph(adjacency + np.eye(self.num_node))
        for k in range(self.max_hop):
            A_pow.append(A_k)
            A_k = np.matmul(A_k, adj_with_self)
        return np.stack(A_pow)

# =========================
# MSG-3D BLOCKS
# =========================
class MS_TCN(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilations=[1, 2, 3, 4]):
        super().__init__()
        assert out_channels % (len(dilations) + 2) == 0, "out_channels must be divisible by branches"
        branch_c = out_channels // (len(dilations) + 2)
        
        self.branches = nn.ModuleList()
        # 1x1 conv branch
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_c, 1, stride=(stride, 1)),
            nn.BatchNorm2d(branch_c)
        ))
        # max pool branch
        self.branches.append(nn.Sequential(
            nn.MaxPool2d((3, 1), stride=(stride, 1), padding=(1, 0)),
            nn.Conv2d(in_channels, branch_c, 1),
            nn.BatchNorm2d(branch_c)
        ))
        # dilation branches
        for d in dilations:
            self.branches.append(nn.Sequential(
                nn.Conv2d(in_channels, branch_c, 1),
                nn.BatchNorm2d(branch_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(branch_c, branch_c, (3, 1), stride=(stride, 1), padding=(d, 0), dilation=(d, 1)),
                nn.BatchNorm2d(branch_c)
            ))
            
    def forward(self, x):
        return torch.cat([branch(x) for branch in self.branches], dim=1)

class MS_GCN(nn.Module):
    def __init__(self, in_channels, out_channels, A_scales):
        super().__init__()
        self.num_scales = A_scales.size(0)
        self.conv = nn.Conv2d(in_channels * self.num_scales, out_channels, 1)
        self.register_buffer('A', A_scales)
        
    def forward(self, x):
        N, C, T, V = x.size()
        x_scales = []
        for s in range(self.num_scales):
            # x: (N, C, T, V), A: (V, V)
            x_s = torch.einsum('nctv,vw->nctw', x, self.A[s])
            x_scales.append(x_s)
        x_out = torch.cat(x_scales, dim=1)
        return self.conv(x_out)

class MSG3D_Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, A_scales=None):
        super().__init__()
        self.msgcn = MS_GCN(in_channels, out_channels, A_scales)
        self.mstcn = MS_TCN(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU(inplace=True)
        if in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        res = self.residual(x)
        x = self.relu(self.msgcn(x))
        x = self.mstcn(x)
        return self.relu(x + res)

class MSG3D(nn.Module):
    def __init__(self, in_channels, num_classes, num_node=17):
        super().__init__()
        self.graph = Graph(num_node, max_hop=3)
        A_scales = torch.tensor(self.graph.A, dtype=torch.float32)
        
        self.data_bn = nn.BatchNorm1d(in_channels * num_node)
        
        self.blocks = nn.ModuleList([
            MSG3D_Block(in_channels, 96, A_scales=A_scales),
            MSG3D_Block(96, 96, A_scales=A_scales),
            MSG3D_Block(96, 192, stride=2, A_scales=A_scales),
            MSG3D_Block(192, 192, A_scales=A_scales),
            MSG3D_Block(192, 384, stride=2, A_scales=A_scales),
            MSG3D_Block(384, 384, A_scales=A_scales)
        ])
        
        self.fc = nn.Linear(384, num_classes)
        
    def forward(self, x):
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous() # (N, M, V, C, T)
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous() # (N, M, C, T, V)
        x = x.view(N * M, C, T, V)
        
        for block in self.blocks:
            x = block(x)
            
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(N, M, -1).mean(dim=1)
        return self.fc(x)

class MultiStream_MSG3D(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.joint_model = MSG3D(in_channels, num_classes)
        self.bone_model = MSG3D(in_channels, num_classes)
        self.vel_model = MSG3D(in_channels, num_classes)
        
        self.parents = [0, 0, 0, 1, 2, 0, 0, 5, 6, 7, 8, 5, 6, 11, 12, 13, 14]
        
    def forward(self, x):
        # Bone
        b = torch.zeros_like(x)
        for v in range(x.size(3)):
            u = self.parents[v]
            b[:, :, :, v, :] = x[:, :, :, v, :] - x[:, :, :, u, :]
            
        # Velocity
        v = torch.zeros_like(x)
        v[:, :, :-1, :, :] = x[:, :, 1:, :, :] - x[:, :, :-1, :, :]
        
        out = self.joint_model(x) + self.bone_model(b) + self.vel_model(v)
        return out

# =========================
# DATASET
# =========================
class NTUDataset(Dataset):
    def __init__(self, data_path, split_name, max_frames=64, num_joints=17, max_persons=2, is_training=True):
        print(f"Loading data from {data_path} for split {split_name}...")
        
        if not os.path.exists(data_path) and os.path.exists('models/ntu60_hrnet.pkl'):
            data_path = 'models/ntu60_hrnet.pkl'
            
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
            
        self.split_ids = set(self.data['split'][split_name])
        self.samples = []
        for ann in tqdm(self.data['annotations'], desc=f"Filtering {split_name}"):
            if ann['frame_dir'] in self.split_ids:
                self.samples.append(ann)
                
        self.max_frames = max_frames
        self.max_persons = max_persons
        self.is_training = is_training
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        ann = self.samples[idx]
        kp = ann['keypoint'].copy()
        score = ann.get('keypoint_score', None)
        if score is not None:
            score = score.copy()
            
        M, T, V, C = kp.shape
        
        # Temporal Crop/Pad
        if T >= self.max_frames:
            if self.is_training:
                start = np.random.randint(0, T - self.max_frames + 1)
            else:
                start = (T - self.max_frames) // 2
            kp = kp[:, start:start+self.max_frames, :, :]
            if score is not None:
                score = score[:, start:start+self.max_frames, :]
        else:
            pad_len = self.max_frames - T
            pad_kp = np.zeros((M, pad_len, V, C), dtype=kp.dtype)
            kp = np.concatenate([kp, pad_kp], axis=1)
            if score is not None:
                pad_score = np.zeros((M, pad_len, V), dtype=score.dtype)
                score = np.concatenate([score, pad_score], axis=1)
                
        # Person Padding
        if M < self.max_persons:
            pad_m = self.max_persons - M
            pad_kp = np.zeros((pad_m, self.max_frames, V, C), dtype=kp.dtype)
            kp = np.concatenate([kp, pad_kp], axis=0)
            if score is not None:
                pad_score = np.zeros((pad_m, self.max_frames, V), dtype=score.dtype)
                score = np.concatenate([score, pad_score], axis=0)
        elif M > self.max_persons:
            kp = kp[:self.max_persons]
            if score is not None:
                score = score[:self.max_persons]
                
        # Normalization
        valid_mask = (kp.sum(axis=-1) != 0)
        if valid_mask.any():
            m_idx, t_idx, _ = np.where(valid_mask)
            center = kp[m_idx[0], t_idx[0], 0, :].copy()
            mask = np.expand_dims(valid_mask, axis=-1)
            kp = kp - center * mask
            
        # Augmentations
        if self.is_training:
            scale = np.random.uniform(0.8, 1.2)
            kp[:, :, :, :2] = kp[:, :, :, :2] * scale
            
            num_mask = int(self.max_frames * 0.1)
            mask_indices = np.random.choice(self.max_frames, num_mask, replace=False)
            kp[:, mask_indices, :, :] = 0
            if score is not None:
                score[:, mask_indices, :] = 0
                
        if score is not None:
            score = np.expand_dims(score, axis=-1)
            kp = np.concatenate([kp, score], axis=-1)
            
        kp = kp.transpose((3, 1, 2, 0)) # (C, T, V, M)
        return torch.tensor(kp, dtype=torch.float32), torch.tensor(ann['label'], dtype=torch.long)

# =========================
# UTILITIES
# =========================
def plot_metrics(history, output_dir):
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss History')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('Accuracy History')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'msg3d_curves.png'))
    plt.close()

def evaluate_model(model, dataloader, output_dir):
    model.eval()
    y_true, y_pred = [], []
    print("\n--- Evaluating Test Set ---")
    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Testing"):
            out = model(x.to(DEVICE))
            pred = out.argmax(dim=1).cpu().numpy()
            y_pred.extend(pred)
            y_true.extend(y.numpy())
            
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(20, 16))
    sns.heatmap(cm, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(output_dir, 'msg3d_confusion.png'))
    plt.close()
    print("Metrics saved successfully.")

# =========================
# MAIN ROUTINE
# =========================
def main():
    os.makedirs(output_dir, exist_ok=True)
    best_model_path = saved_model_path.replace('.pth', '_best.pth')
    
    train_data = NTUDataset(MODEL_PATH, 'xview_train', is_training=True)
    val_data = NTUDataset(MODEL_PATH, 'xview_val', is_training=False)
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    
    in_channels = train_data[0][0].size(0)
    model = MultiStream_MSG3D(in_channels, NUM_CLASSES).to(DEVICE)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 40], gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    
    start_epoch = 0
    best_val_acc = 0.0
    history = {'train_loss':[], 'train_acc':[], 'val_loss':[], 'val_acc':[]}
    best_model_state = None
    
    if os.path.exists(saved_model_path):
        print(f"Loading checkpoint {saved_model_path}")
        ckpt = torch.load(saved_model_path, map_location=DEVICE)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['opt_state_dict'])
        scheduler.load_state_dict(ckpt['sch_state_dict'])
        start_epoch = ckpt['epoch']
        best_val_acc = ckpt['best_val_acc']
        history = ckpt['history']
        best_model_state = ckpt.get('best_model_state_dict', None)
        print(f"Resumed at epoch {start_epoch} with Best Acc: {best_val_acc:.2f}%")
        
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        t_loss, correct, total = 0, 0, 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            
            t_loss += loss.item()
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
            
        train_loss = t_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        model.eval()
        v_loss, v_corr, v_tot = 0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = model(x)
                loss = criterion(out, y)
                v_loss += loss.item()
                v_corr += (out.argmax(1) == y).sum().item()
                v_tot += y.size(0)
                
        val_loss = v_loss / len(val_loader)
        val_acc = 100. * v_corr / v_tot
        
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Acc: {val_acc:.2f}%")
        
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'best_model_state_dict': best_model_state,
            'opt_state_dict': optimizer.state_dict(),
            'sch_state_dict': scheduler.state_dict(),
            'best_val_acc': best_val_acc,
            'history': history
        }, saved_model_path)
        
        if is_best:
            torch.save({'model_state_dict': best_model_state}, best_model_path)
            
        plot_metrics(history, output_dir)
        
    print("Training Complete. Evaluating Best Model...")
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    evaluate_model(model, val_loader, output_dir)

if __name__ == '__main__':
    main()
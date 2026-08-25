print("Training EEGNet on all 109 PhysioNet subjects")
print("This will take 20-30 minutes — let it run!")
print("=" * 50)

import mne
from mne.datasets import eegbci
from mne import Epochs
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
mne.set_log_level('WARNING')

# ── EEGNet definition ─────────────────────────────────────────
class EEGNet4(nn.Module):
    def __init__(self, n_channels, n_timepoints,
                 n_classes=4, F1=8, D=2, F2=16, dropout=0.5):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, 64),
                      padding=(0, 32), bias=False),
            nn.BatchNorm2d(F1))
        self.block2 = nn.Sequential(
            nn.Conv2d(F1, F1*D, kernel_size=(n_channels, 1),
                      groups=F1, bias=False),
            nn.BatchNorm2d(F1*D), nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout))
        self.block3 = nn.Sequential(
            nn.Conv2d(F2, F2, kernel_size=(1, 16),
                      padding=(0, 8), bias=False),
            nn.Conv2d(F2, F2, kernel_size=1, bias=False),
            nn.BatchNorm2d(F2), nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout))
        flat = self._flat(n_channels, n_timepoints, F1, D, F2)
        self.classifier = nn.Sequential(
            nn.Linear(flat, 64), nn.ELU(),
            nn.Dropout(0.3), nn.Linear(64, n_classes))

    def _flat(self, nc, nt, F1, D, F2):
        with torch.no_grad():
            x = torch.zeros(1, 1, nc, nt)
            x = self.block1(x); x = self.block2(x)
            x = self.block3(x)
            return x.view(1, -1).shape[1]

    def forward(self, x):
        x = self.block1(x); x = self.block2(x)
        x = self.block3(x)
        return self.classifier(x.view(x.size(0), -1))

# ── Load all 109 subjects ─────────────────────────────────────
print("\n[1/4] Loading data from all 109 subjects...")
print("      (Downloads happen automatically — be patient)")

CLASS_NAMES = ['Left hand','Right hand','Both hands','Feet']
MENU_NAMES  = ['Basic needs','Emotions','Actions','People']

X_list, y_list = [], []
success = 0
failed  = 0

for subject in range(1, 110):
    try:
        # Run set A: left vs right hand
        fnames_a = eegbci.load_data(subject, [6, 10, 14])
        raw_a = mne.io.concatenate_raws([
            mne.io.read_raw_edf(f, preload=True)
            for f in fnames_a])
        eegbci.standardize(raw_a)
        raw_a.set_montage(
            mne.channels.make_standard_montage('standard_1005'))
        raw_a.filter(l_freq=1.0, h_freq=40.0)
        events_a, _ = mne.events_from_annotations(raw_a)
        ep_a = Epochs(raw_a, events_a,
                      event_id={'left hand':2,'right hand':3},
                      tmin=0.0, tmax=2.0, proj=True,
                      picks='eeg', baseline=None, preload=True)

        # Run set B: both hands vs feet
        fnames_b = eegbci.load_data(subject, [8, 12])
        raw_b = mne.io.concatenate_raws([
            mne.io.read_raw_edf(f, preload=True)
            for f in fnames_b])
        eegbci.standardize(raw_b)
        raw_b.set_montage(
            mne.channels.make_standard_montage('standard_1005'))
        raw_b.filter(l_freq=1.0, h_freq=40.0)
        events_b, _ = mne.events_from_annotations(raw_b)
        ep_b = Epochs(raw_b, events_b,
                      event_id={'both hands':2,'feet':3},
                      tmin=0.0, tmax=2.0, proj=True,
                      picks='eeg', baseline=None, preload=True)

        # Extract and label
        data_a = ep_a.get_data().astype(np.float32)
        labs_a = ep_a.events[:, 2]
        mapped_a = np.where(labs_a == 2, 0, 1)

        data_b = ep_b.get_data().astype(np.float32)
        labs_b = ep_b.events[:, 2]
        mapped_b = np.where(labs_b == 2, 2, 3)

        X_list.append(data_a)
        X_list.append(data_b)
        y_list.append(mapped_a)
        y_list.append(mapped_b)

        success += 1
        if subject % 10 == 0:
            print(f"    Loaded {subject}/109 subjects "
                  f"({success} ok, {failed} skipped)")

    except Exception as e:
        failed += 1
        continue

print(f"\n    Done! {success} subjects loaded, {failed} skipped")

X = np.concatenate(X_list, axis=0)
y = np.concatenate(y_list, axis=0)

print(f"    Total trials : {len(X)}")
for i, name in enumerate(CLASS_NAMES):
    print(f"    {name:12s}: {(y==i).sum()} trials")

# ── Normalize + prepare ───────────────────────────────────────
print("\n[2/4] Preparing tensors...")
X_mean = X.mean(axis=2, keepdims=True)
X_std  = X.std(axis=2,  keepdims=True) + 1e-6
X      = (X - X_mean) / X_std
X      = X[:, np.newaxis, :, :]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42,
    stratify=y_train)

print(f"    Train:{len(X_train)} Val:{len(X_val)} Test:{len(X_test)}")

def to_loader(X, y, batch=64, shuffle=True):
    ds = TensorDataset(
        torch.tensor(X),
        torch.tensor(y.astype(np.int64)))
    return DataLoader(ds, batch_size=batch, shuffle=shuffle)

train_loader = to_loader(X_train, y_train)
val_loader   = to_loader(X_val,   y_val,   shuffle=False)
test_loader  = to_loader(X_test,  y_test,  shuffle=False)

# ── Train ─────────────────────────────────────────────────────
print("\n[3/4] Training EEGNet on 109 subjects...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"    Device: {device}")

n_ch = X.shape[2]
n_tp = X.shape[3]

model     = EEGNet4(n_ch, n_tp, n_classes=4).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001,
                       weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=5, factor=0.5)

EPOCHS     = 60
best_val   = 0.0
best_state = None
patience_c = 0
PATIENCE   = 12

print(f"    Epoch | Train Acc |  Val Acc")
print(f"    " + "-"*30)

for epoch in range(1, EPOCHS+1):
    model.train()
    t_correct = t_total = 0
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out  = model(Xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        t_correct += (out.argmax(1)==yb).sum().item()
        t_total   += len(yb)

    model.eval()
    v_correct = v_total = v_loss_sum = 0
    with torch.no_grad():
        for Xb, yb in val_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            out     = model(Xb)
            v_loss_sum += criterion(out, yb).item()*len(yb)
            v_correct  += (out.argmax(1)==yb).sum().item()
            v_total    += len(yb)

    t_acc  = t_correct / t_total
    v_acc  = v_correct / v_total
    v_loss = v_loss_sum / v_total
    scheduler.step(v_loss)

    if v_acc > best_val:
        best_val   = v_acc
        best_state = {k:v.clone()
                      for k,v in model.state_dict().items()}
        patience_c = 0
        tag = ' ← best'
    else:
        patience_c += 1
        tag = ''

    if epoch % 5 == 0 or epoch == 1:
        print(f"    {epoch:5d} | {t_acc:9.2%} | "
              f"{v_acc:8.2%}{tag}")

    if patience_c >= PATIENCE:
        print(f"\n    Early stopping at epoch {epoch}")
        break

print(f"\n    Best validation accuracy: {best_val:.2%}")

# ── Test + save ───────────────────────────────────────────────
print("\n[4/4] Testing and saving...")
model.load_state_dict(best_state)
model.eval()

all_preds, all_true = [], []
with torch.no_grad():
    for Xb, yb in test_loader:
        Xb = Xb.to(device)
        all_preds.extend(model(Xb).argmax(1).cpu().numpy())
        all_true.extend(yb.numpy())

test_acc = (np.array(all_preds)==np.array(all_true)).mean()
print(f"    Final test accuracy : {test_acc:.2%}")
print(f"    Previous (10 subj) : 52.67%")
print(f"    Improvement        : +{(test_acc-0.5267)*100:.1f}%")

# Save
save_path = r'C:\Users\SHISHIR\Desktop\BCI Project\eegnet_4class.pth'
torch.save({
    'model_state':  best_state,
    'n_channels':   n_ch,
    'n_timepoints': n_tp,
    'n_classes':    4,
    'class_names':  CLASS_NAMES,
    'menu_names':   MENU_NAMES,
    'accuracy':     test_acc,
    'n_subjects':   success,
}, save_path)

print(f"\n    Saved: eegnet_4class.pth")
print(f"    Trained on {success} subjects")
print("\n" + "="*50)
print(f"  TRAINING COMPLETE!")
print(f"  New accuracy : {test_acc:.2%}")
print(f"  Subjects     : {success}")
print("="*50)
input("\nPress Enter to close...")
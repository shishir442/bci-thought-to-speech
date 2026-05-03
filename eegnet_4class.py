print("Hybrid BCI — Stage 1: 4-Class EEGNet")
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
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
mne.set_log_level('WARNING')

# ══════════════════════════════════════════════════════════════
# STEP 1 — Load 4-class motor imagery data
# ══════════════════════════════════════════════════════════════
# PhysioNet run numbers:
#   Runs 6,10,14  → left fist vs right fist  (classes 2,3)
#   Runs 8,12,16  → both fists vs both feet  (classes 2,3)
# We combine them to get 4 classes:
#   left hand, right hand, both hands, feet
print("\n[1/6] Loading 4-class EEG data from 10 subjects...")

all_epochs = []

for subject in range(1, 11):
    try:
        # Run set A: left hand (2) vs right hand (3)
        fnames_a = eegbci.load_data(subject, [6, 10, 14])
        raw_a = mne.io.concatenate_raws([
            mne.io.read_raw_edf(f, preload=True) for f in fnames_a
        ])
        eegbci.standardize(raw_a)
        raw_a.set_montage(mne.channels.make_standard_montage('standard_1005'))
        raw_a.filter(l_freq=1.0, h_freq=40.0)
        events_a, _ = mne.events_from_annotations(raw_a)
        ep_a = Epochs(raw_a, events_a,
                      event_id={'left hand': 2, 'right hand': 3},
                      tmin=0.0, tmax=2.0, proj=True,
                      picks='eeg', baseline=None, preload=True)

        # Run set B: both hands (2) vs feet (3)
        fnames_b = eegbci.load_data(subject, [8, 12])
        raw_b = mne.io.concatenate_raws([
            mne.io.read_raw_edf(f, preload=True) for f in fnames_b
        ])
        eegbci.standardize(raw_b)
        raw_b.set_montage(mne.channels.make_standard_montage('standard_1005'))
        raw_b.filter(l_freq=1.0, h_freq=40.0)
        events_b, _ = mne.events_from_annotations(raw_b)
        ep_b = Epochs(raw_b, events_b,
                      event_id={'both hands': 2, 'feet': 3},
                      tmin=0.0, tmax=2.0, proj=True,
                      picks='eeg', baseline=None, preload=True)

        all_epochs.append(ep_a)
        all_epochs.append(ep_b)
        print(f"    Subject {subject:2d} — "
              f"{len(ep_a)} hand trials + {len(ep_b)} feet trials")

    except Exception as e:
        print(f"    Subject {subject:2d} — skipped ({e})")

# ══════════════════════════════════════════════════════════════
# STEP 2 — Combine and relabel to 4 classes
# ══════════════════════════════════════════════════════════════
print("\n[2/6] Combining and relabelling to 4 classes...")

# Relabel everything consistently
# We'll build X and y manually from each epoch set
X_list, y_list = [], []

for i, ep in enumerate(all_epochs):
    data = ep.get_data().astype(np.float32)
    labels = ep.events[:, 2]

    # Even index = hand runs (left=2, right=3)
    # Odd index  = feet runs (both hands=2→4, feet=3→5... remap below)
    if i % 2 == 0:
        # left hand → 0, right hand → 1
        mapped = np.where(labels == 2, 0, 1)
    else:
        # both hands → 2, feet → 3
        mapped = np.where(labels == 2, 2, 3)

    X_list.append(data)
    y_list.append(mapped)

X = np.concatenate(X_list, axis=0)
y = np.concatenate(y_list, axis=0)

# Class names for display
CLASS_NAMES = ['Left hand', 'Right hand', 'Both hands', 'Feet']
MENU_NAMES  = ['Basic needs', 'Emotions', 'Actions', 'People']

print(f"    Total trials  : {len(X)}")
for i, name in enumerate(CLASS_NAMES):
    count = (y == i).sum()
    print(f"    Class {i} — {name:12s} ({MENU_NAMES[i]:12s}): {count} trials")

# ══════════════════════════════════════════════════════════════
# STEP 3 — Prepare tensors
# ══════════════════════════════════════════════════════════════
print("\n[3/6] Preparing tensors...")

# Normalize
X_mean = X.mean(axis=2, keepdims=True)
X_std  = X.std(axis=2,  keepdims=True) + 1e-6
X      = (X - X_mean) / X_std

# Add conv dimension
X = X[:, np.newaxis, :, :]

print(f"    Input shape : {X.shape}")
print(f"    Classes     : {len(np.unique(y))}")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
)
print(f"    Train:{len(X_train)} Val:{len(X_val)} Test:{len(X_test)}")

def to_loader(X, y, batch=32, shuffle=True):
    ds = TensorDataset(torch.tensor(X), torch.tensor(y.astype(np.int64)))
    return DataLoader(ds, batch_size=batch, shuffle=shuffle)

train_loader = to_loader(X_train, y_train)
val_loader   = to_loader(X_val,   y_val,   shuffle=False)
test_loader  = to_loader(X_test,  y_test,  shuffle=False)

# ══════════════════════════════════════════════════════════════
# STEP 4 — EEGNet (4-class version)
# ══════════════════════════════════════════════════════════════
print("\n[4/6] Building 4-class EEGNet...")

class EEGNet4(nn.Module):
    def __init__(self, n_channels, n_timepoints,
                 n_classes=4, F1=8, D=2, F2=16, dropout=0.5):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, 64),
                      padding=(0, 32), bias=False),
            nn.BatchNorm2d(F1)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(F1, F1*D, kernel_size=(n_channels, 1),
                      groups=F1, bias=False),
            nn.BatchNorm2d(F1*D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(F2, F2, kernel_size=(1, 16),
                      padding=(0, 8), bias=False),
            nn.Conv2d(F2, F2, kernel_size=1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout)
        )
        flat = self._flat(n_channels, n_timepoints, F1, D, F2)
        self.classifier = nn.Sequential(
            nn.Linear(flat, 64),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes)
        )

    def _flat(self, nc, nt, F1, D, F2):
        with torch.no_grad():
            x = torch.zeros(1, 1, nc, nt)
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
            return x.view(1, -1).shape[1]

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.classifier(x.view(x.size(0), -1))

device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_channels = X.shape[2]
n_tp       = X.shape[3]

model = EEGNet4(n_channels, n_tp, n_classes=4).to(device)
total = sum(p.numel() for p in model.parameters())
print(f"    Device     : {device}")
print(f"    Parameters : {total:,}")

# ══════════════════════════════════════════════════════════════
# STEP 5 — Train with early stopping
# ══════════════════════════════════════════════════════════════
print("\n[5/6] Training...")
print("    Epoch | Train Acc |  Val Acc | Status")
print("    " + "-" * 42)

criterion  = nn.CrossEntropyLoss()
optimizer  = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler  = optim.lr_scheduler.ReduceLROnPlateau(
                 optimizer, patience=8, factor=0.5)

EPOCHS         = 80
best_val       = 0.0
best_state     = None
patience_count = 0
PATIENCE       = 15

train_accs, val_accs = [], []

for epoch in range(1, EPOCHS + 1):

    # Train
    model.train()
    t_correct = t_total = 0
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out  = model(Xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        t_correct += (out.argmax(1) == yb).sum().item()
        t_total   += len(yb)

    # Validate
    model.eval()
    v_correct = v_total = v_loss_sum = 0
    with torch.no_grad():
        for Xb, yb in val_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            out    = model(Xb)
            v_loss_sum += criterion(out, yb).item() * len(yb)
            v_correct  += (out.argmax(1) == yb).sum().item()
            v_total    += len(yb)

    t_acc = t_correct / t_total
    v_acc = v_correct / v_total
    v_loss = v_loss_sum / v_total
    train_accs.append(t_acc)
    val_accs.append(v_acc)
    scheduler.step(v_loss)

    # Early stopping
    if v_acc > best_val:
        best_val   = v_acc
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        patience_count = 0
        status = "← best"
    else:
        patience_count += 1
        status = f"patience {patience_count}/{PATIENCE}"

    if epoch % 5 == 0 or epoch == 1:
        print(f"    {epoch:5d} | {t_acc:9.2%} | {v_acc:8.2%} | {status}")

    if patience_count >= PATIENCE:
        print(f"\n    Early stopping at epoch {epoch}")
        break

print(f"\n    Best val accuracy: {best_val:.2%}")

# ══════════════════════════════════════════════════════════════
# STEP 6 — Evaluate + save
# ══════════════════════════════════════════════════════════════
print("\n[6/6] Evaluating and saving...")

model.load_state_dict(best_state)
model.eval()

all_preds, all_true, all_probs = [], [], []
with torch.no_grad():
    for Xb, yb in test_loader:
        Xb = Xb.to(device)
        probs = F.softmax(model(Xb), dim=1).cpu().numpy()
        all_probs.extend(probs)
        all_preds.extend(probs.argmax(axis=1))
        all_true.extend(yb.numpy())

all_preds = np.array(all_preds)
all_true  = np.array(all_true)
all_probs = np.array(all_probs)
test_acc  = (all_preds == all_true).mean()

print(f"\n    Test accuracy : {test_acc:.2%}")
print(f"    Chance level  : 25.0% (4 classes)")
print(f"    Above chance  : +{(test_acc-0.25)*100:.1f}%")

print("\n    Per-class results:")
print(classification_report(all_true, all_preds,
      target_names=CLASS_NAMES))

# ── Plots ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('4-Class EEGNet — Hybrid BCI Navigation Model',
             fontsize=13, fontweight='bold')

# Plot 1 — Training curves
axes[0].plot(train_accs, color='#534AB7', lw=1.5, label='Train')
axes[0].plot(val_accs,   color='#1D9E75', lw=1.5, label='Val')
axes[0].axhline(0.25, color='#888780', linestyle='--',
                lw=1, label='Chance 25%')
axes[0].set_title('Training accuracy'); axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy'); axes[0].legend()

# Plot 2 — Confusion matrix
cm = confusion_matrix(all_true, all_preds)
im = axes[1].imshow(cm, cmap='Blues', interpolation='nearest')
axes[1].set_title(f'Confusion matrix\nTest acc: {test_acc:.2%}')
axes[1].set_xticks(range(4))
axes[1].set_xticklabels(CLASS_NAMES, rotation=20, ha='right', fontsize=8)
axes[1].set_yticks(range(4))
axes[1].set_yticklabels(CLASS_NAMES, fontsize=8)
axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('True')
plt.colorbar(im, ax=axes[1])
for i in range(4):
    for j in range(4):
        axes[1].text(j, i, str(cm[i,j]), ha='center', va='center',
                     fontsize=10, fontweight='bold',
                     color='white' if cm[i,j] > cm.max()/2 else 'black')

# Plot 3 — Per-class accuracy bar
per_class_acc = cm.diagonal() / cm.sum(axis=1)
colors = ['#534AB7', '#1D9E75', '#BA7517', '#D85A30']
bars = axes[2].bar(CLASS_NAMES, per_class_acc,
                   color=colors, edgecolor='none', width=0.5)
axes[2].set_ylim(0, 1.15); axes[2].set_ylabel('Accuracy')
axes[2].set_title('Per-class accuracy')
axes[2].axhline(0.25, color='#888780', linestyle='--',
                lw=1, label='Chance')
for bar, acc in zip(bars, per_class_acc):
    axes[2].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.03,
                 f'{acc:.1%}', ha='center', fontsize=9, fontweight='bold')
axes[2].legend(); axes[2].tick_params(axis='x', labelsize=8)

plt.tight_layout()
plt.show()

# ── Save model ──────────────────────────────────────────────
torch.save({
    'model_state':  best_state,
    'n_channels':   n_channels,
    'n_timepoints': n_tp,
    'n_classes':    4,
    'class_names':  CLASS_NAMES,
    'menu_names':   MENU_NAMES,
    'accuracy':     test_acc
}, 'eegnet_4class.pth')

print("\n    Saved: eegnet_4class.pth")
print("\n" + "=" * 50)
print(f"  Stage 1 COMPLETE!")
print(f"  4-class accuracy : {test_acc:.2%}")
print(f"  Chance level     : 25.00%")
print(f"  Each class opens a different word menu:")
for i, (cls, menu) in enumerate(zip(CLASS_NAMES, MENU_NAMES)):
    print(f"    {cls:12s} → {menu}")
print("=" * 50)

input("\nPress Enter to close...")
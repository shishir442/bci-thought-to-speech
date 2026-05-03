print("Fast P300 — Smarter ML Model")
print("=" * 50)

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
import joblib
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════
# STEP 1 — Generate richer P300 training data
# ══════════════════════════════════════════════════════════════
print("\n[1/4] Generating rich P300 training data...")

np.random.seed(42)

SFREQ      = 256
EPOCH_LEN  = 0.6
N_SAMPLES  = int(SFREQ * EPOCH_LEN)
N_CHANNELS = 8
CH_NAMES   = ['Fz','Cz','Pz','Oz','P3','P4','PO7','PO8']

def generate_p300_epoch(is_target, noise_level=1.0):
    """
    Generate realistic P300 epoch with variable noise.
    Higher noise = harder to classify = more realistic.
    """
    t     = np.linspace(0, EPOCH_LEN, N_SAMPLES)
    epoch = np.zeros((N_CHANNELS, N_SAMPLES))
    for ch in range(N_CHANNELS):
        # Background EEG
        alpha = 3e-6 * np.sin(
            2*np.pi*10*t + np.random.rand()*2*np.pi)
        theta = 2e-6 * np.sin(
            2*np.pi*6*t  + np.random.rand()*2*np.pi)
        noise = noise_level * 2e-6 * np.random.randn(N_SAMPLES)
        signal = alpha + theta + noise

        if is_target:
            # P300 component
            p300_lat = int((0.28 + np.random.randn()*0.02) * SFREQ)
            p300_w   = int(0.08 * SFREQ)
            ch_w     = [0.6,0.9,1.0,0.7,0.8,0.8,0.5,0.5]
            amp      = (3.5 + np.random.randn()*0.5) * 1e-6 * ch_w[ch]
            p300     = amp * np.exp(
                -0.5*((np.arange(N_SAMPLES)-p300_lat)/p300_w)**2)

            # N200 component
            n200_lat = int(0.20 * SFREQ)
            n200     = -1.5e-6 * np.exp(
                -0.5*((np.arange(N_SAMPLES)-n200_lat)/(p300_w*0.7))**2)

            # N100 component
            n100_lat = int(0.10 * SFREQ)
            n100     = -1e-6 * np.exp(
                -0.5*((np.arange(N_SAMPLES)-n100_lat)/(p300_w*0.5))**2)

            signal = signal + p300 + n200 + n100

        epoch[ch] = signal
    return epoch

# Generate large dataset with variable noise levels
N_TRIALS   = 1000
X_list, y_list = [], []

for i in range(N_TRIALS):
    noise = 0.8 + np.random.rand() * 0.8   # noise between 0.8 and 1.6
    is_t  = (i % 5 == 0)                   # 20% targets
    X_list.append(generate_p300_epoch(is_t, noise))
    y_list.append(1 if is_t else 0)

X_all = np.array(X_list)
y_all = np.array(y_list)

print(f"    Generated {N_TRIALS} epochs")
print(f"    Targets     : {y_all.sum()}")
print(f"    Non-targets : {(y_all==0).sum()}")

# ══════════════════════════════════════════════════════════════
# STEP 2 — Rich feature extraction
# ══════════════════════════════════════════════════════════════
print("\n[2/4] Extracting rich features...")

def extract_rich_features(epochs):
    """
    Extract comprehensive P300 features:
    - Mean amplitude in multiple time windows
    - Peak amplitude and latency
    - Area under curve
    - Slope features
    - Channel ratios
    """
    features = []
    windows = [
        (0.00, 0.10),   # 0-100ms   baseline
        (0.10, 0.20),   # 100-200ms N100/N200
        (0.20, 0.35),   # 200-350ms N200/P300 onset
        (0.30, 0.50),   # 300-500ms P300 peak
        (0.45, 0.60),   # 450-600ms P300 tail
    ]

    for epoch in epochs:
        feat = []
        for ch in range(epoch.shape[0]):
            sig = epoch[ch]

            # Window features
            for w_start, w_end in windows:
                s = int(w_start * SFREQ)
                e = int(w_end   * SFREQ)
                window = sig[s:e]
                feat.append(window.mean())      # mean amplitude
                feat.append(window.max())       # peak
                feat.append(window.min())       # trough
                feat.append(np.abs(window).mean())  # absolute mean

            # P300 - baseline difference (key discriminating feature)
            p300_win  = sig[int(0.30*SFREQ):int(0.50*SFREQ)].mean()
            base_win  = sig[int(0.00*SFREQ):int(0.15*SFREQ)].mean()
            feat.append(p300_win - base_win)

            # Peak latency in P300 window
            p300_seg  = sig[int(0.20*SFREQ):int(0.55*SFREQ)]
            peak_lat  = np.argmax(p300_seg) / SFREQ
            feat.append(peak_lat)

            # Area under curve (P300 window)
            feat.append(np.trapezoid(
                sig[int(0.25*SFREQ):int(0.55*SFREQ)]))

        features.append(feat)
    return np.array(features)

X_feat = extract_rich_features(X_all)
print(f"    Feature shape : {X_feat.shape}")
print(f"    Features/epoch: {X_feat.shape[1]}")

# ══════════════════════════════════════════════════════════════
# STEP 3 — Train and compare 3 models
# ══════════════════════════════════════════════════════════════
print("\n[3/4] Training and comparing models...")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Model 1 — Standard LDA (current)
lda = Pipeline([
    ('sc',  StandardScaler()),
    ('lda', LinearDiscriminantAnalysis())
])
lda_auc = cross_val_score(
    lda, X_feat, y_all, cv=cv, scoring='roc_auc').mean()

# Model 2 — Bayesian LDA with shrinkage (smarter)
blda = Pipeline([
    ('sc',   StandardScaler()),
    ('blda', LinearDiscriminantAnalysis(
        solver='eigen', shrinkage='auto'))
])
blda_auc = cross_val_score(
    blda, X_feat, y_all, cv=cv, scoring='roc_auc').mean()

# Model 3 — Gradient Boosting (most powerful)
gbc = Pipeline([
    ('sc',  StandardScaler()),
    ('gbc', GradientBoostingClassifier(
        n_estimators=100, max_depth=3,
        learning_rate=0.1, random_state=42))
])
gbc_auc = cross_val_score(
    gbc, X_feat, y_all, cv=cv, scoring='roc_auc').mean()

print(f"\n    Standard LDA  AUC : {lda_auc:.3f}")
print(f"    Bayesian LDA  AUC : {blda_auc:.3f}")
print(f"    Gradient Boost AUC: {gbc_auc:.3f}")

# Pick best model
best_name, best_auc, best_model = max(
    [('LDA',           lda_auc,  lda),
     ('Bayesian LDA',  blda_auc, blda),
     ('Gradient Boost',gbc_auc,  gbc)],
    key=lambda x: x[1]
)
print(f"\n    Best model: {best_name} (AUC={best_auc:.3f})")

# Train best model on all data
best_model.fit(X_feat, y_all)

# ══════════════════════════════════════════════════════════════
# STEP 4 — Early stopping simulation
# ══════════════════════════════════════════════════════════════
print("\n[4/4] Testing early stopping (how few flashes needed)...")

def simulate_word_selection(target_idx, n_words=8,
                             max_rounds=8, threshold=0.75):
    """
    Simulate selecting a word with early stopping.
    Stops as soon as one word's confidence exceeds threshold.
    Returns: selected word index, rounds needed, correct or not
    """
    word_scores = np.zeros(n_words)

    for round_num in range(1, max_rounds + 1):
        order = list(range(n_words))
        np.random.shuffle(order)

        for word_idx in order:
            is_target = (word_idx == target_idx)
            epoch     = generate_p300_epoch(is_target)
            feat      = extract_rich_features(
                epoch[np.newaxis])[0].reshape(1, -1)
            score     = best_model.predict_proba(feat)[0][1]
            word_scores[word_idx] += score

        # Check early stopping
        avg_scores  = word_scores / round_num
        best_idx    = np.argmax(avg_scores)
        best_conf   = avg_scores[best_idx]

        if best_conf >= threshold:
            return best_idx, round_num, best_idx == target_idx

    return np.argmax(word_scores/max_rounds), max_rounds, \
           np.argmax(word_scores/max_rounds) == target_idx

# Run 100 simulations
print("\n    Running 100 selection simulations...")
results = []
for _ in range(100):
    target  = np.random.randint(0, 8)
    sel, rounds, correct = simulate_word_selection(target)
    results.append((rounds, correct))

rounds_arr  = np.array([r[0] for r in results])
correct_arr = np.array([r[1] for r in results])

print(f"\n    Results with early stopping (threshold=0.75):")
print(f"    Accuracy          : {correct_arr.mean():.1%}")
print(f"    Avg rounds needed : {rounds_arr.mean():.1f} / 8")
print(f"    Min rounds        : {rounds_arr.min()}")
print(f"    Max rounds        : {rounds_arr.max()}")

flash_on  = 80
flash_off = 50
avg_time  = rounds_arr.mean() * 8 * (flash_on + flash_off) / 1000
print(f"    Avg time per word : {avg_time:.1f} seconds")
print(f"    (down from ~15 seconds with old approach)")

# Save fast model
joblib.dump({
    'pipeline':       best_model,
    'model_name':     best_name,
    'ch_names':       CH_NAMES,
    'sfreq':          SFREQ,
    'n_samples':      N_SAMPLES,
    'n_channels':     N_CHANNELS,
    'auc':            best_auc,
    'feature_func':   'extract_rich_features'
}, r'C:\Users\SHISHIR\Desktop\BCI Project\p300_fast.pkl')

print(f"\n    Saved: p300_fast.pkl")
print("\n" + "=" * 50)
print(f"  Fast P300 model ready!")
print(f"  Model : {best_name}")
print(f"  AUC   : {best_auc:.3f}")
print(f"  Speed : ~{avg_time:.1f} sec per word")
print("=" * 50)

input("\nPress Enter to close...")
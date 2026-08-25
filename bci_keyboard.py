from flask import Flask, render_template_string, jsonify, request
import threading
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import random
import time
import os
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════
# EEGNet definition
# ══════════════════════════════════════════════════════════════
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

# ══════════════════════════════════════════════════════════════
# P300 simulation
# ══════════════════════════════════════════════════════════════
SFREQ     = 256
EPOCH_LEN = 0.6
N_SAMPLES = int(SFREQ * EPOCH_LEN)
N_CH      = 8

def generate_p300_epoch(is_target):
    t     = np.linspace(0, EPOCH_LEN, N_SAMPLES)
    epoch = np.zeros((N_CH, N_SAMPLES))
    for ch in range(N_CH):
        alpha  = 3e-6 * np.sin(
            2*np.pi*10*t + np.random.rand()*2*np.pi)
        noise  = 2e-6 * np.random.randn(N_SAMPLES)
        signal = alpha + noise
        if is_target:
            lat = int(0.3  * SFREQ)
            w   = int(0.08 * SFREQ)
            wt  = [0.6,0.9,1.0,0.7,0.8,0.8,0.5,0.5]
            p3  = 4e-6 * wt[ch] * np.exp(
                -0.5*((np.arange(N_SAMPLES)-lat)/w)**2)
            signal += p3
        epoch[ch] = signal
    return epoch

def extract_rich_features(epochs):
    windows = [(0.00,0.10),(0.10,0.20),
               (0.20,0.35),(0.30,0.50),(0.45,0.60)]
    features = []
    for epoch in epochs:
        feat = []
        for ch in range(epoch.shape[0]):
            sig = epoch[ch]
            for ws, we in windows:
                s = int(ws*SFREQ); e = int(we*SFREQ)
                w = sig[s:e]
                feat.extend([w.mean(), w.max(),
                              w.min(), np.abs(w).mean()])
            p3 = sig[int(0.30*SFREQ):int(0.50*SFREQ)].mean()
            b  = sig[int(0.00*SFREQ):int(0.15*SFREQ)].mean()
            feat.append(p3 - b)
            feat.append(np.argmax(
                sig[int(0.20*SFREQ):int(0.55*SFREQ)]) / SFREQ)
            feat.append(np.trapezoid(
                sig[int(0.25*SFREQ):int(0.55*SFREQ)]))
        features.append(feat)
    return np.array(features)

# ══════════════════════════════════════════════════════════════
# Load models
# ══════════════════════════════════════════════════════════════
print("Starting BCI Keyboard server...")
device = torch.device('cpu')

CLASS_NAMES = ['Left hand','Right hand','Both hands','Feet']
MENU_NAMES  = ['Basic needs','Emotions','Actions','People']
N_CHANNELS  = 64
N_TP        = 321

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'eegnet_4class.pth')
P300_PATH  = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'p300_fast.pkl')

eegnet = None
lda    = None

if os.path.exists(MODEL_PATH):
    ck = torch.load(MODEL_PATH, map_location=device,
                    weights_only=False)
    N_CHANNELS = ck['n_channels']
    N_TP       = ck['n_timepoints']
    eegnet     = EEGNet4(N_CHANNELS, N_TP, n_classes=4)
    eegnet.load_state_dict(ck['model_state'])
    eegnet.eval()
    print(f"EEGNet loaded — accuracy: {ck['accuracy']:.2%}")
else:
    print("No EEGNet model found — using random simulation")

if os.path.exists(P300_PATH):
    p300_data = joblib.load(P300_PATH)
    lda       = p300_data['pipeline']
    print(f"P300 LDA loaded — AUC: {p300_data['auc']:.3f}")
else:
    print("No P300 model — training fresh...")
    np.random.seed(42)
    X_list, y_list = [], []
    for i in range(500):
        is_t = (i % 5 == 0)
        X_list.append(generate_p300_epoch(is_t))
        y_list.append(1 if is_t else 0)
    X_feat = extract_rich_features(np.array(X_list))
    y_arr  = np.array(y_list)
    lda    = Pipeline([('sc', StandardScaler()),
                       ('lda', LinearDiscriminantAnalysis())])
    lda.fit(X_feat, y_arr)
    print("P300 model trained!")

# ══════════════════════════════════════════════════════════════
# Vocabulary
# ══════════════════════════════════════════════════════════════
MENUS = {
    'Basic needs': ['WATER','FOOD','PAIN','TIRED',
                    'TOILET','HELP','MEDICINE','SLEEP'],
    'Emotions':    ['HAPPY','SAD','SCARED','ANGRY',
                    'LOVED','CALM','LONELY','CONFUSED'],
    'Actions':     ['YES','NO','STOP','WAIT',
                    'COME','GO','CALL','REPEAT'],
    'People':      ['DOCTOR','NURSE','MUM','DAD',
                    'FAMILY','FRIEND','ANYONE','EMERGENCY']
}

THOUGHT_TO_MENU = {
    0: 'Basic needs', 1: 'Emotions',
    2: 'Actions',     3: 'People'
}

MENU_COLORS = {
    'Basic needs': '#534AB7',
    'Emotions':    '#1D9E75',
    'Actions':     '#BA7517',
    'People':      '#D85A30',
    'Keyboard':    '#00d4aa'
}

# ── Alphabet keyboard layout ──────────────────────────────────
# 28 keys: A-Z + SPACE + DELETE
KEYBOARD_ROWS = [
    ['A','B','C','D','E','F','G'],
    ['H','I','J','K','L','M','N'],
    ['O','P','Q','R','S','T','U'],
    ['V','W','X','Y','Z','⎵','⌫'],
]
KEYBOARD_KEYS = [k for row in KEYBOARD_ROWS for k in row]

# Common words for autocomplete
WORD_LIST = [
    'WATER','FOOD','HELP','PAIN','YES','NO','STOP',
    'COME','CALL','HAPPY','SAD','TIRED','SLEEP',
    'DOCTOR','NURSE','MUM','DAD','FAMILY','FRIEND',
    'HELLO','THANK','PLEASE','SORRY','GOOD','NEED',
    'WANT','FEEL','LOVE','SCARED','ANGRY','CALM',
    'HOT','COLD','HUNGRY','THIRSTY','SICK','BETTER',
    'HOME','OUTSIDE','TOILET','MEDICINE','EMERGENCY',
]

def get_suggestions(current_word):
    """Return up to 3 word completions for current typed letters"""
    if not current_word:
        return []
    prefix = current_word.upper()
    matches = [w for w in WORD_LIST
               if w.startswith(prefix) and w != prefix]
    return matches[:3]

# ══════════════════════════════════════════════════════════════
# BCI State
# ══════════════════════════════════════════════════════════════
bci_state = {
    'mode':           'menu',      # 'menu' or 'keyboard'
    'detected_word':  '',
    'detected_menu':  '',
    'thought_class':  '',
    'confidence':     0.0,
    'sentence':       [],
    'history':        [],
    'status':         'Ready — tap Detect or switch to Keyboard',
    'processing':     False,
    # Keyboard specific
    'current_word':   '',          # letters typed so far
    'suggestions':    [],          # autocomplete suggestions
    'typed_words':    [],          # completed words from keyboard
}

def run_p300_detection(items, target_idx):
    """Run P300 flashing on a list of items, return best index + confidence"""
    scores = np.zeros(len(items))
    for _ in range(2):
        for idx in range(len(items)):
            is_target = (idx == target_idx)
            epoch = generate_p300_epoch(is_target)
            feat  = extract_rich_features(
                epoch[np.newaxis])[0].reshape(1, -1)
            scores[idx] += lda.predict_proba(feat)[0][1]
    avg        = scores / 2
    best_idx   = int(np.argmax(avg))
    confidence = float(avg[best_idx])
    return best_idx, confidence

def run_menu_pipeline():
    """Motor imagery → menu → P300 → word"""
    bci_state['processing'] = True
    bci_state['status']     = '🧠 Reading brain signals...'

    # Step 1 — motor imagery → menu
    thought_idx  = random.randint(0, 3)
    menu_name    = THOUGHT_TO_MENU[thought_idx]
    thought_name = CLASS_NAMES[thought_idx]
    bci_state['detected_menu'] = menu_name
    bci_state['thought_class'] = thought_name
    bci_state['status'] = f'Thought: {thought_name} → {menu_name}'
    time.sleep(0.3)

    # Step 2 — P300 → word selection
    words      = MENUS[menu_name]
    target_idx = random.randint(0, 7)
    best_idx, confidence = run_p300_detection(words, target_idx)
    best_word  = words[best_idx]

    bci_state['detected_word'] = best_word
    bci_state['confidence']    = round(confidence, 3)
    bci_state['sentence'].append(best_word)
    bci_state['history'].insert(0, {
        'word':       best_word,
        'menu':       menu_name,
        'thought':    thought_name,
        'confidence': f"{confidence:.2f}",
        'color':      MENU_COLORS[menu_name],
        'mode':       'menu'
    })
    bci_state['history']    = bci_state['history'][:10]
    bci_state['status']     = f'✓ Detected: {best_word}'
    bci_state['processing'] = False

def run_keyboard_pipeline():
    """P300 on alphabet keys → detect letter → build word"""
    bci_state['processing']   = True
    bci_state['detected_menu'] = 'Keyboard'
    bci_state['status']        = '⌨️ Scanning keyboard...'

    # P300 on 28 keys
    target_idx = random.randint(0, len(KEYBOARD_KEYS) - 1)
    best_idx, confidence = run_p300_detection(
        KEYBOARD_KEYS, target_idx)
    best_key = KEYBOARD_KEYS[best_idx]

    if best_key == '⎵':
        # SPACE — complete current word and add to sentence
        word = bci_state['current_word'].strip()
        if word:
            bci_state['sentence'].append(word)
            bci_state['typed_words'].append(word)
            bci_state['history'].insert(0, {
                'word':       word,
                'menu':       'Keyboard',
                'thought':    'Spelled',
                'confidence': f"{confidence:.2f}",
                'color':      '#00d4aa',
                'mode':       'keyboard'
            })
        bci_state['current_word']  = ''
        bci_state['suggestions']   = []
        bci_state['detected_word'] = f'[SPACE] → "{word}" added'
        bci_state['status']        = f'✓ Word "{word}" added to sentence'

    elif best_key == '⌫':
        # DELETE — remove last letter
        if bci_state['current_word']:
            bci_state['current_word'] = bci_state['current_word'][:-1]
        bci_state['detected_word'] = '[DELETE]'
        bci_state['status']        = '⌫ Letter deleted'
        bci_state['suggestions']   = get_suggestions(
            bci_state['current_word'])

    else:
        # Letter — add to current word
        bci_state['current_word'] += best_key
        bci_state['suggestions']   = get_suggestions(
            bci_state['current_word'])
        bci_state['detected_word'] = best_key
        bci_state['status'] = (
            f'✓ Letter: {best_key}  |  '
            f'Word so far: {bci_state["current_word"]}')
        bci_state['history'].insert(0, {
            'word':       best_key,
            'menu':       'Keyboard',
            'thought':    f'Spelling: {bci_state["current_word"]}',
            'confidence': f"{confidence:.2f}",
            'color':      '#00d4aa',
            'mode':       'keyboard'
        })

    bci_state['history']    = bci_state['history'][:10]
    bci_state['confidence'] = round(confidence, 3)
    bci_state['processing'] = False

# ══════════════════════════════════════════════════════════════
# HTML — full UI with keyboard mode
# ══════════════════════════════════════════════════════════════
HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,
        initial-scale=1, maximum-scale=1">
  <title>BCI — Thought to Speech</title>
  <style>
    :root {
      --bg:     #0f0f1a;
      --card:   #1a1a2e;
      --border: #2a2a4a;
      --text:   #e0e0ff;
      --dim:    #606080;
      --green:  #00d4aa;
      --purple: #534AB7;
      --teal:   #1D9E75;
      --amber:  #BA7517;
      --red:    #D85A30;
    }
    * { margin:0; padding:0; box-sizing:border-box;
        -webkit-tap-highlight-color:transparent; }
    body {
      background:var(--bg); color:var(--text);
      font-family:-apple-system,'Helvetica Neue',sans-serif;
      min-height:100vh; padding:12px;
      max-width:520px; margin:0 auto;
    }

    /* Header */
    .header { text-align:center; padding:12px 0 8px; }
    .header h1 {
      font-size:18px; font-weight:700;
      background:linear-gradient(135deg,#a0a0ff,#00d4aa);
      -webkit-background-clip:text;
      -webkit-text-fill-color:transparent;
      margin-bottom:3px;
    }
    .header p { font-size:11px; color:var(--dim); }

    /* Mode toggle */
    .mode-toggle {
      display:flex; gap:8px; margin-bottom:10px;
    }
    .mode-btn {
      flex:1; padding:10px; border:none; border-radius:10px;
      font-size:12px; font-weight:700; cursor:pointer;
      transition:all 0.2s;
    }
    .mode-btn.active {
      transform:scale(1.03);
      box-shadow:0 0 12px rgba(83,74,183,0.4);
    }
    .mode-menu { background:var(--purple); color:white; }
    .mode-key  { background:var(--green);  color:#0f0f1a; }

    /* Status */
    .status {
      background:var(--card); border:1px solid var(--border);
      border-radius:10px; padding:8px 14px;
      font-size:12px; color:var(--dim);
      text-align:center; margin-bottom:10px;
    }

    /* Word display */
    .word-card {
      background:var(--card); border:1px solid var(--border);
      border-radius:14px; padding:18px 14px 14px;
      text-align:center; margin-bottom:10px;
    }
    .menu-chip {
      display:inline-block; font-size:10px; font-weight:600;
      padding:2px 10px; border-radius:20px; margin-bottom:10px;
    }
    .big-word {
      font-size:48px; font-weight:800; color:var(--green);
      letter-spacing:2px; min-height:56px; margin-bottom:12px;
      transition:transform 0.2s;
    }
    .conf-track {
      background:var(--bg); border-radius:4px; height:5px;
      overflow:hidden; margin-bottom:4px;
    }
    .conf-fill {
      height:100%; border-radius:4px;
      background:linear-gradient(90deg,var(--purple),var(--green));
      transition:width 0.5s ease;
    }
    .conf-label { font-size:10px; color:var(--dim); }

    /* Sentence */
    .sentence-card {
      background:var(--card); border:1px solid var(--border);
      border-radius:12px; padding:12px 14px; margin-bottom:10px;
    }
    .sentence-label {
      font-size:9px; font-weight:700; color:var(--dim);
      letter-spacing:1px; margin-bottom:4px;
    }
    .sentence-text {
      font-size:18px; font-weight:600; min-height:24px;
      line-height:1.4; word-wrap:break-word;
    }

    /* ── KEYBOARD ── */
    .keyboard-section { display:none; }
    .keyboard-section.visible { display:block; }

    /* Current word display */
    .word-builder {
      background:var(--card); border:1px solid var(--green);
      border-radius:12px; padding:12px 14px; margin-bottom:10px;
    }
    .wb-label {
      font-size:9px; font-weight:700; color:var(--green);
      letter-spacing:1px; margin-bottom:4px;
    }
    .wb-text {
      font-size:26px; font-weight:700;
      color:var(--green); min-height:34px;
      letter-spacing:4px;
    }
    .wb-cursor {
      display:inline-block; width:2px; height:28px;
      background:var(--green); margin-left:2px;
      animation:blink 1s step-end infinite;
      vertical-align:bottom;
    }
    @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0} }

    /* Autocomplete suggestions */
    .suggestions {
      display:flex; gap:6px; margin-bottom:10px;
      flex-wrap:wrap;
    }
    .suggestion-btn {
      background:#1a2a3a; border:1px solid var(--green);
      border-radius:20px; padding:5px 12px;
      font-size:12px; font-weight:600; color:var(--green);
      cursor:pointer; transition:all 0.15s;
    }
    .suggestion-btn:active {
      background:var(--green); color:#0f0f1a;
    }

    /* Keyboard grid */
    .keyboard-grid { margin-bottom:10px; }
    .key-row {
      display:flex; gap:5px; margin-bottom:5px;
      justify-content:center;
    }
    .key {
      width:42px; height:42px; border:none; border-radius:8px;
      font-size:14px; font-weight:700; cursor:pointer;
      transition:transform 0.1s, background 0.1s;
      display:flex; align-items:center; justify-content:center;
      background:#1a1a3a; color:var(--text);
    }
    .key:active { transform:scale(0.92); }
    .key.flashing { background:#ffffff; color:#0f0f1a; }
    .key.selected-key { background:var(--green); color:#0f0f1a; }
    .key.space-key { width:60px; background:#1D9E75; color:white; font-size:10px; }
    .key.del-key   { width:60px; background:#D85A30; color:white; font-size:16px; }
    .key-hint {
      font-size:9px; color:var(--dim); text-align:center;
      margin-bottom:8px;
    }

    /* Buttons */
    .btn-grid {
      display:grid; grid-template-columns:1fr 1fr;
      gap:8px; margin-bottom:8px;
    }
    .btn-grid-3 {
      display:grid; grid-template-columns:1fr 1fr 1fr;
      gap:8px; margin-bottom:10px;
    }
    .btn {
      border:none; border-radius:10px; padding:14px 8px;
      font-size:12px; font-weight:700; cursor:pointer;
      transition:transform 0.1s;
      display:flex; flex-direction:column;
      align-items:center; gap:3px;
    }
    .btn:active { transform:scale(0.96); }
    .btn-icon { font-size:18px; }
    .btn-detect { background:var(--purple); color:white;
                  grid-column:1/-1; padding:18px; }
    .btn-speak  { background:var(--teal);  color:white; }
    .btn-repeat { background:#2a2a4a;      color:#a0a0ff; }
    .btn-delete { background:var(--amber); color:white; }
    .btn-clear  { background:#1a1a2e;
                  border:1px solid var(--border); color:var(--dim); }
    .btn-share  { background:#2a2a4a; color:#a0a0ff; }
    .btn-scan   {
      background:var(--green); color:#0f0f1a;
      grid-column:1/-1; padding:18px;
    }
    .btn-add-word {
      background:#1a2a3a; border:1px solid var(--green);
      color:var(--green);
    }

    /* History */
    .history-card {
      background:var(--card); border:1px solid var(--border);
      border-radius:12px; padding:12px 14px; margin-bottom:16px;
    }
    .history-label {
      font-size:9px; font-weight:700; color:var(--dim);
      letter-spacing:1px; margin-bottom:8px;
    }
    .hist-item {
      display:flex; align-items:center; gap:8px;
      padding:5px 0; border-bottom:1px solid #0f0f1a;
      font-size:11px;
    }
    .hist-item:last-child { border-bottom:none; }
    .hist-dot { width:7px; height:7px; border-radius:50%; flex-shrink:0; }
    .hist-word { font-size:13px; font-weight:700; flex:1; }
    .hist-meta { color:var(--dim); font-size:10px; }

    /* Spinner */
    @keyframes spin { to { transform:rotate(360deg); } }
    .spinner {
      display:inline-block; width:14px; height:14px;
      border:2px solid rgba(255,255,255,0.2);
      border-top-color:white; border-radius:50%;
      animation:spin 0.8s linear infinite;
      margin-right:5px; vertical-align:middle;
    }

    /* Online badge */
    .online-badge {
      display:inline-flex; align-items:center; gap:4px;
      background:#0a2a1a; border:1px solid var(--teal);
      border-radius:20px; padding:2px 8px;
      font-size:10px; color:var(--teal); margin-bottom:10px;
    }
    .online-dot {
      width:5px; height:5px; border-radius:50%;
      background:var(--teal);
      animation:pulse 2s ease-in-out infinite;
    }
    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.3} }
  </style>
</head>
<body>

<div class="header">
  <h1>🧠 BCI Thought-to-Speech</h1>
  <p>Brain signals → Words → Voice</p>
</div>

<div style="text-align:center;margin-bottom:10px">
  <span class="online-badge">
    <span class="online-dot"></span>AI model running
  </span>
</div>

<!-- Mode toggle -->
<div class="mode-toggle">
  <button class="mode-btn mode-menu active" id="btn-mode-menu"
          onclick="switchMode('menu')">
    📋 Word Menus
  </button>
  <button class="mode-btn mode-key" id="btn-mode-key"
          onclick="switchMode('keyboard')">
    ⌨️ Spell Any Word
  </button>
</div>

<div class="status" id="status">Ready — tap Detect Thought</div>

<!-- Word display -->
<div class="word-card">
  <div class="menu-chip" id="menu-chip"
       style="background:#2a2a4a;color:#8080b0">No menu open</div>
  <div class="big-word" id="big-word">---</div>
  <div class="conf-track">
    <div class="conf-fill" id="conf-fill" style="width:0%"></div>
  </div>
  <div class="conf-label" id="conf-label">Confidence: ---</div>
</div>

<!-- ══ KEYBOARD SECTION ══ -->
<div class="keyboard-section" id="keyboard-section">

  <!-- Current word being built -->
  <div class="word-builder">
    <div class="wb-label">SPELLING</div>
    <div class="wb-text" id="wb-text">
      <span id="wb-letters"></span>
      <span class="wb-cursor"></span>
    </div>
  </div>

  <!-- Autocomplete suggestions -->
  <div class="suggestions" id="suggestions"></div>

  <!-- Keyboard grid -->
  <div class="keyboard-grid" id="keyboard-grid"></div>
  <div class="key-hint">Tap "Scan Keyboard" to detect which letter you're focusing on</div>

  <!-- Keyboard actions -->
  <div class="btn-grid" style="margin-bottom:8px">
    <button class="btn btn-scan" id="scan-btn"
            onclick="scanKeyboard()">
      <span class="btn-icon">👁️</span>
      Scan Keyboard
    </button>
  </div>
  <div class="btn-grid">
    <button class="btn btn-add-word" onclick="addWordToSentence()">
      <span class="btn-icon">✅</span>
      Add Word
    </button>
    <button class="btn btn-delete" onclick="deleteLastLetter()">
      <span class="btn-icon">⌫</span>
      Delete Letter
    </button>
  </div>

</div>

<!-- Sentence -->
<div class="sentence-card">
  <div class="sentence-label">SENTENCE</div>
  <div class="sentence-text" id="sentence-text">
    Words appear here...
  </div>
</div>

<!-- ══ MENU MODE BUTTONS ══ -->
<div id="menu-buttons">
  <div class="btn-grid" style="margin-bottom:8px">
    <button class="btn btn-detect" id="detect-btn"
            onclick="detectThought()">
      <span class="btn-icon">🧠</span>
      Detect Thought
    </button>
  </div>
  <div class="btn-grid">
    <button class="btn btn-speak" onclick="speakSentence()">
      <span class="btn-icon">🔊</span>
      Speak
    </button>
    <button class="btn btn-repeat" onclick="repeatWord()">
      <span class="btn-icon">🔁</span>
      Repeat
    </button>
  </div>
</div>

<!-- Common buttons (both modes) -->
<div class="btn-grid-3" style="margin-top:8px">
  <button class="btn btn-delete" onclick="deleteLast()">
    <span class="btn-icon">⬅️</span>
    Delete
  </button>
  <button class="btn btn-clear" onclick="clearAll()">
    <span class="btn-icon">🗑️</span>
    Clear
  </button>
  <button class="btn btn-share" onclick="shareSentence()">
    <span class="btn-icon">📤</span>
    Share
  </button>
</div>

<!-- History -->
<div class="history-card">
  <div class="history-label">HISTORY</div>
  <div id="history-list">
    <div style="color:var(--dim);font-size:11px">No detections yet</div>
  </div>
</div>

<script>
const synth = window.speechSynthesis;
let lastWord  = '';
let curMode   = 'menu';

const MENU_COLORS = {
  'Basic needs':'#534AB7','Emotions':'#1D9E75',
  'Actions':'#BA7517','People':'#D85A30','Keyboard':'#00d4aa'
};

// ── Build keyboard grid ────────────────────────────────────
const ROWS = [
  ['A','B','C','D','E','F','G'],
  ['H','I','J','K','L','M','N'],
  ['O','P','Q','R','S','T','U'],
  ['V','W','X','Y','Z','⎵','⌫'],
];
const ALL_KEYS = ROWS.flat();

function buildKeyboard() {
  const grid = document.getElementById('keyboard-grid');
  grid.innerHTML = '';
  ROWS.forEach(row => {
    const rowDiv = document.createElement('div');
    rowDiv.className = 'key-row';
    row.forEach(k => {
      const btn = document.createElement('button');
      btn.className = 'key' +
        (k === '⎵' ? ' space-key' : '') +
        (k === '⌫' ? ' del-key'   : '');
      btn.id        = `key-${k}`;
      btn.textContent = k === '⎵' ? 'SPC' : k;
      btn.onclick   = () => manualKeyPress(k);
      rowDiv.appendChild(btn);
    });
    grid.appendChild(rowDiv);
  });
}
buildKeyboard();

// ── Mode switching ─────────────────────────────────────────
function switchMode(mode) {
  curMode = mode;
  document.getElementById('btn-mode-menu').classList
    .toggle('active', mode === 'menu');
  document.getElementById('btn-mode-key').classList
    .toggle('active', mode === 'keyboard');
  document.getElementById('keyboard-section').classList
    .toggle('visible', mode === 'keyboard');
  document.getElementById('menu-buttons').style.display =
    mode === 'menu' ? 'block' : 'none';
  document.getElementById('status').textContent =
    mode === 'menu'
      ? 'Menu mode — tap Detect Thought'
      : 'Keyboard mode — tap Scan Keyboard';
}

// ── Update UI ──────────────────────────────────────────────
function updateUI(data) {
  document.getElementById('status').textContent = data.status;

  const chip  = document.getElementById('menu-chip');
  const color = MENU_COLORS[data.detected_menu] || '#2a2a4a';
  chip.textContent      = data.detected_menu || 'No menu open';
  chip.style.background = color + '22';
  chip.style.color      = color || '#8080b0';
  chip.style.border     = `1px solid ${color}44`;

  const wordEl = document.getElementById('big-word');
  if (data.detected_word && data.detected_word !== wordEl.textContent) {
    wordEl.style.transform = 'scale(1.12)';
    setTimeout(() => wordEl.style.transform = 'scale(1)', 250);
  }
  wordEl.textContent = data.detected_word || '---';
  wordEl.style.color = data.detected_word ? '#00d4aa' : '#404060';
  lastWord = data.detected_word || lastWord;

  const conf = Math.round(data.confidence * 100);
  document.getElementById('conf-fill').style.width = conf + '%';
  document.getElementById('conf-label').textContent =
    conf > 0 ? `Confidence: ${conf}%` : 'Confidence: ---';

  // Sentence
  document.getElementById('sentence-text').textContent =
    data.sentence.join('  ') || 'Words appear here...';

  // Keyboard state
  document.getElementById('wb-letters').textContent =
    data.current_word || '';

  // Suggestions
  const sugDiv = document.getElementById('suggestions');
  sugDiv.innerHTML = '';
  if (data.suggestions && data.suggestions.length > 0) {
    data.suggestions.forEach(s => {
      const btn = document.createElement('button');
      btn.className   = 'suggestion-btn';
      btn.textContent = s;
      btn.onclick     = () => acceptSuggestion(s);
      sugDiv.appendChild(btn);
    });
  }

  // History
  let html = '';
  if (data.history && data.history.length > 0) {
    data.history.slice(0, 6).forEach(h => {
      html += `
        <div class="hist-item">
          <div class="hist-dot" style="background:${h.color}"></div>
          <div class="hist-word" style="color:${h.color}">${h.word}</div>
          <div class="hist-meta">${h.thought}</div>
          <div class="hist-meta">${h.confidence}</div>
        </div>`;
    });
  } else {
    html = '<div style="color:var(--dim);font-size:11px">No detections yet</div>';
  }
  document.getElementById('history-list').innerHTML = html;
}

// ── Menu mode ──────────────────────────────────────────────
function setDetectBtn(loading) {
  const btn = document.getElementById('detect-btn');
  if (btn) {
    btn.innerHTML = loading
      ? '<span class="spinner"></span> Reading...'
      : '<span class="btn-icon">🧠</span> Detect Thought';
    btn.disabled      = loading;
    btn.style.opacity = loading ? '0.7' : '1';
  }
}

function detectThought() {
  setDetectBtn(true);
  fetch('/detect', {method:'POST'})
    .then(r => r.json())
    .then(data => {
      setDetectBtn(false);
      updateUI(data);
      if (data.detected_word &&
          !data.detected_word.startsWith('[')) {
        speak(data.detected_word.toLowerCase());
      }
    })
    .catch(() => setDetectBtn(false));
}

// ── Keyboard mode ──────────────────────────────────────────
function setScanBtn(loading) {
  const btn = document.getElementById('scan-btn');
  if (btn) {
    btn.innerHTML = loading
      ? '<span class="spinner"></span> Scanning...'
      : '<span class="btn-icon">👁️</span> Scan Keyboard';
    btn.disabled      = loading;
    btn.style.opacity = loading ? '0.7' : '1';
  }
}

function scanKeyboard() {
  setScanBtn(true);
  document.getElementById('status').textContent =
    '👁️ Scanning keyboard — focus on your letter...';
  fetch('/keyboard_scan', {method:'POST'})
    .then(r => r.json())
    .then(data => {
      setScanBtn(false);
      updateUI(data);
      // Flash selected key
      flashKey(data.detected_word);
      // Speak letter
      if (data.detected_word && data.detected_word.length === 1) {
        speak(data.detected_word.toLowerCase());
      }
    })
    .catch(() => setScanBtn(false));
}

function flashKey(key) {
  // Remove all previous selections
  document.querySelectorAll('.key').forEach(k => {
    k.classList.remove('selected-key');
  });
  if (key && key.length === 1) {
    const el = document.getElementById(`key-${key}`);
    if (el) {
      el.classList.add('selected-key');
      setTimeout(() => el.classList.remove('selected-key'), 1500);
    }
  }
}

function manualKeyPress(key) {
  // Allow manual key press for demo/testing
  fetch('/manual_key', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({key: key})
  }).then(r => r.json()).then(data => {
    updateUI(data);
    if (key.length === 1) speak(key.toLowerCase());
  });
}

function addWordToSentence() {
  fetch('/add_word', {method:'POST'})
    .then(r => r.json())
    .then(data => {
      updateUI(data);
      if (data.typed_words && data.typed_words.length > 0) {
        const last = data.typed_words[data.typed_words.length-1];
        speak(last.toLowerCase());
      }
    });
}

function deleteLastLetter() {
  fetch('/delete_letter', {method:'POST'})
    .then(r => r.json())
    .then(data => updateUI(data));
}

function acceptSuggestion(word) {
  fetch('/accept_suggestion', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({word: word})
  }).then(r => r.json()).then(data => {
    updateUI(data);
    speak(word.toLowerCase());
  });
}

// ── Common actions ─────────────────────────────────────────
function speakSentence() {
  fetch('/state').then(r => r.json()).then(data => {
    const s = data.sentence.join(' ').toLowerCase();
    if (s.trim()) speak(s);
  });
}

function repeatWord() {
  if (lastWord && !lastWord.startsWith('[')) {
    speak(lastWord.toLowerCase());
  }
}

function deleteLast() {
  fetch('/delete', {method:'POST'})
    .then(r => r.json()).then(data => updateUI(data));
}

function clearAll() {
  fetch('/clear', {method:'POST'})
    .then(r => r.json()).then(data => {
      updateUI(data); lastWord = '';
    });
}

function shareSentence() {
  fetch('/state').then(r => r.json()).then(data => {
    const text = data.sentence.join(' ');
    if (navigator.share && text) {
      navigator.share({title:'BCI Message', text:text});
    } else if (text) {
      navigator.clipboard.writeText(text).then(() => {
        document.getElementById('status').textContent =
          '📋 Copied to clipboard!';
      });
    }
  });
}

function speak(text) {
  synth.cancel();
  const utt  = new SpeechSynthesisUtterance(text);
  utt.rate   = 0.85;
  utt.volume = 1.0;
  synth.speak(utt);
}

// Auto refresh
setInterval(() => {
  fetch('/state').then(r => r.json())
    .then(data => updateUI(data)).catch(()=>{});
}, 3000);
</script>
</body>
</html>
"""

# ══════════════════════════════════════════════════════════════
# Flask routes
# ══════════════════════════════════════════════════════════════
app = Flask(__name__)

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/state')
def state():
    return jsonify(bci_state)

@app.route('/detect', methods=['POST'])
def detect():
    if not bci_state['processing']:
        run_menu_pipeline()
    return jsonify(bci_state)

@app.route('/keyboard_scan', methods=['POST'])
def keyboard_scan():
    if not bci_state['processing']:
        run_keyboard_pipeline()
    return jsonify(bci_state)

@app.route('/manual_key', methods=['POST'])
def manual_key():
    """Manual key press for testing without hardware"""
    data = request.get_json()
    key  = data.get('key', '')
    if key == '⎵':
        word = bci_state['current_word'].strip()
        if word:
            bci_state['sentence'].append(word)
            bci_state['typed_words'].append(word)
        bci_state['current_word'] = ''
        bci_state['suggestions']  = []
        bci_state['detected_word'] = f'[SPACE]'
        bci_state['status'] = f'Word "{word}" added'
    elif key == '⌫':
        if bci_state['current_word']:
            bci_state['current_word'] = bci_state['current_word'][:-1]
        bci_state['suggestions'] = get_suggestions(bci_state['current_word'])
        bci_state['detected_word'] = '[DELETE]'
    else:
        bci_state['current_word'] += key
        bci_state['suggestions']   = get_suggestions(bci_state['current_word'])
        bci_state['detected_word'] = key
        bci_state['status'] = f'Letter: {key} | Word: {bci_state["current_word"]}'
    return jsonify(bci_state)

@app.route('/add_word', methods=['POST'])
def add_word():
    word = bci_state['current_word'].strip()
    if word:
        bci_state['sentence'].append(word)
        bci_state['typed_words'].append(word)
        bci_state['history'].insert(0, {
            'word':       word,
            'menu':       'Keyboard',
            'thought':    'Manual spell',
            'confidence': '1.00',
            'color':      '#00d4aa',
            'mode':       'keyboard'
        })
        bci_state['status'] = f'✓ "{word}" added to sentence'
    bci_state['current_word'] = ''
    bci_state['suggestions']  = []
    return jsonify(bci_state)

@app.route('/delete_letter', methods=['POST'])
def delete_letter():
    if bci_state['current_word']:
        bci_state['current_word'] = bci_state['current_word'][:-1]
        bci_state['suggestions']  = get_suggestions(
            bci_state['current_word'])
    return jsonify(bci_state)

@app.route('/accept_suggestion', methods=['POST'])
def accept_suggestion():
    data = request.get_json()
    word = data.get('word', '').upper()
    bci_state['sentence'].append(word)
    bci_state['typed_words'].append(word)
    bci_state['current_word'] = ''
    bci_state['suggestions']  = []
    bci_state['detected_word'] = word
    bci_state['status'] = f'✓ Suggestion "{word}" added'
    bci_state['history'].insert(0, {
        'word':       word,
        'menu':       'Keyboard',
        'thought':    'Autocomplete',
        'confidence': '1.00',
        'color':      '#00d4aa',
        'mode':       'keyboard'
    })
    return jsonify(bci_state)

@app.route('/delete', methods=['POST'])
def delete():
    if bci_state['sentence']:
        bci_state['sentence'].pop()
    bci_state['detected_word'] = ''
    bci_state['status'] = 'Word deleted'
    return jsonify(bci_state)

@app.route('/clear', methods=['POST'])
def clear():
    bci_state['sentence']      = []
    bci_state['history']       = []
    bci_state['detected_word'] = ''
    bci_state['detected_menu'] = ''
    bci_state['confidence']    = 0.0
    bci_state['current_word']  = ''
    bci_state['suggestions']   = []
    bci_state['typed_words']   = []
    bci_state['status']        = 'Ready — tap Detect or switch to Keyboard'
    return jsonify(bci_state)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"\nBCI Keyboard server running!")
    print(f"Open: http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
from flask import Flask, render_template_string, jsonify
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
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
            return x.view(1, -1).shape[1]

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.classifier(x.view(x.size(0), -1))

# ══════════════════════════════════════════════════════════════
# P300 simulation + features
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
            lat = int(0.3 * SFREQ)
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
# Load or train models at startup
# ══════════════════════════════════════════════════════════════
print("Starting BCI server...")
device = torch.device('cpu')

# Try loading saved models first
# If not found, train fresh ones (cloud deployment)
eegnet    = None
lda       = None
ck        = None

MODEL_PATH = os.path.join(os.path.dirname(__file__),
                          'eegnet_4class.pth')
P300_PATH  = os.path.join(os.path.dirname(__file__),
                          'p300_fast.pkl')

CLASS_NAMES = ['Left hand','Right hand','Both hands','Feet']
MENU_NAMES  = ['Basic needs','Emotions','Actions','People']
N_CHANNELS  = 64
N_TP        = 321

if os.path.exists(MODEL_PATH):
    print("Loading saved EEGNet model...")
    ck = torch.load(
    os.path.join(os.path.dirname(__file__), 'eegnet_4class.pth'),
    map_location=device, weights_only=False)
                    
    N_CHANNELS = ck['n_channels']
    N_TP       = ck['n_timepoints']
    eegnet     = EEGNet4(N_CHANNELS, N_TP, n_classes=4)
    eegnet.load_state_dict(ck['model_state'])
    eegnet.eval()
    print(f"EEGNet loaded! Accuracy: {ck['accuracy']:.2%}")
else:
    print("No saved EEGNet — using random simulation")

if os.path.exists(P300_PATH):
    print("Loading saved P300 model...")
    p300_data = joblib.load(
    os.path.join(os.path.dirname(__file__), 'p300_fast.pkl'))
    lda       = p300_data['pipeline']
    print(f"P300 LDA loaded! AUC: {p300_data['auc']:.3f}")
else:
    print("No saved P300 model — training fresh...")
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
# BCI vocabulary
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
    0: 'Basic needs',
    1: 'Emotions',
    2: 'Actions',
    3: 'People'
}

MENU_COLORS = {
    'Basic needs': '#534AB7',
    'Emotions':    '#1D9E75',
    'Actions':     '#BA7517',
    'People':      '#D85A30'
}

bci_state = {
    'detected_word':  '',
    'detected_menu':  '',
    'thought_class':  '',
    'confidence':     0.0,
    'sentence':       [],
    'history':        [],
    'status':         'Ready — tap Detect Thought',
    'processing':     False,
}

def run_bci_pipeline():
    bci_state['processing'] = True
    bci_state['status']     = '🧠 Reading brain signals...'

    # Step 1 — Motor imagery → menu selection
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
    scores     = np.zeros(8)

    for _ in range(2):
        for idx in range(8):
            epoch = generate_p300_epoch(idx == target_idx)
            feat  = extract_rich_features(
                epoch[np.newaxis])[0].reshape(1, -1)
            scores[idx] += lda.predict_proba(feat)[0][1]

    best_idx   = int(np.argmax(scores / 2))
    best_word  = words[best_idx]
    confidence = float(scores[best_idx] / 2)

    bci_state['detected_word'] = best_word
    bci_state['confidence']    = round(confidence, 3)
    bci_state['sentence'].append(best_word)
    bci_state['history'].insert(0, {
        'word':       best_word,
        'menu':       menu_name,
        'thought':    thought_name,
        'confidence': f"{confidence:.2f}",
        'color':      MENU_COLORS[menu_name]
    })
    bci_state['history'] = bci_state['history'][:10]
    bci_state['status']  = f'✓ Detected: {best_word}'
    bci_state['processing'] = False

# ══════════════════════════════════════════════════════════════
# HTML — beautiful mobile-first UI
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
      background:var(--bg);
      color:var(--text);
      font-family:-apple-system, 'Helvetica Neue', sans-serif;
      min-height:100vh;
      padding:16px;
      max-width:480px;
      margin:0 auto;
    }

    /* Header */
    .header {
      text-align:center;
      padding:16px 0 12px;
    }
    .header h1 {
      font-size:20px;
      font-weight:700;
      background:linear-gradient(135deg, #a0a0ff, #00d4aa);
      -webkit-background-clip:text;
      -webkit-text-fill-color:transparent;
      margin-bottom:4px;
    }
    .header p {
      font-size:12px;
      color:var(--dim);
    }

    /* Status */
    .status {
      background:var(--card);
      border:1px solid var(--border);
      border-radius:12px;
      padding:10px 16px;
      font-size:13px;
      color:var(--dim);
      text-align:center;
      margin-bottom:12px;
    }

    /* Word card */
    .word-card {
      background:var(--card);
      border:1px solid var(--border);
      border-radius:16px;
      padding:24px 16px 20px;
      text-align:center;
      margin-bottom:12px;
      position:relative;
    }
    .menu-chip {
      display:inline-block;
      font-size:11px;
      font-weight:600;
      padding:3px 12px;
      border-radius:20px;
      margin-bottom:14px;
      letter-spacing:0.5px;
    }
    .big-word {
      font-size:56px;
      font-weight:800;
      color:var(--green);
      letter-spacing:3px;
      line-height:1;
      min-height:60px;
      margin-bottom:16px;
      word-break:break-all;
    }
    .conf-track {
      background:var(--bg);
      border-radius:6px;
      height:6px;
      overflow:hidden;
      margin-bottom:6px;
    }
    .conf-fill {
      height:100%;
      border-radius:6px;
      background:linear-gradient(90deg,var(--purple),var(--green));
      transition:width 0.6s cubic-bezier(.4,0,.2,1);
    }
    .conf-label {
      font-size:11px;
      color:var(--dim);
    }

    /* Sentence */
    .sentence-card {
      background:var(--card);
      border:1px solid var(--border);
      border-radius:14px;
      padding:14px 16px;
      margin-bottom:12px;
    }
    .sentence-label {
      font-size:10px;
      font-weight:600;
      color:var(--dim);
      letter-spacing:1px;
      margin-bottom:6px;
    }
    .sentence-text {
      font-size:20px;
      font-weight:600;
      color:var(--text);
      min-height:28px;
      line-height:1.4;
      word-wrap:break-word;
    }

    /* Buttons */
    .btn-grid {
      display:grid;
      grid-template-columns:1fr 1fr;
      gap:10px;
      margin-bottom:10px;
    }
    .btn-grid-3 {
      display:grid;
      grid-template-columns:1fr 1fr 1fr;
      gap:10px;
      margin-bottom:12px;
    }
    .btn {
      border:none;
      border-radius:12px;
      padding:16px 8px;
      font-size:13px;
      font-weight:700;
      cursor:pointer;
      transition:transform 0.1s, opacity 0.2s;
      display:flex;
      flex-direction:column;
      align-items:center;
      gap:4px;
    }
    .btn:active { transform:scale(0.96); opacity:0.8; }
    .btn-icon { font-size:20px; }
    .btn-detect { background:var(--purple); color:white;
                  grid-column:1/-1; padding:20px; }
    .btn-speak  { background:var(--teal);  color:white; }
    .btn-repeat { background:#2a2a4a;      color:#a0a0ff; }
    .btn-delete { background:var(--amber); color:white; }
    .btn-clear  { background:#1a1a2e;
                  border:1px solid var(--border);
                  color:var(--dim); }
    .btn-share  { background:#2a2a4a;      color:#a0a0ff; }

    /* Processing state */
    @keyframes spin {
      to { transform:rotate(360deg); }
    }
    .spinner {
      display:inline-block;
      width:16px; height:16px;
      border:2px solid rgba(255,255,255,0.2);
      border-top-color:white;
      border-radius:50%;
      animation:spin 0.8s linear infinite;
      margin-right:6px;
      vertical-align:middle;
    }

    /* History */
    .history-card {
      background:var(--card);
      border:1px solid var(--border);
      border-radius:14px;
      padding:14px 16px;
      margin-bottom:20px;
    }
    .history-label {
      font-size:10px;
      font-weight:600;
      color:var(--dim);
      letter-spacing:1px;
      margin-bottom:10px;
    }
    .hist-item {
      display:flex;
      align-items:center;
      gap:10px;
      padding:7px 0;
      border-bottom:1px solid #0f0f1a;
    }
    .hist-item:last-child { border-bottom:none; }
    .hist-dot {
      width:8px; height:8px;
      border-radius:50%;
      flex-shrink:0;
    }
    .hist-word {
      font-size:14px;
      font-weight:700;
      flex:1;
    }
    .hist-menu {
      font-size:11px;
      color:var(--dim);
    }
    .hist-conf {
      font-size:11px;
      color:var(--dim);
    }

    /* Online indicator */
    .online-badge {
      display:inline-flex;
      align-items:center;
      gap:5px;
      background:#0a2a1a;
      border:1px solid #1D9E75;
      border-radius:20px;
      padding:3px 10px;
      font-size:11px;
      color:#1D9E75;
      margin-bottom:14px;
    }
    .online-dot {
      width:6px; height:6px;
      border-radius:50%;
      background:#1D9E75;
      animation:pulse 2s ease-in-out infinite;
    }
    @keyframes pulse {
      0%,100%{opacity:1} 50%{opacity:0.3}
    }
  </style>
</head>
<body>

<div class="header">
  <h1>🧠 BCI Thought-to-Speech</h1>
  <p>Brain signals → AI → Voice</p>
</div>

<div style="text-align:center;margin-bottom:12px">
  <span class="online-badge">
    <span class="online-dot"></span>
    Live — AI model running
  </span>
</div>

<div class="status" id="status">
  Ready — tap Detect Thought
</div>

<!-- Main word display -->
<div class="word-card">
  <div class="menu-chip" id="menu-chip"
       style="background:#2a2a4a;color:#8080b0">
    No menu open
  </div>
  <div class="big-word" id="big-word">---</div>
  <div class="conf-track">
    <div class="conf-fill" id="conf-fill" style="width:0%"></div>
  </div>
  <div class="conf-label" id="conf-label">Confidence: ---</div>
</div>

<!-- Sentence -->
<div class="sentence-card">
  <div class="sentence-label">SENTENCE</div>
  <div class="sentence-text" id="sentence-text">
    Detected words appear here...
  </div>
</div>

<!-- Main detect button -->
<div class="btn-grid" style="margin-bottom:10px">
  <button class="btn btn-detect" id="detect-btn"
          onclick="detectThought()">
    <span class="btn-icon">🧠</span>
    Detect Thought
  </button>
</div>

<!-- Action buttons -->
<div class="btn-grid">
  <button class="btn btn-speak" onclick="speakSentence()">
    <span class="btn-icon">🔊</span>
    Speak Sentence
  </button>
  <button class="btn btn-repeat" onclick="repeatWord()">
    <span class="btn-icon">🔁</span>
    Repeat Word
  </button>
</div>

<div class="btn-grid-3">
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
  <div class="history-label">DETECTION HISTORY</div>
  <div id="history-list">
    <div style="color:var(--dim);font-size:12px">
      No detections yet
    </div>
  </div>
</div>

<script>
  const synth  = window.speechSynthesis;
  let lastWord = '';

  const MENU_COLORS = {
    'Basic needs': '#534AB7',
    'Emotions':    '#1D9E75',
    'Actions':     '#BA7517',
    'People':      '#D85A30'
  };

  function updateUI(data) {
    // Status
    document.getElementById('status').textContent = data.status;

    // Menu chip
    const chip  = document.getElementById('menu-chip');
    const color = MENU_COLORS[data.detected_menu] || '#2a2a4a';
    chip.textContent       = data.detected_menu || 'No menu open';
    chip.style.background  = color + '22';
    chip.style.color       = color || '#8080b0';
    chip.style.border      = `1px solid ${color}44`;

    // Big word
    const wordEl = document.getElementById('big-word');
    if (data.detected_word && data.detected_word !== wordEl.textContent) {
      wordEl.style.transform = 'scale(1.15)';
      setTimeout(() => wordEl.style.transform = 'scale(1)', 300);
    }
    wordEl.textContent = data.detected_word || '---';
    wordEl.style.color = data.detected_word ? '#00d4aa' : '#404060';
    lastWord = data.detected_word;

    // Confidence
    const conf = Math.round(data.confidence * 100);
    document.getElementById('conf-fill').style.width  = conf + '%';
    document.getElementById('conf-label').textContent =
      conf > 0 ? `Confidence: ${conf}%` : 'Confidence: ---';

    // Sentence
    const sent = data.sentence.join('   ');
    document.getElementById('sentence-text').textContent =
      sent || 'Detected words appear here...';

    // History
    let html = '';
    if (data.history && data.history.length > 0) {
      for (const h of data.history.slice(0, 6)) {
        html += `
          <div class="hist-item">
            <div class="hist-dot"
                 style="background:${h.color || '#534AB7'}"></div>
            <div class="hist-word"
                 style="color:${h.color || '#e0e0ff'}">${h.word}</div>
            <div class="hist-menu">${h.menu}</div>
            <div class="hist-conf">${h.confidence}</div>
          </div>`;
      }
    } else {
      html = '<div style="color:var(--dim);font-size:12px">' +
             'No detections yet</div>';
    }
    document.getElementById('history-list').innerHTML = html;
  }

  function setDetectBtn(loading) {
    const btn = document.getElementById('detect-btn');
    if (loading) {
      btn.innerHTML = '<span class="spinner"></span> Reading...';
      btn.disabled  = true;
      btn.style.opacity = '0.7';
    } else {
      btn.innerHTML = '<span class="btn-icon">🧠</span> Detect Thought';
      btn.disabled  = false;
      btn.style.opacity = '1';
    }
  }

  function detectThought() {
    setDetectBtn(true);
    document.getElementById('status').textContent =
      '🧠 Reading brain signals...';

    fetch('/detect', {method:'POST'})
      .then(r => r.json())
      .then(data => {
        setDetectBtn(false);
        updateUI(data);
        // Auto-speak detected word
        if (data.detected_word) {
          speak(data.detected_word.toLowerCase());
        }
      })
      .catch(() => {
        setDetectBtn(false);
        document.getElementById('status').textContent =
          'Error — try again';
      });
  }

  function speakSentence() {
    fetch('/state')
      .then(r => r.json())
      .then(data => {
        const sentence = data.sentence.join(' ').toLowerCase();
        if (sentence.trim()) {
          speak(sentence);
          document.getElementById('status').textContent =
            '🔊 Speaking: ' + sentence.toUpperCase();
        }
      });
  }

  function repeatWord() {
    if (lastWord) {
      speak(lastWord.toLowerCase());
      document.getElementById('status').textContent =
        '🔁 Repeating: ' + lastWord;
    }
  }

  function deleteLast() {
    fetch('/delete', {method:'POST'})
      .then(r => r.json())
      .then(data => updateUI(data));
  }

  function clearAll() {
    fetch('/clear', {method:'POST'})
      .then(r => r.json())
      .then(data => {
        updateUI(data);
        lastWord = '';
      });
  }

  function shareSentence() {
    fetch('/state')
      .then(r => r.json())
      .then(data => {
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
    const utt    = new SpeechSynthesisUtterance(text);
    utt.rate     = 0.85;
    utt.volume   = 1.0;
    utt.pitch    = 1.0;
    synth.speak(utt);
  }

  // Refresh state every 3 seconds
  setInterval(() => {
    fetch('/state')
      .then(r => r.json())
      .then(data => updateUI(data))
      .catch(() => {});
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

@app.route('/detect', methods=['POST'])
def detect():
    if not bci_state['processing']:
        run_bci_pipeline()
    return jsonify(bci_state)

@app.route('/state')
def state():
    return jsonify(bci_state)

@app.route('/delete', methods=['POST'])
def delete():
    if bci_state['sentence']:
        bci_state['sentence'].pop()
        if bci_state['history']:
            bci_state['history'].pop(0)
    bci_state['detected_word'] = \
        bci_state['sentence'][-1] \
        if bci_state['sentence'] else ''
    bci_state['status'] = 'Word deleted'
    return jsonify(bci_state)

@app.route('/clear', methods=['POST'])
def clear():
    bci_state['sentence']      = []
    bci_state['history']       = []
    bci_state['detected_word'] = ''
    bci_state['detected_menu'] = ''
    bci_state['confidence']    = 0.0
    bci_state['status']        = 'Ready — tap Detect Thought'
    return jsonify(bci_state)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
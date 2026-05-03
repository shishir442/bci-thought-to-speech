print("BCI Phone Display Server")
print("=" * 50)

from flask import Flask, render_template_string, jsonify
import threading
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import time
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
        alpha  = 3e-6 * np.sin(2*np.pi*10*t + np.random.rand()*2*np.pi)
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
            p3  = sig[int(0.30*SFREQ):int(0.50*SFREQ)].mean()
            b   = sig[int(0.00*SFREQ):int(0.15*SFREQ)].mean()
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
print("\n[1/3] Loading models...")
device = torch.device('cpu')

ck = torch.load(
    r'C:\Users\SHISHIR\Desktop\BCI Project\eegnet_4class.pth',
    map_location=device, weights_only=False)
eegnet = EEGNet4(ck['n_channels'], ck['n_timepoints'],
                 n_classes=4).to(device)
eegnet.load_state_dict(ck['model_state'])
eegnet.eval()

p300_data = joblib.load(
    r'C:\Users\SHISHIR\Desktop\BCI Project\p300_fast.pkl')
lda = p300_data['pipeline']

print("    Models loaded!")

# ══════════════════════════════════════════════════════════════
# BCI State
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

bci_state = {
    'detected_word':   '',
    'detected_menu':   '',
    'confidence':      0.0,
    'sentence':        [],
    'history':         [],
    'status':          'Ready',
    'processing':      False,
    'thought_class':   '',
}

def simulate_thought_and_word():
    """
    Simulate full BCI pipeline:
    1. EEGNet detects thought → menu
    2. P300 detects word in that menu
    """
    bci_state['processing'] = True
    bci_state['status']     = 'Reading brain signals...'

    # Step 1 — Simulate motor imagery (EEGNet)
    thought_idx  = random.randint(0, 3)
    menu_name    = THOUGHT_TO_MENU[thought_idx]
    thought_name = ck['class_names'][thought_idx]
    bci_state['detected_menu']  = menu_name
    bci_state['thought_class']  = thought_name
    bci_state['status']         = f'Thought detected: {thought_name} → {menu_name}'

    time.sleep(0.5)

    # Step 2 — Simulate P300 word selection
    words      = MENUS[menu_name]
    target_idx = random.randint(0, 7)
    scores     = np.zeros(8)

    for _ in range(2):   # 2 flash rounds
        for idx in range(8):
            epoch = generate_p300_epoch(idx == target_idx)
            feat  = extract_rich_features(
                epoch[np.newaxis])[0].reshape(1, -1)
            scores[idx] += lda.predict_proba(feat)[0][1]

    best_idx   = int(np.argmax(scores / 2))
    best_word  = words[best_idx]
    confidence = float(scores[best_idx] / 2)

    bci_state['detected_word'] = best_word
    bci_state['confidence']    = confidence
    bci_state['sentence'].append(best_word)
    bci_state['history'].append({
        'word':       best_word,
        'menu':       menu_name,
        'thought':    thought_name,
        'confidence': f"{confidence:.2f}"
    })
    bci_state['status']     = f'Detected: {best_word}'
    bci_state['processing'] = False

# ══════════════════════════════════════════════════════════════
# Flask web app
# ══════════════════════════════════════════════════════════════
print("\n[2/3] Setting up web server...")

app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>BCI Device</title>
  <style>
    * { margin:0; padding:0; box-sizing:border-box; }
    body {
      background:#0f0f1a;
      color:#e0e0ff;
      font-family:'Helvetica Neue', sans-serif;
      min-height:100vh;
      display:flex;
      flex-direction:column;
      align-items:center;
      padding:20px;
    }

    h1 {
      font-size:20px;
      font-weight:600;
      color:#a0a0ff;
      margin-bottom:4px;
      text-align:center;
    }

    .subtitle {
      font-size:12px;
      color:#404060;
      margin-bottom:20px;
      text-align:center;
    }

    /* Status bar */
    .status-bar {
      width:100%;
      max-width:500px;
      background:#1a1a2e;
      border-radius:10px;
      padding:10px 16px;
      font-size:13px;
      color:#8080b0;
      margin-bottom:16px;
      text-align:center;
      border:1px solid #2a2a4a;
    }

    /* Main word display */
    .word-display {
      width:100%;
      max-width:500px;
      background:#1a1a2e;
      border-radius:16px;
      padding:30px 20px;
      text-align:center;
      margin-bottom:16px;
      border:1px solid #2a2a4a;
    }

    .menu-tag {
      display:inline-block;
      font-size:11px;
      font-weight:500;
      padding:3px 10px;
      border-radius:20px;
      background:#2a2a4a;
      color:#8080b0;
      margin-bottom:12px;
    }

    .detected-word {
      font-size:52px;
      font-weight:700;
      color:#00d4aa;
      letter-spacing:2px;
      margin-bottom:8px;
      min-height:64px;
    }

    .confidence-bar-bg {
      width:100%;
      background:#0f0f1a;
      border-radius:6px;
      height:8px;
      margin-top:12px;
    }

    .confidence-bar-fill {
      height:8px;
      border-radius:6px;
      background:linear-gradient(90deg, #534AB7, #00d4aa);
      transition:width 0.5s ease;
    }

    .conf-text {
      font-size:11px;
      color:#404060;
      margin-top:6px;
    }

    /* Sentence display */
    .sentence-box {
      width:100%;
      max-width:500px;
      background:#1a1a2e;
      border-radius:12px;
      padding:16px;
      margin-bottom:16px;
      border:1px solid #2a2a4a;
    }

    .sentence-label {
      font-size:11px;
      color:#404060;
      margin-bottom:6px;
    }

    .sentence-text {
      font-size:18px;
      font-weight:600;
      color:#e0e0ff;
      min-height:28px;
      word-wrap:break-word;
    }

    /* Buttons */
    .btn-row {
      display:flex;
      gap:10px;
      width:100%;
      max-width:500px;
      margin-bottom:16px;
      flex-wrap:wrap;
    }

    .btn {
      flex:1;
      padding:14px 10px;
      border:none;
      border-radius:10px;
      font-size:13px;
      font-weight:600;
      cursor:pointer;
      transition:opacity 0.2s;
    }
    .btn:hover { opacity:0.85; }
    .btn:active { opacity:0.7; }

    .btn-detect  { background:#534AB7; color:white; }
    .btn-speak   { background:#1D9E75; color:white; }
    .btn-delete  { background:#BA7517; color:white; }
    .btn-clear   { background:#333350; color:#a0a0c0; }

    /* Processing spinner */
    .processing {
      display:none;
      color:#534AB7;
      font-size:13px;
      margin-bottom:10px;
    }

    /* History */
    .history-box {
      width:100%;
      max-width:500px;
      background:#1a1a2e;
      border-radius:12px;
      padding:14px 16px;
      border:1px solid #2a2a4a;
    }

    .history-label {
      font-size:11px;
      color:#404060;
      margin-bottom:8px;
    }

    .history-item {
      display:flex;
      justify-content:space-between;
      align-items:center;
      padding:6px 0;
      border-bottom:1px solid #0f0f1a;
      font-size:12px;
    }

    .history-item:last-child { border-bottom:none; }
    .hw { color:#00d4aa; font-weight:600; }
    .hm { color:#534AB7; font-size:11px; }
    .hc { color:#404060; font-size:11px; }

    /* Speaking animation */
    @keyframes pulse {
      0%,100% { opacity:1; }
      50% { opacity:0.4; }
    }
    .speaking { animation:pulse 1s ease-in-out infinite; }
  </style>
</head>
<body>

<h1>🧠 BCI Thought-to-Speech</h1>
<p class="subtitle">Brain signals → Words → Voice</p>

<div class="status-bar" id="status">Ready — press Detect Thought</div>

<div class="word-display">
  <div class="menu-tag" id="menu-tag">No menu open</div>
  <div class="detected-word" id="word-display">---</div>
  <div class="confidence-bar-bg">
    <div class="confidence-bar-fill" id="conf-bar" style="width:0%"></div>
  </div>
  <div class="conf-text" id="conf-text">Confidence: ---</div>
</div>

<div class="sentence-box">
  <div class="sentence-label">SENTENCE</div>
  <div class="sentence-text" id="sentence-display">Words will appear here...</div>
</div>

<div class="processing" id="processing">
  ⏳ Reading brain signals...
</div>

<div class="btn-row">
  <button class="btn btn-detect" onclick="detectThought()">
    🧠 Detect Thought
  </button>
  <button class="btn btn-speak" onclick="speakSentence()">
    🔊 Speak
  </button>
</div>

<div class="btn-row">
  <button class="btn btn-delete" onclick="deleteLast()">
    ← Delete
  </button>
  <button class="btn btn-clear" onclick="clearAll()">
    Clear All
  </button>
</div>

<div class="history-box">
  <div class="history-label">DETECTION HISTORY</div>
  <div id="history-list">
    <div style="color:#404060;font-size:12px">No detections yet</div>
  </div>
</div>

<script>
  let synth = window.speechSynthesis;

  function updateDisplay(data) {
    document.getElementById('status').textContent = data.status;
    document.getElementById('menu-tag').textContent =
      data.detected_menu || 'No menu open';
    document.getElementById('word-display').textContent =
      data.detected_word || '---';

    let conf = Math.round(data.confidence * 100);
    document.getElementById('conf-bar').style.width = conf + '%';
    document.getElementById('conf-text').textContent =
      'Confidence: ' + conf + '%';

    let sent = data.sentence.join('  ');
    document.getElementById('sentence-display').textContent =
      sent || 'Words will appear here...';

    // Update history
    let hist = data.history.slice(-5).reverse();
    let histHtml = '';
    for (let h of hist) {
      histHtml += `
        <div class="history-item">
          <span class="hw">${h.word}</span>
          <span class="hm">${h.menu}</span>
          <span class="hc">conf: ${h.confidence}</span>
        </div>`;
    }
    document.getElementById('history-list').innerHTML =
      histHtml || '<div style="color:#404060;font-size:12px">No detections yet</div>';
  }

  function detectThought() {
    document.getElementById('processing').style.display = 'block';
    document.getElementById('status').textContent = 'Reading brain signals...';

    fetch('/detect', {method: 'POST'})
      .then(r => r.json())
      .then(data => {
        document.getElementById('processing').style.display = 'none';
        updateDisplay(data);

        // Auto speak detected word
        if (data.detected_word && data.detected_word !== '---') {
          let utt = new SpeechSynthesisUtterance(
            data.detected_word.toLowerCase());
          utt.rate   = 0.9;
          utt.volume = 1.0;
          synth.speak(utt);
        }
      });
  }

  function speakSentence() {
    fetch('/state')
      .then(r => r.json())
      .then(data => {
        let sentence = data.sentence.join(' ');
        if (sentence) {
          let utt = new SpeechSynthesisUtterance(sentence.toLowerCase());
          utt.rate   = 0.85;
          utt.volume = 1.0;
          synth.speak(utt);
          document.getElementById('status').textContent =
            'Speaking: ' + sentence;
        }
      });
  }

  function deleteLast() {
    fetch('/delete', {method:'POST'})
      .then(r => r.json())
      .then(data => updateDisplay(data));
  }

  function clearAll() {
    fetch('/clear', {method:'POST'})
      .then(r => r.json())
      .then(data => updateDisplay(data));
  }

  // Auto refresh state every 2 seconds
  setInterval(() => {
    fetch('/state')
      .then(r => r.json())
      .then(data => updateDisplay(data));
  }, 2000);
</script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/detect', methods=['POST'])
def detect():
    if not bci_state['processing']:
        t = threading.Thread(
            target=simulate_thought_and_word, daemon=True)
        t.start()
        t.join()   # wait for detection to complete
    return jsonify(bci_state)

@app.route('/state')
def state():
    return jsonify(bci_state)

@app.route('/delete', methods=['POST'])
def delete():
    if bci_state['sentence']:
        bci_state['sentence'].pop()
        if bci_state['history']:
            bci_state['history'].pop()
    bci_state['detected_word'] = \
        bci_state['sentence'][-1] if bci_state['sentence'] else ''
    return jsonify(bci_state)

@app.route('/clear', methods=['POST'])
def clear():
    bci_state['sentence']      = []
    bci_state['history']       = []
    bci_state['detected_word'] = ''
    bci_state['detected_menu'] = ''
    bci_state['confidence']    = 0.0
    bci_state['status']        = 'Ready'
    return jsonify(bci_state)

# ══════════════════════════════════════════════════════════════
# Start server
# ══════════════════════════════════════════════════════════════
print("\n[3/3] Starting server...")
print("\n" + "=" * 50)
print("  BCI Server is LIVE!")
print("  Open on THIS computer : http://localhost:5000")
print("  Open on PHONE/TV      : http://192.168.x.x:5000")
print("  (replace x.x with your WiFi IP address)")
print("=" * 50)
print("\n  To find your WiFi IP — open a NEW terminal and type:")
print("  ipconfig")
print("  Look for 'IPv4 Address' under WiFi")
print("\n  Press Ctrl+C to stop the server")

app.run(host='0.0.0.0', port=5000, debug=False)
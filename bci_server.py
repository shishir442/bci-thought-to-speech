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
import json
import datetime
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════
# EEGNet
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
print("Starting BCI server...")
device = torch.device('cpu')

CLASS_NAMES = ['Left hand','Right hand','Both hands','Feet']
MENU_NAMES  = ['Basic needs','Emotions','Actions','People']
N_CHANNELS  = 64
N_TP        = 321

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),'eegnet_4class.pth')
P300_PATH  = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),'p300_fast.pkl')

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
    print("No EEGNet model found")

if os.path.exists(P300_PATH):
    p300_data = joblib.load(P300_PATH)
    lda       = p300_data['pipeline']
    print(f"P300 LDA loaded — AUC: {p300_data['auc']:.3f}")
else:
    print("Training fresh P300 model...")
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
    print("P300 model ready!")

# ══════════════════════════════════════════════════════════════
# Multilingual vocabulary
# ══════════════════════════════════════════════════════════════
LANGUAGES = {
    'English': {
        'Basic needs': ['WATER','FOOD','PAIN','TIRED',
                        'TOILET','HELP','MEDICINE','SLEEP'],
        'Emotions':    ['HAPPY','SAD','SCARED','ANGRY',
                        'LOVED','CALM','LONELY','CONFUSED'],
        'Actions':     ['YES','NO','STOP','WAIT',
                        'COME','GO','CALL','REPEAT'],
        'People':      ['DOCTOR','NURSE','MUM','DAD',
                        'FAMILY','FRIEND','ANYONE','EMERGENCY'],
    },
    'Kannada': {
        'Basic needs': ['ನೀರು','ಊಟ','ನೋವು','ಆಯಾಸ',
                        'ಶೌಚಾಲಯ','ಸಹಾಯ','ಔಷಧಿ','ನಿದ್ರೆ'],
        'Emotions':    ['ಖುಷಿ','ದುಃಖ','ಭಯ','ಕೋಪ',
                        'ಪ್ರೀತಿ','ಶಾಂತ','ಒಂಟಿ','ಗೊಂದಲ'],
        'Actions':     ['ಹೌದು','ಇಲ್ಲ','ನಿಲ್ಲು','ತಡೆ',
                        'ಬಾ','ಹೋಗು','ಕರೆ','ಮತ್ತೆ'],
        'People':      ['ವೈದ್ಯ','ದಾದಿ','ಅಮ್ಮ','ಅಪ್ಪ',
                        'ಕುಟುಂಬ','ಗೆಳೆಯ','ಯಾರಾದರೂ','ತುರ್ತು'],
    },
    'Hindi': {
        'Basic needs': ['पानी','खाना','दर्द','थका',
                        'शौचालय','मदद','दवाई','नींद'],
        'Emotions':    ['खुश','दुखी','डरा','गुस्सा',
                        'प्यार','शांत','अकेला','उलझन'],
        'Actions':     ['हाँ','नहीं','रुको','ठहरो',
                        'आओ','जाओ','बुलाओ','दोबारा'],
        'People':      ['डॉक्टर','नर्स','माँ','पिता',
                        'परिवार','दोस्त','कोई भी','आपातकाल'],
    }
}

MENUS = LANGUAGES['English']

THOUGHT_TO_MENU = {
    0:'Basic needs', 1:'Emotions',
    2:'Actions',     3:'People'
}

MENU_COLORS = {
    'Basic needs':'#534AB7', 'Emotions':'#1D9E75',
    'Actions':'#BA7517',     'People':'#D85A30',
    'Keyboard':'#00d4aa'
}

# ══════════════════════════════════════════════════════════════
# Alert system
# ══════════════════════════════════════════════════════════════
ALERT_WORDS = [
    'HELP','EMERGENCY','PAIN','CALL','DOCTOR',
    'ಸಹಾಯ','ತುರ್ತು','ನೋವು','ಕರೆ','ವೈದ್ಯ',
    'मदद','आपातकाल','दर्द','बुलाओ','डॉक्टर'
]

TWILIO_ENABLED  = False
TWILIO_SID      = "YOUR_ACCOUNT_SID"
TWILIO_TOKEN    = "YOUR_AUTH_TOKEN"
TWILIO_FROM     = "whatsapp:+14155238886"
CAREGIVER_PHONE = "whatsapp:+91XXXXXXXXXX"

def send_alert(word, sentence):
    print(f"🚨 ALERT: '{word}' detected")
    if TWILIO_ENABLED:
        try:
            from twilio.rest import Client
            client = Client(TWILIO_SID, TWILIO_TOKEN)
            client.messages.create(
                body=(f"🚨 BCI ALERT!\nWord: {word}\n"
                      f"Sentence: {' '.join(sentence)}\n"
                      f"Please respond immediately."),
                from_=TWILIO_FROM, to=CAREGIVER_PHONE)
        except Exception as e:
            print(f"Alert failed: {e}")

def check_alert(word, sentence):
    if word.upper() in [w.upper() for w in ALERT_WORDS]:
        threading.Thread(target=send_alert,
                         args=(word, sentence),
                         daemon=True).start()
        return True
    return False

# ══════════════════════════════════════════════════════════════
# Patient profiles
# ══════════════════════════════════════════════════════════════
PROFILES_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'profiles.json')

def load_profiles():
    if os.path.exists(PROFILES_FILE):
        with open(PROFILES_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {
        'profiles': {
            'Default': {
                'name':         'Default Patient',
                'language':     'English',
                'voice':        'female',
                'speed':        'normal',
                'custom_words': []
            }
        },
        'active': 'Default'
    }

def save_profiles(data):
    with open(PROFILES_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

profiles_data = load_profiles()

# ══════════════════════════════════════════════════════════════
# Session history
# ══════════════════════════════════════════════════════════════
HISTORY_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'session_log.json')

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {'sessions': []}

def save_to_history(sentence, language):
    if not sentence:
        return
    hist = load_history()
    hist['sessions'].insert(0, {
        'sentence':  ' '.join(sentence),
        'language':  language,
        'timestamp': datetime.datetime.now().strftime(
                         '%d %b %Y %I:%M %p'),
        'words':     len(sentence)
    })
    hist['sessions'] = hist['sessions'][:100]
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(hist, f, ensure_ascii=False, indent=2)

# ══════════════════════════════════════════════════════════════
# Keyboard
# ══════════════════════════════════════════════════════════════
KEYBOARD_ROWS = [
    ['A','B','C','D','E','F','G'],
    ['H','I','J','K','L','M','N'],
    ['O','P','Q','R','S','T','U'],
    ['V','W','X','Y','Z','⎵','⌫'],
]
KEYBOARD_KEYS = [k for row in KEYBOARD_ROWS for k in row]

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
    if not current_word:
        return []
    prefix = current_word.upper()
    return [w for w in WORD_LIST
            if w.startswith(prefix) and w != prefix][:3]

# ══════════════════════════════════════════════════════════════
# BCI State
# ══════════════════════════════════════════════════════════════
bci_state = {
    'mode':          'menu',
    'language':      'English',
    'detected_word': '',
    'detected_menu': '',
    'thought_class': '',
    'confidence':    0.0,
    'sentence':      [],
    'history':       [],
    'status':        'Ready',
    'processing':    False,
    'is_alert':      False,
    'current_word':  '',
    'suggestions':   [],
    'typed_words':   [],
    'voice_gender':  'female',
    'voice_speed':   'normal',
    'dashboard': {
        'total_words':     0,
        'total_sentences': 0,
        'alert_count':     0,
        'session_start':   datetime.datetime.now().strftime(
                               '%d %b %Y %I:%M %p'),
        'recent_activity': []
    }
}

def run_p300_detection(items, target_idx):
    scores = np.zeros(len(items))
    for _ in range(2):
        for idx in range(len(items)):
            epoch = generate_p300_epoch(idx == target_idx)
            feat  = extract_rich_features(
                epoch[np.newaxis])[0].reshape(1,-1)
            scores[idx] += lda.predict_proba(feat)[0][1]
    avg = scores / 2
    best_idx = int(np.argmax(avg))
    return best_idx, float(avg[best_idx])

def update_dashboard(word, is_alert=False):
    d = bci_state['dashboard']
    d['total_words'] += 1
    if is_alert:
        d['alert_count'] += 1
    d['recent_activity'].insert(0, {
        'word':     word,
        'time':     datetime.datetime.now().strftime('%I:%M %p'),
        'is_alert': is_alert
    })
    d['recent_activity'] = d['recent_activity'][:20]

def run_menu_pipeline():
    bci_state['processing'] = True
    bci_state['is_alert']   = False
    bci_state['status']     = '🧠 Reading brain signals...'

    thought_idx  = random.randint(0, 3)
    menu_name    = THOUGHT_TO_MENU[thought_idx]
    thought_name = CLASS_NAMES[thought_idx]
    bci_state['detected_menu'] = menu_name
    bci_state['thought_class'] = thought_name
    bci_state['status'] = f'Thought: {thought_name} → {menu_name}'
    time.sleep(0.3)

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
        'time':       datetime.datetime.now().strftime('%I:%M %p'),
        'mode':       'menu'
    })
    bci_state['history'] = bci_state['history'][:10]

    is_alert = check_alert(best_word, bci_state['sentence'])
    bci_state['is_alert'] = is_alert
    bci_state['status']   = (
        f'🚨 ALERT — {best_word}!' if is_alert
        else f'✓ Detected: {best_word}')

    update_dashboard(best_word, is_alert)
    bci_state['processing'] = False

def run_keyboard_pipeline():
    bci_state['processing']    = True
    bci_state['detected_menu'] = 'Keyboard'
    bci_state['status']        = '⌨️ Scanning keyboard...'

    target_idx = random.randint(0, len(KEYBOARD_KEYS)-1)
    best_idx, confidence = run_p300_detection(
        KEYBOARD_KEYS, target_idx)
    best_key = KEYBOARD_KEYS[best_idx]

    if best_key == '⎵':
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
                'time':       datetime.datetime.now().strftime(
                                  '%I:%M %p'),
                'mode':       'keyboard'
            })
            update_dashboard(word)
        bci_state['current_word']  = ''
        bci_state['suggestions']   = []
        bci_state['detected_word'] = f'[SPACE] → "{word}"'
        bci_state['status']        = f'✓ "{word}" added'
    elif best_key == '⌫':
        if bci_state['current_word']:
            bci_state['current_word'] = \
                bci_state['current_word'][:-1]
        bci_state['suggestions'] = get_suggestions(
            bci_state['current_word'])
        bci_state['detected_word'] = '[DELETE]'
        bci_state['status']        = '⌫ Letter deleted'
    else:
        bci_state['current_word'] += best_key
        bci_state['suggestions']   = get_suggestions(
            bci_state['current_word'])
        bci_state['detected_word'] = best_key
        bci_state['status'] = (
            f'✓ Letter: {best_key}  |  '
            f'Word: {bci_state["current_word"]}')
        bci_state['history'].insert(0, {
            'word':       best_key,
            'menu':       'Keyboard',
            'thought':    f'Spelling: {bci_state["current_word"]}',
            'confidence': f"{confidence:.2f}",
            'color':      '#00d4aa',
            'time':       datetime.datetime.now().strftime(
                              '%I:%M %p'),
            'mode':       'keyboard'
        })

    bci_state['history']    = bci_state['history'][:10]
    bci_state['confidence'] = round(confidence, 3)
    bci_state['processing'] = False

# ══════════════════════════════════════════════════════════════
# HTML
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
    :root{
      --bg:#0f0f1a;--card:#1a1a2e;--border:#2a2a4a;
      --text:#e0e0ff;--dim:#606080;
      --green:#00d4aa;--purple:#534AB7;
      --teal:#1D9E75;--amber:#BA7517;--red:#D85A30;
    }
    *{margin:0;padding:0;box-sizing:border-box;
      -webkit-tap-highlight-color:transparent;}
    body{background:var(--bg);color:var(--text);
      font-family:-apple-system,'Helvetica Neue',sans-serif;
      min-height:100vh;padding:12px;
      max-width:520px;margin:0 auto;}

    /* Tabs */
    .tabs{display:flex;gap:4px;margin-bottom:12px;
      background:var(--card);border-radius:10px;padding:4px;}
    .tab{flex:1;padding:8px 4px;border:none;border-radius:8px;
      font-size:11px;font-weight:700;cursor:pointer;
      background:transparent;color:var(--dim);transition:all 0.2s;}
    .tab.active{background:var(--purple);color:white;}
    .tab-content{display:none;}
    .tab-content.active{display:block;}

    /* Header */
    .header{text-align:center;padding:12px 0 8px;}
    .header h1{font-size:18px;font-weight:700;
      background:linear-gradient(135deg,#a0a0ff,#00d4aa);
      -webkit-background-clip:text;
      -webkit-text-fill-color:transparent;margin-bottom:3px;}
    .header p{font-size:11px;color:var(--dim);}

    /* Language */
    .lang-row{display:flex;gap:6px;margin-bottom:10px;}
    .lang-btn{flex:1;padding:7px;border:1px solid var(--border);
      border-radius:8px;font-size:11px;font-weight:600;
      cursor:pointer;background:var(--card);color:var(--dim);
      transition:all 0.2s;}
    .lang-btn.active{background:var(--purple);color:white;
      border-color:var(--purple);}

    /* Mode toggle */
    .mode-toggle{display:flex;gap:8px;margin-bottom:10px;}
    .mode-btn{flex:1;padding:10px;border:none;border-radius:10px;
      font-size:12px;font-weight:700;cursor:pointer;transition:all 0.2s;}
    .mode-btn.active{transform:scale(1.03);
      box-shadow:0 0 12px rgba(83,74,183,0.4);}
    .mode-menu{background:var(--purple);color:white;}
    .mode-key{background:var(--green);color:#0f0f1a;}

    /* Status */
    .status{background:var(--card);border:1px solid var(--border);
      border-radius:10px;padding:8px 14px;font-size:12px;
      color:var(--dim);text-align:center;margin-bottom:10px;
      transition:border-color 0.3s;}
    .status.alert{border-color:var(--red);color:var(--red);
      font-weight:700;}

    /* Word display */
    .word-card{background:var(--card);border:1px solid var(--border);
      border-radius:14px;padding:18px 14px 14px;
      text-align:center;margin-bottom:10px;
      transition:border-color 0.3s;}
    .word-card.alert{border-color:var(--red);}
    .menu-chip{display:inline-block;font-size:10px;font-weight:600;
      padding:2px 10px;border-radius:20px;margin-bottom:10px;}
    .big-word{font-size:44px;font-weight:800;color:var(--green);
      letter-spacing:2px;min-height:52px;margin-bottom:12px;
      transition:transform 0.2s,color 0.3s;}
    .big-word.alert{color:var(--red);}
    .conf-track{background:var(--bg);border-radius:4px;
      height:5px;overflow:hidden;margin-bottom:4px;}
    .conf-fill{height:100%;border-radius:4px;
      background:linear-gradient(90deg,var(--purple),var(--green));
      transition:width 0.5s ease;}
    .conf-label{font-size:10px;color:var(--dim);}

    /* Alert banner */
    .alert-banner{display:none;background:#2a0a0a;
      border:1px solid var(--red);border-radius:10px;
      padding:10px 14px;margin-bottom:10px;
      text-align:center;font-size:13px;font-weight:700;
      color:var(--red);}
    .alert-banner.visible{display:block;}

    /* Sentence */
    .sentence-card{background:var(--card);
      border:1px solid var(--border);border-radius:12px;
      padding:12px 14px;margin-bottom:10px;}
    .section-label{font-size:9px;font-weight:700;color:var(--dim);
      letter-spacing:1px;margin-bottom:4px;}
    .sentence-text{font-size:18px;font-weight:600;
      min-height:24px;line-height:1.4;word-wrap:break-word;}

    /* Voice */
    .voice-card{background:var(--card);
      border:1px solid var(--border);border-radius:12px;
      padding:12px 14px;margin-bottom:10px;}
    .voice-row{display:flex;gap:6px;margin-top:6px;flex-wrap:wrap;}
    .voice-btn{padding:6px 12px;border:1px solid var(--border);
      border-radius:20px;font-size:11px;font-weight:600;
      cursor:pointer;background:var(--card);color:var(--dim);
      transition:all 0.2s;}
    .voice-btn.active{background:var(--teal);color:white;
      border-color:var(--teal);}

    /* Profile */
    .profile-card{background:var(--card);
      border:1px solid var(--border);border-radius:12px;
      padding:12px 14px;margin-bottom:10px;}
    .profile-list{display:flex;flex-direction:column;gap:6px;}
    .profile-item{display:flex;align-items:center;gap:8px;
      padding:8px 10px;background:var(--bg);border-radius:8px;
      cursor:pointer;border:1px solid transparent;}
    .profile-item.active{border-color:var(--purple);}
    .profile-item-name{font-size:13px;font-weight:600;flex:1;}
    .profile-item-detail{font-size:10px;color:var(--dim);}
    .add-profile-btn{width:100%;padding:10px;
      border:1px dashed var(--border);border-radius:8px;
      background:transparent;color:var(--dim);
      font-size:12px;cursor:pointer;margin-top:6px;}

    /* Keyboard */
    .keyboard-section{display:none;}
    .keyboard-section.visible{display:block;}
    .word-builder{background:var(--card);
      border:1px solid var(--green);border-radius:12px;
      padding:12px 14px;margin-bottom:10px;}
    .wb-text{font-size:26px;font-weight:700;
      color:var(--green);min-height:34px;letter-spacing:4px;}
    .wb-cursor{display:inline-block;width:2px;height:28px;
      background:var(--green);margin-left:2px;
      animation:blink 1s step-end infinite;vertical-align:bottom;}
    @keyframes blink{0%,100%{opacity:1}50%{opacity:0}}
    .suggestions{display:flex;gap:6px;margin-bottom:10px;
      flex-wrap:wrap;}
    .suggestion-btn{background:#1a2a3a;
      border:1px solid var(--green);border-radius:20px;
      padding:5px 12px;font-size:12px;font-weight:600;
      color:var(--green);cursor:pointer;}
    .keyboard-grid{margin-bottom:10px;}
    .key-row{display:flex;gap:5px;margin-bottom:5px;
      justify-content:center;}
    .key{width:42px;height:42px;border:none;border-radius:8px;
      font-size:14px;font-weight:700;cursor:pointer;
      background:#1a1a3a;color:var(--text);
      transition:transform 0.1s;}
    .key:active{transform:scale(0.92);}
    .key.selected-key{background:var(--green);color:#0f0f1a;}
    .key.space-key{width:60px;background:var(--teal);
      color:white;font-size:10px;}
    .key.del-key{width:60px;background:var(--red);
      color:white;font-size:16px;}

    /* Buttons */
    .btn-grid{display:grid;grid-template-columns:1fr 1fr;
      gap:8px;margin-bottom:8px;}
    .btn-grid-3{display:grid;grid-template-columns:1fr 1fr 1fr;
      gap:8px;margin-bottom:10px;}
    .btn{border:none;border-radius:10px;padding:14px 8px;
      font-size:12px;font-weight:700;cursor:pointer;
      transition:transform 0.1s;display:flex;
      flex-direction:column;align-items:center;gap:3px;}
    .btn:active{transform:scale(0.96);}
    .btn-icon{font-size:18px;}
    .btn-detect{background:var(--purple);color:white;
      grid-column:1/-1;padding:18px;}
    .btn-speak{background:var(--teal);color:white;}
    .btn-repeat{background:#2a2a4a;color:#a0a0ff;}
    .btn-delete{background:var(--amber);color:white;}
    .btn-clear{background:#1a1a2e;
      border:1px solid var(--border);color:var(--dim);}
    .btn-share{background:#2a2a4a;color:#a0a0ff;}
    .btn-scan{background:var(--green);color:#0f0f1a;
      grid-column:1/-1;padding:18px;}
    .btn-add-word{background:#1a2a3a;
      border:1px solid var(--green);color:var(--green);}
    .btn-save{background:var(--teal);color:white;}

    /* Dashboard */
    .dash-grid{display:grid;grid-template-columns:1fr 1fr;
      gap:8px;margin-bottom:10px;}
    .dash-stat{background:var(--card);border-radius:10px;
      padding:12px;text-align:center;}
    .dash-num{font-size:28px;font-weight:700;color:var(--green);}
    .dash-label{font-size:10px;color:var(--dim);margin-top:2px;}
    .activity-item{display:flex;align-items:center;gap:8px;
      padding:6px 0;border-bottom:1px solid var(--card);}
    .activity-item:last-child{border-bottom:none;}
    .activity-word{font-weight:700;flex:1;}
    .activity-time{color:var(--dim);font-size:10px;}

    /* History */
    .history-entry{background:var(--card);border-radius:8px;
      padding:10px 12px;margin-bottom:6px;}
    .history-sentence{font-size:14px;font-weight:600;
      color:var(--text);margin-bottom:3px;}
    .history-meta{font-size:10px;color:var(--dim);}

    .hist-item{display:flex;align-items:center;gap:8px;
      padding:5px 0;border-bottom:1px solid #0f0f1a;}
    .hist-item:last-child{border-bottom:none;}
    .hist-dot{width:7px;height:7px;border-radius:50%;flex-shrink:0;}
    .hist-word{font-size:13px;font-weight:700;flex:1;}
    .hist-meta{color:var(--dim);font-size:10px;}

    @keyframes spin{to{transform:rotate(360deg);}}
    .spinner{display:inline-block;width:14px;height:14px;
      border:2px solid rgba(255,255,255,0.2);
      border-top-color:white;border-radius:50%;
      animation:spin 0.8s linear infinite;
      margin-right:5px;vertical-align:middle;}

    .online-badge{display:inline-flex;align-items:center;
      gap:4px;background:#0a2a1a;border:1px solid var(--teal);
      border-radius:20px;padding:2px 8px;
      font-size:10px;color:var(--teal);margin-bottom:10px;}
    .online-dot{width:5px;height:5px;border-radius:50%;
      background:var(--teal);
      animation:pulse 2s ease-in-out infinite;}
    @keyframes pulse{0%,100%{opacity:1}50%{opacity:0.3}}
  </style>
</head>
<body>

<div class="header">
  <h1>🧠 BCI Thought-to-Speech</h1>
  <p>Brain signals → Words → Voice</p>
</div>

<div style="text-align:center;margin-bottom:10px">
  <span class="online-badge">
    <span class="online-dot"></span>AI running
  </span>
</div>

<!-- Main tabs -->
<div class="tabs">
  <button class="tab active" onclick="showTab('device',this)">
    🧠 Device
  </button>
  <button class="tab" onclick="showTab('caregiver',this)">
    👨‍⚕️ Caregiver
  </button>
  <button class="tab" onclick="showTab('history',this)">
    📋 History
  </button>
  <button class="tab" onclick="showTab('settings',this)">
    ⚙️ Settings
  </button>
</div>

<!-- ══ TAB 1: DEVICE ══ -->
<div class="tab-content active" id="tab-device">

  <div class="lang-row">
    <button class="lang-btn active" id="lang-en"
            onclick="setLang('English')">🇬🇧 EN</button>
    <button class="lang-btn" id="lang-kn"
            onclick="setLang('Kannada')">🇮🇳 ಕನ್ನಡ</button>
    <button class="lang-btn" id="lang-hi"
            onclick="setLang('Hindi')">🇮🇳 हिंदी</button>
  </div>

  <div class="mode-toggle">
    <button class="mode-btn mode-menu active" id="btn-mode-menu"
            onclick="switchMode('menu')">
      📋 Word Menus
    </button>
    <button class="mode-btn mode-key" id="btn-mode-key"
            onclick="switchMode('keyboard')">
      ⌨️ Spell Word
    </button>
  </div>

  <div class="status" id="status">Ready — tap Detect Thought</div>
  <div class="alert-banner" id="alert-banner">
    🚨 EMERGENCY ALERT SENT 🚨
  </div>

  <div class="word-card" id="word-card">
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

  <!-- Keyboard -->
  <div class="keyboard-section" id="keyboard-section">
    <div class="word-builder">
      <div class="section-label">SPELLING</div>
      <div class="wb-text">
        <span id="wb-letters"></span>
        <span class="wb-cursor"></span>
      </div>
    </div>
    <div class="suggestions" id="suggestions"></div>
    <div class="keyboard-grid" id="keyboard-grid"></div>
    <div style="font-size:9px;color:var(--dim);
                text-align:center;margin-bottom:8px">
      Tap Scan to detect focused letter via P300
    </div>
    <div class="btn-grid" style="margin-bottom:8px">
      <button class="btn btn-scan" id="scan-btn"
              onclick="scanKeyboard()">
        <span class="btn-icon">👁️</span>Scan Keyboard
      </button>
    </div>
    <div class="btn-grid">
      <button class="btn btn-add-word"
              onclick="addWordToSentence()">
        <span class="btn-icon">✅</span>Add Word
      </button>
      <button class="btn btn-delete" onclick="deleteLastLetter()">
        <span class="btn-icon">⌫</span>Delete Letter
      </button>
    </div>
  </div>

  <div class="sentence-card">
    <div class="section-label">SENTENCE</div>
    <div class="sentence-text" id="sentence-text">
      Words appear here...
    </div>
  </div>

  <div id="menu-buttons">
    <div class="btn-grid" style="margin-bottom:8px">
      <button class="btn btn-detect" id="detect-btn"
              onclick="detectThought()">
        <span class="btn-icon">🧠</span>Detect Thought
      </button>
    </div>
    <div class="btn-grid">
      <button class="btn btn-speak" onclick="speakSentence()">
        <span class="btn-icon">🔊</span>Speak
      </button>
      <button class="btn btn-repeat" onclick="repeatWord()">
        <span class="btn-icon">🔁</span>Repeat
      </button>
    </div>
  </div>

  <div class="btn-grid-3" style="margin-top:8px">
    <button class="btn btn-save" onclick="saveSentence()">
      <span class="btn-icon">💾</span>Save
    </button>
    <button class="btn btn-delete" onclick="deleteLast()">
      <span class="btn-icon">⬅️</span>Delete
    </button>
    <button class="btn btn-clear" onclick="clearAll()">
      <span class="btn-icon">🗑️</span>Clear
    </button>
  </div>

  <div class="sentence-card" style="margin-top:8px">
    <div class="section-label">DETECTION HISTORY</div>
    <div id="history-list">
      <div style="color:var(--dim);font-size:11px">
        No detections yet
      </div>
    </div>
  </div>

</div>

<!-- ══ TAB 2: CAREGIVER ══ -->
<div class="tab-content" id="tab-caregiver">
  <div style="padding:8px 0">
    <div class="section-label" style="margin-bottom:8px">
      LIVE PATIENT ACTIVITY
    </div>
    <div class="dash-grid">
      <div class="dash-stat">
        <div class="dash-num" id="dash-words">0</div>
        <div class="dash-label">Words detected</div>
      </div>
      <div class="dash-stat">
        <div class="dash-num" id="dash-sentences">0</div>
        <div class="dash-label">Sentences saved</div>
      </div>
      <div class="dash-stat">
        <div class="dash-num" id="dash-alerts"
             style="color:var(--red)">0</div>
        <div class="dash-label">🚨 Alerts</div>
      </div>
      <div class="dash-stat">
        <div class="dash-num" id="dash-session"
             style="font-size:13px;margin-top:6px">--:--</div>
        <div class="dash-label">Session start</div>
      </div>
    </div>
    <div class="sentence-card">
      <div class="section-label">CURRENT SENTENCE (LIVE)</div>
      <div class="sentence-text" id="dash-current">
        No words yet...
      </div>
    </div>
    <div class="sentence-card">
      <div class="section-label">RECENT ACTIVITY</div>
      <div id="dash-activity">
        <div style="color:var(--dim);font-size:11px">
          No activity yet
        </div>
      </div>
    </div>
    <button class="btn btn-speak"
            style="width:100%;margin-top:4px"
            onclick="speakSentence()">
      <span class="btn-icon">🔊</span>
      Speak Current Sentence
    </button>
  </div>
</div>

<!-- ══ TAB 3: HISTORY ══ -->
<div class="tab-content" id="tab-history">
  <div style="padding:8px 0">
    <div class="section-label" style="margin-bottom:8px">
      SAVED SENTENCES
    </div>
    <div id="history-log">
      <div style="color:var(--dim);font-size:11px">
        No saved sentences yet — tap 💾 Save on Device tab
      </div>
    </div>
    <button class="btn btn-clear"
            style="width:100%;margin-top:10px"
            onclick="clearHistory()">
      Clear History
    </button>
  </div>
</div>

<!-- ══ TAB 4: SETTINGS ══ -->
<div class="tab-content" id="tab-settings">
  <div style="padding:8px 0">

    <div class="voice-card">
      <div class="section-label">VOICE SETTINGS</div>
      <div style="font-size:12px;color:var(--dim);margin-top:4px">
        Gender
      </div>
      <div class="voice-row">
        <button class="voice-btn active" id="voice-female"
                onclick="setVoice('gender','female')">
          👩 Female
        </button>
        <button class="voice-btn" id="voice-male"
                onclick="setVoice('gender','male')">
          👨 Male
        </button>
      </div>
      <div style="font-size:12px;color:var(--dim);margin-top:8px">
        Speed
      </div>
      <div class="voice-row">
        <button class="voice-btn" id="speed-slow"
                onclick="setVoice('speed','slow')">
          🐢 Slow
        </button>
        <button class="voice-btn active" id="speed-normal"
                onclick="setVoice('speed','normal')">
          ▶️ Normal
        </button>
        <button class="voice-btn" id="speed-fast"
                onclick="setVoice('speed','fast')">
          ⚡ Fast
        </button>
      </div>
      <div style="font-size:11px;color:var(--dim);
                  margin-top:10px;line-height:1.6">
        ℹ️ Kannada words are spoken using romanized
        pronunciation (e.g. ನೀರು → "neeru") for better
        clarity on all devices.
      </div>
    </div>

    <div class="profile-card">
      <div class="section-label">PATIENT PROFILES</div>
      <div style="font-size:11px;color:var(--dim);
                  margin:4px 0 8px">
        Each patient has their own language, voice and
        custom word list
      </div>
      <div class="profile-list" id="profile-list"></div>
      <button class="add-profile-btn" onclick="addProfile()">
        + Add New Patient Profile
      </button>
    </div>

    <div class="voice-card">
      <div class="section-label">CUSTOM WORDS</div>
      <div style="font-size:11px;color:var(--dim);
                  margin:4px 0 8px">
        Add words specific to this patient
      </div>
      <div id="custom-words-list"
           style="min-height:40px;margin-bottom:8px"></div>
      <div style="display:flex;gap:6px">
        <input id="new-word-input" type="text"
               placeholder="Type a word..."
               style="flex:1;padding:8px 10px;
                      background:var(--bg);color:var(--text);
                      border:1px solid var(--border);
                      border-radius:8px;font-size:12px;
                      outline:none;">
        <button class="btn btn-save"
                style="padding:8px 14px;font-size:12px"
                onclick="addCustomWord()">Add</button>
      </div>
    </div>

    <div class="voice-card">
      <div class="section-label">SYSTEM INFO</div>
      <div style="font-size:11px;color:var(--dim);
                  line-height:1.8;margin-top:4px">
        GitHub: github.com/shishir442/bci-thought-to-speech<br>
        Model: EEGNet 4-class (74 subjects, 59.46%)<br>
        P300: LDA classifier (AUC=1.000, 1 sec)<br>
        Languages: English · ಕನ್ನಡ · हिंदी<br>
        Vocabulary: 32 words + full A-Z keyboard
      </div>
    </div>

  </div>
</div>

<script>
const synth = window.speechSynthesis;
let lastWord    = '';
let curMode     = 'menu';
let curLang     = 'English';
let voiceGender = 'female';
let voiceSpeed  = 1.0;

const MENU_COLORS = {
  'Basic needs':'#534AB7','Emotions':'#1D9E75',
  'Actions':'#BA7517','People':'#D85A30','Keyboard':'#00d4aa'
};

// ── Kannada romanized pronunciation map ────────────────────
const KN_SPEAK = {
  'ನೀರು':'neeru',      'ಊಟ':'oota',
  'ನೋವು':'novu',       'ಆಯಾಸ':'aayaasa',
  'ಶೌಚಾಲಯ':'shouchalaya','ಸಹಾಯ':'sahaaya',
  'ಔಷಧಿ':'aushadhi',   'ನಿದ್ರೆ':'nidre',
  'ಖುಷಿ':'khushi',     'ದುಃಖ':'dukha',
  'ಭಯ':'bhaya',        'ಕೋಪ':'kopa',
  'ಪ್ರೀತಿ':'preeti',   'ಶಾಂತ':'shaanta',
  'ಒಂಟಿ':'onti',       'ಗೊಂದಲ':'gondala',
  'ಹೌದು':'haudu',      'ಇಲ್ಲ':'illa',
  'ನಿಲ್ಲು':'nillu',    'ತಡೆ':'tade',
  'ಬಾ':'baa',          'ಹೋಗು':'hoogu',
  'ಕರೆ':'kare',        'ಮತ್ತೆ':'matte',
  'ವೈದ್ಯ':'vaidya',    'ದಾದಿ':'daadi',
  'ಅಮ್ಮ':'amma',       'ಅಪ್ಪ':'appa',
  'ಕುಟುಂಬ':'kutumba',  'ಗೆಳೆಯ':'geleya',
  'ಯಾರಾದರೂ':'yaaradaroo','ತುರ್ತು':'turtu'
};

// ── Tabs ───────────────────────────────────────────────────
function showTab(tab, el) {
  document.querySelectorAll('.tab-content').forEach(t =>
    t.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(t =>
    t.classList.remove('active'));
  document.getElementById(`tab-${tab}`).classList.add('active');
  el.classList.add('active');
  if (tab==='history') loadHistoryLog();
  if (tab==='settings') loadProfiles();
}

// ── Keyboard ───────────────────────────────────────────────
const ROWS = [
  ['A','B','C','D','E','F','G'],
  ['H','I','J','K','L','M','N'],
  ['O','P','Q','R','S','T','U'],
  ['V','W','X','Y','Z','⎵','⌫'],
];
function buildKeyboard() {
  const grid = document.getElementById('keyboard-grid');
  grid.innerHTML = '';
  ROWS.forEach(row => {
    const rowDiv = document.createElement('div');
    rowDiv.className = 'key-row';
    row.forEach(k => {
      const btn = document.createElement('button');
      btn.className = 'key' +
        (k==='⎵'?' space-key':'')+(k==='⌫'?' del-key':'');
      btn.id = `key-${k}`;
      btn.textContent = k==='⎵' ? 'SPC' : k;
      btn.onclick = () => manualKeyPress(k);
      rowDiv.appendChild(btn);
    });
    grid.appendChild(rowDiv);
  });
}
buildKeyboard();

// ── Language ───────────────────────────────────────────────
function setLang(lang) {
  curLang = lang;
  ['en','kn','hi'].forEach(l =>
    document.getElementById(`lang-${l}`).classList.remove('active'));
  const m = {'English':'en','Kannada':'kn','Hindi':'hi'};
  document.getElementById(`lang-${m[lang]}`).classList.add('active');
  fetch('/set_language',{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({language:lang})})
  .then(r=>r.json()).then(data=>{
    updateUI(data);
    document.getElementById('status').textContent =
      `Language: ${lang}`;
  });
}

// ── Mode ───────────────────────────────────────────────────
function switchMode(mode) {
  curMode = mode;
  document.getElementById('btn-mode-menu')
    .classList.toggle('active', mode==='menu');
  document.getElementById('btn-mode-key')
    .classList.toggle('active', mode==='keyboard');
  document.getElementById('keyboard-section')
    .classList.toggle('visible', mode==='keyboard');
  document.getElementById('menu-buttons').style.display =
    mode==='menu' ? 'block' : 'none';
}

// ── Voice ──────────────────────────────────────────────────
function setVoice(type, value) {
  if (type==='gender') {
    voiceGender = value;
    document.querySelectorAll('[id^="voice-"]').forEach(b=>
      b.classList.remove('active'));
    document.getElementById(`voice-${value}`)
      .classList.add('active');
  } else {
    voiceSpeed = value==='slow'?0.6:value==='fast'?1.4:1.0;
    document.querySelectorAll('[id^="speed-"]').forEach(b=>
      b.classList.remove('active'));
    document.getElementById(`speed-${value}`)
      .classList.add('active');
  }
}

// ── SPEAK — fixed for Kannada ──────────────────────────────
function speak(text) {
  if (!text || text.startsWith('[')) return;
  synth.cancel();

  const utt = new SpeechSynthesisUtterance();
  utt.volume = 1.0;
  utt.rate   = voiceSpeed;

  const voices = synth.getVoices();

  if (curLang === 'Kannada') {
    // Look up romanized pronunciation for each word
    const words = text.trim().split(/\s+/);
    const spoken = words.map(w => {
      const lookup = KN_SPEAK[w.trim()];
      return lookup ? lookup : w;
    }).join(' ');

    utt.text = spoken;
    utt.lang = 'en-IN';

    // Prefer Indian English voice
    const indVoice = voices.find(v =>
      v.lang === 'en-IN' ||
      v.lang.startsWith('en-IN') ||
      v.name.toLowerCase().includes('india'));
    if (indVoice) utt.voice = indVoice;

  } else if (curLang === 'Hindi') {
    utt.text = text;
    // Try to find Hindi voice
    const hiVoice = voices.find(v =>
      v.lang.startsWith('hi') ||
      v.lang === 'hi-IN');
    if (hiVoice) {
      utt.voice = hiVoice;
      utt.lang  = 'hi-IN';
    } else {
      // Fallback to Indian English
      utt.lang = 'en-IN';
      const indVoice = voices.find(v =>
        v.lang === 'en-IN' ||
        v.name.toLowerCase().includes('india'));
      if (indVoice) utt.voice = indVoice;
    }

  } else {
    // English
    utt.text = text;
    utt.lang = 'en-US';
    const enVoice = voices.find(v =>
      voiceGender === 'female'
        ? v.name.includes('Female') ||
          v.name.includes('Samantha') ||
          v.name.includes('Google UK English Female') ||
          (v.lang.startsWith('en') && v.name.includes('f'))
        : v.name.includes('Male') ||
          v.name.includes('Alex') ||
          v.name.includes('Google UK English Male'));
    if (enVoice) utt.voice = enVoice;
  }

  synth.speak(utt);
}

// ── Update UI ──────────────────────────────────────────────
function updateUI(data) {
  const statusEl = document.getElementById('status');
  statusEl.textContent = data.status;
  statusEl.className = 'status'+(data.is_alert?' alert':'');

  document.getElementById('alert-banner').className =
    'alert-banner'+(data.is_alert?' visible':'');

  document.getElementById('word-card').className =
    'word-card'+(data.is_alert?' alert':'');

  const chip  = document.getElementById('menu-chip');
  const color = MENU_COLORS[data.detected_menu]||'#2a2a4a';
  chip.textContent      = data.detected_menu||'No menu open';
  chip.style.background = color+'22';
  chip.style.color      = color||'#8080b0';
  chip.style.border     = `1px solid ${color}44`;

  const wordEl = document.getElementById('big-word');
  if (data.detected_word &&
      data.detected_word!==wordEl.textContent){
    wordEl.style.transform='scale(1.12)';
    setTimeout(()=>wordEl.style.transform='scale(1)',250);
  }
  wordEl.textContent = data.detected_word||'---';
  wordEl.className   = 'big-word'+(data.is_alert?' alert':'');
  if(!data.is_alert)
    wordEl.style.color=data.detected_word?'#00d4aa':'#404060';
  lastWord = data.detected_word||lastWord;

  const conf=Math.round(data.confidence*100);
  document.getElementById('conf-fill').style.width=conf+'%';
  document.getElementById('conf-label').textContent =
    conf>0?`Confidence: ${conf}%`:'Confidence: ---';

  const sent=data.sentence.join('  ');
  document.getElementById('sentence-text').textContent =
    sent||'Words appear here...';

  // Dashboard
  if (data.dashboard) {
    const d=data.dashboard;
    document.getElementById('dash-words').textContent =
      d.total_words;
    document.getElementById('dash-sentences').textContent =
      d.total_sentences;
    document.getElementById('dash-alerts').textContent =
      d.alert_count;
    document.getElementById('dash-session').textContent =
      d.session_start.split(' ').slice(-2).join(' ');
    document.getElementById('dash-current').textContent =
      sent||'No words yet...';
    const actEl=document.getElementById('dash-activity');
    if(d.recent_activity&&d.recent_activity.length>0){
      actEl.innerHTML=d.recent_activity.slice(0,8).map(a=>`
        <div class="activity-item">
          <div class="activity-word"
               style="color:${a.is_alert?'var(--red)':'var(--green)'}">
            ${a.is_alert?'🚨 ':''}${a.word}</div>
          <div class="activity-time">${a.time}</div>
        </div>`).join('');
    } else {
      actEl.innerHTML=
        '<div style="color:var(--dim);font-size:11px">' +
        'No activity yet</div>';
    }
  }

  // Keyboard
  document.getElementById('wb-letters').textContent =
    data.current_word||'';

  const sugDiv=document.getElementById('suggestions');
  sugDiv.innerHTML='';
  if(data.suggestions&&data.suggestions.length>0){
    data.suggestions.forEach(s=>{
      const btn=document.createElement('button');
      btn.className='suggestion-btn';
      btn.textContent=s;
      btn.onclick=()=>acceptSuggestion(s);
      sugDiv.appendChild(btn);
    });
  }

  // History
  let html='';
  if(data.history&&data.history.length>0){
    data.history.slice(0,6).forEach(h=>{
      html+=`<div class="hist-item">
        <div class="hist-dot" style="background:${h.color}"></div>
        <div class="hist-word" style="color:${h.color}">
          ${h.word}</div>
        <div class="hist-meta">${h.thought}</div>
        <div class="hist-meta">${h.time||''}</div>
      </div>`;
    });
  } else {
    html='<div style="color:var(--dim);font-size:11px">' +
         'No detections yet</div>';
  }
  document.getElementById('history-list').innerHTML=html;
}

// ── Detect ─────────────────────────────────────────────────
function setDetectBtn(loading){
  const btn=document.getElementById('detect-btn');
  if(btn){
    btn.innerHTML=loading
      ?'<span class="spinner"></span> Reading...'
      :'<span class="btn-icon">🧠</span>Detect Thought';
    btn.disabled=loading;
    btn.style.opacity=loading?'0.7':'1';
  }
}

function detectThought(){
  setDetectBtn(true);
  fetch('/detect',{method:'POST'})
    .then(r=>r.json())
    .then(data=>{
      setDetectBtn(false);
      updateUI(data);
      if(data.detected_word&&!data.detected_word.startsWith('['))
        speak(data.detected_word);
      if(data.is_alert)
        navigator.vibrate&&navigator.vibrate([500,200,500,200,500]);
    })
    .catch(()=>setDetectBtn(false));
}

// ── Keyboard ───────────────────────────────────────────────
function setScanBtn(loading){
  const btn=document.getElementById('scan-btn');
  if(btn){
    btn.innerHTML=loading
      ?'<span class="spinner"></span> Scanning...'
      :'<span class="btn-icon">👁️</span>Scan Keyboard';
    btn.disabled=loading;
    btn.style.opacity=loading?'0.7':'1';
  }
}

function scanKeyboard(){
  setScanBtn(true);
  fetch('/keyboard_scan',{method:'POST'})
    .then(r=>r.json())
    .then(data=>{
      setScanBtn(false);
      updateUI(data);
      flashKey(data.detected_word);
      if(data.detected_word&&data.detected_word.length===1)
        speak(data.detected_word.toLowerCase());
    })
    .catch(()=>setScanBtn(false));
}

function flashKey(key){
  document.querySelectorAll('.key').forEach(k=>
    k.classList.remove('selected-key'));
  if(key&&key.length===1){
    const el=document.getElementById(`key-${key}`);
    if(el){
      el.classList.add('selected-key');
      setTimeout(()=>el.classList.remove('selected-key'),1500);
    }
  }
}

function manualKeyPress(key){
  fetch('/manual_key',{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({key:key})})
  .then(r=>r.json()).then(data=>{
    updateUI(data);
    if(key.length===1) speak(key.toLowerCase());
  });
}

function addWordToSentence(){
  fetch('/add_word',{method:'POST'})
    .then(r=>r.json()).then(data=>{
      updateUI(data);
      if(data.typed_words&&data.typed_words.length>0)
        speak(data.typed_words[data.typed_words.length-1]);
    });
}

function deleteLastLetter(){
  fetch('/delete_letter',{method:'POST'})
    .then(r=>r.json()).then(data=>updateUI(data));
}

function acceptSuggestion(word){
  fetch('/accept_suggestion',{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({word:word})})
  .then(r=>r.json()).then(data=>{
    updateUI(data); speak(word.toLowerCase());
  });
}

// ── Common ─────────────────────────────────────────────────
function speakSentence(){
  fetch('/state').then(r=>r.json()).then(data=>{
    const s=data.sentence.join(' ');
    if(s.trim()) speak(s);
  });
}

function repeatWord(){
  if(lastWord&&!lastWord.startsWith('['))
    speak(lastWord);
}

function saveSentence(){
  fetch('/save_sentence',{method:'POST'})
    .then(r=>r.json()).then(data=>{
      updateUI(data);
      document.getElementById('status').textContent=
        '💾 Sentence saved!';
    });
}

function deleteLast(){
  fetch('/delete',{method:'POST'})
    .then(r=>r.json()).then(data=>updateUI(data));
}

function clearAll(){
  fetch('/clear',{method:'POST'})
    .then(r=>r.json()).then(data=>{
      updateUI(data); lastWord='';
    });
}

// ── Profiles ───────────────────────────────────────────────
function loadProfiles(){
  fetch('/profiles').then(r=>r.json()).then(data=>{
    const list=document.getElementById('profile-list');
    list.innerHTML='';
    Object.entries(data.profiles).forEach(([key,p])=>{
      const item=document.createElement('div');
      item.className='profile-item'+(key===data.active?' active':'');
      item.innerHTML=`
        <div>
          <div class="profile-item-name">${p.name}</div>
          <div class="profile-item-detail">
            ${p.language} · ${p.voice} voice · ${p.speed} speed
          </div>
        </div>`;
      item.onclick=()=>switchProfile(key);
      list.appendChild(item);
    });
    const active=data.profiles[data.active];
    const cwList=document.getElementById('custom-words-list');
    cwList.innerHTML='';
    if(active.custom_words&&active.custom_words.length>0){
      active.custom_words.forEach(w=>{
        const tag=document.createElement('span');
        tag.style.cssText=
          'display:inline-block;background:#1a2a3a;'+
          'border:1px solid var(--teal);border-radius:20px;'+
          'padding:3px 10px;margin:2px;font-size:11px;'+
          'color:var(--teal);cursor:pointer;';
        tag.textContent=w+' ×';
        tag.onclick=()=>removeCustomWord(w);
        cwList.appendChild(tag);
      });
    } else {
      cwList.innerHTML=
        '<span style="color:var(--dim);font-size:11px">' +
        'No custom words yet</span>';
    }
  });
}

function switchProfile(key){
  fetch('/switch_profile',{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({profile:key})})
  .then(r=>r.json()).then(()=>loadProfiles());
}

function addProfile(){
  const name=prompt('Patient name:');
  if(!name) return;
  fetch('/add_profile',{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({name:name})})
  .then(r=>r.json()).then(()=>loadProfiles());
}

function addCustomWord(){
  const input=document.getElementById('new-word-input');
  const word=input.value.trim().toUpperCase();
  if(!word) return;
  fetch('/add_custom_word',{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({word:word})})
  .then(r=>r.json()).then(()=>{input.value='';loadProfiles();});
}

function removeCustomWord(word){
  fetch('/remove_custom_word',{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({word:word})})
  .then(r=>r.json()).then(()=>loadProfiles());
}

// ── History log ────────────────────────────────────────────
function loadHistoryLog(){
  fetch('/session_history').then(r=>r.json()).then(data=>{
    const el=document.getElementById('history-log');
    if(!data.sessions||data.sessions.length===0){
      el.innerHTML=
        '<div style="color:var(--dim);font-size:11px">' +
        'No saved sentences yet</div>';
      return;
    }
    el.innerHTML=data.sessions.map(s=>`
      <div class="history-entry">
        <div class="history-sentence">${s.sentence}</div>
        <div class="history-meta">
          ${s.timestamp} · ${s.language} · ${s.words} words
        </div>
      </div>`).join('');
  });
}

function clearHistory(){
  if(!confirm('Clear all saved sentences?')) return;
  fetch('/clear_history',{method:'POST'})
    .then(()=>loadHistoryLog());
}

// Auto refresh
setInterval(()=>{
  fetch('/state').then(r=>r.json())
    .then(data=>updateUI(data)).catch(()=>{});
},3000);
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

@app.route('/set_language', methods=['POST'])
def set_language():
    global MENUS
    data = request.get_json()
    lang = data.get('language','English')
    if lang in LANGUAGES:
        MENUS = LANGUAGES[lang]
        bci_state['language'] = lang
        bci_state['status']   = f'Language: {lang}'
    return jsonify(bci_state)

@app.route('/manual_key', methods=['POST'])
def manual_key():
    data = request.get_json()
    key  = data.get('key','')
    if key == '⎵':
        word = bci_state['current_word'].strip()
        if word:
            bci_state['sentence'].append(word)
            bci_state['typed_words'].append(word)
            update_dashboard(word)
        bci_state['current_word']  = ''
        bci_state['suggestions']   = []
        bci_state['detected_word'] = '[SPACE]'
        bci_state['status'] = f'"{word}" added'
    elif key == '⌫':
        if bci_state['current_word']:
            bci_state['current_word'] = \
                bci_state['current_word'][:-1]
        bci_state['suggestions'] = get_suggestions(
            bci_state['current_word'])
        bci_state['detected_word'] = '[DELETE]'
    else:
        bci_state['current_word'] += key
        bci_state['suggestions']   = get_suggestions(
            bci_state['current_word'])
        bci_state['detected_word'] = key
        bci_state['status'] = (
            f'Letter: {key} | Word: {bci_state["current_word"]}')
    return jsonify(bci_state)

@app.route('/add_word', methods=['POST'])
def add_word():
    word = bci_state['current_word'].strip()
    if word:
        bci_state['sentence'].append(word)
        bci_state['typed_words'].append(word)
        bci_state['history'].insert(0,{
            'word':       word,
            'menu':       'Keyboard',
            'thought':    'Spelled',
            'confidence': '1.00',
            'color':      '#00d4aa',
            'time':       datetime.datetime.now().strftime('%I:%M %p'),
            'mode':       'keyboard'
        })
        update_dashboard(word)
        bci_state['status'] = f'✓ "{word}" added'
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
    word = data.get('word','').upper()
    bci_state['sentence'].append(word)
    bci_state['typed_words'].append(word)
    bci_state['current_word']  = ''
    bci_state['suggestions']   = []
    bci_state['detected_word'] = word
    bci_state['status'] = f'✓ "{word}" added'
    bci_state['history'].insert(0,{
        'word':       word,
        'menu':       'Keyboard',
        'thought':    'Autocomplete',
        'confidence': '1.00',
        'color':      '#00d4aa',
        'time':       datetime.datetime.now().strftime('%I:%M %p'),
        'mode':       'keyboard'
    })
    update_dashboard(word)
    return jsonify(bci_state)

@app.route('/save_sentence', methods=['POST'])
def save_sentence():
    if bci_state['sentence']:
        save_to_history(bci_state['sentence'],
                        bci_state['language'])
        bci_state['dashboard']['total_sentences'] += 1
        bci_state['status'] = '💾 Sentence saved!'
    return jsonify(bci_state)

@app.route('/session_history')
def session_history():
    return jsonify(load_history())

@app.route('/clear_history', methods=['POST'])
def clear_history():
    with open(HISTORY_FILE,'w') as f:
        json.dump({'sessions':[]}, f)
    return jsonify({'ok':True})

@app.route('/profiles')
def get_profiles():
    return jsonify(load_profiles())

@app.route('/switch_profile', methods=['POST'])
def switch_profile():
    data = request.get_json()
    key  = data.get('profile','Default')
    pd   = load_profiles()
    if key in pd['profiles']:
        pd['active'] = key
        save_profiles(pd)
    return jsonify({'ok':True})

@app.route('/add_profile', methods=['POST'])
def add_profile():
    data = request.get_json()
    name = data.get('name','Patient')
    pd   = load_profiles()
    key  = name.replace(' ','_')
    pd['profiles'][key] = {
        'name':         name,
        'language':     'English',
        'voice':        'female',
        'speed':        'normal',
        'custom_words': []
    }
    pd['active'] = key
    save_profiles(pd)
    return jsonify({'ok':True})

@app.route('/add_custom_word', methods=['POST'])
def add_custom_word():
    data = request.get_json()
    word = data.get('word','').upper()
    pd   = load_profiles()
    active = pd['active']
    if word and word not in \
            pd['profiles'][active]['custom_words']:
        pd['profiles'][active]['custom_words'].append(word)
        WORD_LIST.append(word)
        save_profiles(pd)
    return jsonify({'ok':True})

@app.route('/remove_custom_word', methods=['POST'])
def remove_custom_word():
    data = request.get_json()
    word = data.get('word','')
    pd   = load_profiles()
    active = pd['active']
    if word in pd['profiles'][active]['custom_words']:
        pd['profiles'][active]['custom_words'].remove(word)
        save_profiles(pd)
    return jsonify({'ok':True})

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
    bci_state['is_alert']      = False
    bci_state['status']        = 'Ready — tap Detect Thought'
    return jsonify(bci_state)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"\nBCI server running on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
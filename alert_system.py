"""
Caregiver Alert System
Sends instant alerts when HELP or EMERGENCY is detected
"""

# ── Twilio WhatsApp setup ─────────────────────────────────────
# 1. Go to https://www.twilio.com and create FREE account
# 2. Get your Account SID and Auth Token from dashboard
# 3. Join Twilio WhatsApp sandbox:
#    Send "join <your-sandbox-word>" to +14155238886 on WhatsApp
# 4. Fill in the values below

TWILIO_ENABLED    = False   # Set True after Twilio setup
TWILIO_SID        = "YOUR_ACCOUNT_SID"
TWILIO_TOKEN      = "YOUR_AUTH_TOKEN"
TWILIO_FROM       = "whatsapp:+14155238886"  # Twilio sandbox
CAREGIVER_PHONE   = "whatsapp:+91XXXXXXXXXX" # caregiver number

# Alert words that trigger emergency notification
ALERT_WORDS = ['HELP', 'EMERGENCY', 'PAIN', 'CALL', 'DOCTOR']

def send_whatsapp_alert(word, sentence):
    """Send WhatsApp message to caregiver"""
    if not TWILIO_ENABLED:
        print(f"[ALERT SIMULATION] Would send WhatsApp: '{word}' detected")
        return True
    try:
        from twilio.rest import Client
        client  = Client(TWILIO_SID, TWILIO_TOKEN)
        message = client.messages.create(
            body=(
                f"🚨 BCI ALERT — Patient needs attention!\n\n"
                f"Word detected: {word}\n"
                f"Full sentence: {' '.join(sentence)}\n\n"
                f"Please respond immediately."
            ),
            from_=TWILIO_FROM,
            to=CAREGIVER_PHONE
        )
        print(f"WhatsApp alert sent! SID: {message.sid}")
        return True
    except Exception as e:
        print(f"WhatsApp alert failed: {e}")
        return False

def check_and_alert(word, sentence):
    """Check if word needs alert and send if yes"""
    if word.upper() in ALERT_WORDS:
        print(f"ALERT TRIGGERED: {word}")
        send_whatsapp_alert(word, sentence)
        return True
    return False
"""
Hardware Integration Code
Replace simulation with real EEG headset

Supported devices:
  - OpenBCI Cyton     (~₹16,000)
  - Emotiv EPOC X     (~₹40,000)
  - Muse 2            (~₹20,000)
  - Any brainflow-compatible device
"""

print("Hardware Integration — Real EEG Setup")
print("=" * 50)

# ══════════════════════════════════════════════════════════════
# INSTALL FIRST:
# pip install brainflow
# ══════════════════════════════════════════════════════════════

import numpy as np
import time

# ── Choose your device ────────────────────────────────────────
DEVICE = 'SIMULATE'   # Change to 'OPENBCI', 'EMOTIV', or 'MUSE'

# ══════════════════════════════════════════════════════════════
# REAL HARDWARE READER
# ══════════════════════════════════════════════════════════════
class BCIHardwareReader:
    """
    Universal EEG reader supporting multiple devices.
    Uses BrainFlow library which works with 50+ EEG devices.
    """

    def __init__(self, device='SIMULATE', port=None):
        self.device = device
        self.port   = port
        self.board  = None
        self.sfreq  = 256
        self.n_channels = 8

        if device != 'SIMULATE':
            self._connect_hardware()
        else:
            print("    Running in SIMULATION mode")
            print("    Replace DEVICE = 'SIMULATE' with your device")

    def _connect_hardware(self):
        """Connect to real EEG headset via BrainFlow"""
        try:
            from brainflow.board_shim import (
                BoardShim, BrainFlowInputParams, BoardIds)
            from brainflow.data_filter import DataFilter

            params = BrainFlowInputParams()

            if self.device == 'OPENBCI':
                # OpenBCI Cyton — connect via USB dongle
                # port = 'COM3' on Windows (check Device Manager)
                params.serial_port = self.port or 'COM3'
                board_id = BoardIds.CYTON_BOARD.value
                print(f"    Connecting to OpenBCI Cyton on {params.serial_port}")

            elif self.device == 'EMOTIV':
                # Emotiv EPOC X — connect via Bluetooth
                board_id = BoardIds.EMOTIV_EPOC_X_BOARD.value
                print("    Connecting to Emotiv EPOC X via Bluetooth")

            elif self.device == 'MUSE':
                # Muse 2 — connect via Bluetooth
                params.serial_port = self.port or ''
                board_id = BoardIds.MUSE_2_BOARD.value
                print("    Connecting to Muse 2 via Bluetooth")

            self.board  = BoardShim(board_id, params)
            self.sfreq  = BoardShim.get_sampling_rate(board_id)
            eeg_chans   = BoardShim.get_eeg_channels(board_id)
            self.n_channels = len(eeg_chans)
            self.eeg_channels = eeg_chans

            BoardShim.enable_dev_board_logger()
            self.board.prepare_session()
            self.board.start_stream()
            print(f"    Connected! Sampling rate: {self.sfreq} Hz")
            print(f"    EEG channels: {self.n_channels}")

        except ImportError:
            print("    BrainFlow not installed — run: pip install brainflow")
            print("    Falling back to simulation")
            self.device = 'SIMULATE'

        except Exception as e:
            print(f"    Hardware connection failed: {e}")
            print("    Check that headset is on and paired")
            print("    Falling back to simulation")
            self.device = 'SIMULATE'

    def get_epoch(self, duration_sec=0.6):
        """
        Get one epoch of EEG data.
        In simulation: generates realistic fake data
        With hardware: reads real brain signals
        """
        n_samples = int(self.sfreq * duration_sec)

        if self.device == 'SIMULATE':
            # ── SIMULATION MODE ───────────────────────────────
            # This is what runs now
            # When you get hardware, this block is replaced
            # by the hardware block below
            t      = np.linspace(0, duration_sec, n_samples)
            epoch  = np.zeros((self.n_channels, n_samples))
            for ch in range(self.n_channels):
                alpha = 3e-6 * np.sin(
                    2*np.pi*10*t + np.random.rand()*2*np.pi)
                noise = 2e-6 * np.random.randn(n_samples)
                epoch[ch] = alpha + noise
            return epoch

        else:
            # ── REAL HARDWARE MODE ────────────────────────────
            # Reads actual brain signals from your headset
            # This is the ONE function that changes when
            # you plug in real hardware
            time.sleep(duration_sec)   # wait for data
            data = self.board.get_board_data()
            eeg  = data[self.eeg_channels, -n_samples:]

            # Pad if not enough samples yet
            if eeg.shape[1] < n_samples:
                eeg = np.pad(
                    eeg, ((0,0),(n_samples-eeg.shape[1],0)))

            # Convert to volts (BrainFlow gives µV)
            eeg = eeg * 1e-6
            return eeg[:8, :]   # return first 8 channels

    def get_motor_imagery_epoch(self, duration_sec=2.0):
        """
        Get a longer epoch for motor imagery detection.
        Person imagines movement for 2 seconds.
        """
        n_samples = int(self.sfreq * duration_sec)

        if self.device == 'SIMULATE':
            # Simulate motor imagery signal
            thought_class = np.random.randint(0, 4)
            t = np.linspace(0, duration_sec, n_samples)
            epoch = np.zeros((self.n_channels, n_samples))

            for ch in range(self.n_channels):
                # Background EEG
                alpha = 3e-6 * np.sin(2*np.pi*10*t)
                beta  = 2e-6 * np.sin(2*np.pi*20*t)
                noise = 1e-6 * np.random.randn(n_samples)
                epoch[ch] = alpha + beta + noise

                # ERD/ERS pattern for motor imagery
                # (Event Related Desynchronization)
                if thought_class in [0, 1]:   # hand imagery
                    # Beta suppression at C3/C4
                    if ch in [2, 3]:
                        erd = -1.5e-6 * np.sin(2*np.pi*20*t)
                        epoch[ch] += erd

            return epoch, thought_class

        else:
            # Real hardware — record 2 seconds of EEG
            # Show "IMAGINE" cue on screen first
            print("    Cue: IMAGINE NOW")
            time.sleep(duration_sec)
            data = self.board.get_board_data()
            eeg  = data[self.eeg_channels, -n_samples:]
            if eeg.shape[1] < n_samples:
                eeg = np.pad(
                    eeg, ((0,0),(n_samples-eeg.shape[1],0)))
            eeg = eeg * 1e-6
            return eeg[:8, :], None   # true label unknown

    def stop(self):
        """Cleanly disconnect from headset"""
        if self.board and self.device != 'SIMULATE':
            self.board.stop_stream()
            self.board.release_session()
            print("    Hardware disconnected cleanly")

# ══════════════════════════════════════════════════════════════
# STEP BY STEP HARDWARE SETUP GUIDE
# ══════════════════════════════════════════════════════════════
print("""
HARDWARE SETUP GUIDE
====================

OPTION A — OpenBCI Cyton (~₹16,000 from openbci.com)
-----------------------------------------------------
1. Buy OpenBCI Cyton board + USB dongle + dry electrodes
2. Attach electrodes to scalp at positions:
   Fz, Cz, Pz, Oz, P3, P4, PO7, PO8 (for P300)
   C3, C4, Fz, Pz (for motor imagery)
3. Plug USB dongle into laptop
4. Find COM port: Device Manager → Ports → COMx
5. Change in this file:
   DEVICE = 'OPENBCI'
   port   = 'COM3'   (your COM number)
6. Run: pip install brainflow
7. Run this file — it connects automatically

OPTION B — Emotiv EPOC X (~₹40,000 from emotiv.com)
-----------------------------------------------------
1. Buy Emotiv EPOC X headset
2. Install Emotiv App on laptop
3. Pair via Bluetooth
4. Change in this file:
   DEVICE = 'EMOTIV'
5. Run: pip install brainflow
6. Run this file — connects via Bluetooth

OPTION C — Muse 2 Headband (~₹20,000 from amazon.in)
-----------------------------------------------------
1. Buy Muse 2 headband
2. Enable Bluetooth on laptop
3. Pair Muse 2 in Bluetooth settings
4. Change in this file:
   DEVICE = 'MUSE'
5. Run: pip install brainflow
6. Run this file — connects via Bluetooth

WHAT CHANGES IN YOUR AI CODE
=============================
RIGHT NOW (simulation):
   epoch = generate_p300_epoch(is_target)

WITH HARDWARE (one line change):
   reader = BCIHardwareReader(device='OPENBCI', port='COM3')
   epoch  = reader.get_epoch(duration_sec=0.6)

THAT IS THE ONLY CHANGE NEEDED.
All your EEGNet, LDA, Flask server code stays identical.
""")

# ══════════════════════════════════════════════════════════════
# TEST the reader in simulation mode
# ══════════════════════════════════════════════════════════════
print("[Testing hardware reader in simulation mode...]")

reader = BCIHardwareReader(device=DEVICE)

print("\n    Getting 5 test epochs...")
for i in range(5):
    epoch = reader.get_epoch(duration_sec=0.6)
    print(f"    Epoch {i+1}: shape={epoch.shape}, "
          f"mean={epoch.mean()*1e6:.3f}µV, "
          f"max={epoch.max()*1e6:.3f}µV")

print("\n    Getting motor imagery epoch...")
epoch_mi, cls = reader.get_motor_imagery_epoch(duration_sec=2.0)
print(f"    Motor imagery epoch: shape={epoch_mi.shape}")
if cls is not None:
    labels = ['Left hand','Right hand','Both hands','Feet']
    print(f"    Simulated class: {labels[cls]}")

reader.stop()

print("\n" + "=" * 50)
print("  Hardware integration code ready!")
print("  When you buy a headset:")
print("  1. Change DEVICE = 'OPENBCI' (or EMOTIV/MUSE)")
print("  2. Run: pip install brainflow")
print("  3. Everything else works automatically")
print("=" * 50)

input("\nPress Enter to close...")
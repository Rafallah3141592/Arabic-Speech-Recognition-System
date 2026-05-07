
# Ahmed Rafallah 
#  13 - 1 - 2026

import os
import json
import threading
import queue
import sys
import time
import sounddevice as sd
from vosk import Model, KaldiRecognizer
import pyautogui
import pyperclip
import numpy as np

# ============================================================
# CONFIGURATION
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "vosk-model-ar-small")
SAMPLE_RATE = 16000
CHUNK = 4000
SILENCE_LIMIT = .2
SILENCE_THRESHOLD = 200
MIC_DEVICE_INDEX = None
# ============================================================

if not os.path.exists(MODEL_PATH):
    print("Model not found:", MODEL_PATH)
    sys.exit(1)

print("Loading Arabic model... please wait")
model = Model(MODEL_PATH)
print("Model loaded successfully!\n")

audio_q = queue.Queue()
running = False

def audio_callback(indata, frames, time_info, status):
    if status:
        print("", status, file=sys.stderr)
    audio_q.put(bytes(indata))

def is_silent(data):
    audio_np = np.frombuffer(data, dtype=np.int16)
    return np.abs(audio_np).mean() < SILENCE_THRESHOLD

def recognition_worker():
    global running
    rec = KaldiRecognizer(model, SAMPLE_RATE)
    rec.SetWords(True)

    buffer = b""
    silence_start = None
    speaking = False
    last_final_text = ""

    while running:
        try:
            data = audio_q.get(timeout=0.2)
        except queue.Empty:
            continue

        buffer += data
        silent = is_silent(data)

        if not silent:
            speaking = True
            silence_start = None
            rec.AcceptWaveform(data)

        else:
            if speaking:
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start > SILENCE_LIMIT:

                    result = json.loads(rec.FinalResult())
                    text = result.get("text", "").strip()

                    if text and text != last_final_text:
                        pyperclip.copy(text + " ")
                        pyautogui.hotkey("ctrl", "v")
                        print(" ", text)
                        last_final_text = text

                    rec = KaldiRecognizer(model, SAMPLE_RATE)
                    buffer = b""
                    speaking = False
                    silence_start = None

def start_listening():
    global running, audio_stream, recog_thread

    running = True
    print("Listening...\n")

    audio_stream = sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=CHUNK,
        dtype="int16",
        channels=1,
        callback=audio_callback,
        device=MIC_DEVICE_INDEX,
    )
    audio_stream.start()

    recog_thread = threading.Thread(target=recognition_worker, daemon=True)
    recog_thread.start()

def stop_listening():
    global running, audio_stream
    running = False
    try:
        audio_stream.stop()
        audio_stream.close()
    except:
        pass
    with audio_q.mutex:
        audio_q.queue.clear()
    print("\nStopped listening.\n")

# ------------------------------------------------------------
# START LISTENING IMMEDIATELY AFTER LOADING MODEL
# ------------------------------------------------------------
start_listening()

while True:
    try:
        time.sleep(1)
    except KeyboardInterrupt:
        stop_listening()
        print("Exiting...")
        break

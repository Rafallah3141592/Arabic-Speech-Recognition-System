
# Ahmed Rafallah 
# 10 / 2 /  2026

import os
import queue
import threading
import time
import sounddevice as sd
import pyautogui
import pyperclip
import numpy as np
from faster_whisper import WhisperModel

# ================= CONFIGURATION =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "whisper-model-medium")  # optional local cache
MODEL_SIZE = "medium"          # tiny=fastest, small=fast, medium=accurate
SAMPLE_RATE = 16000
CHUNK = int(SAMPLE_RATE * 0.5)  # 0.7 second chunks → faster processing
SILENCE_THRESHOLD = 100
SILENCE_LIMIT = 0.2       # trigger transcription after 0.3 sec silence
MIC_DEVICE_INDEX = None
BEAM_SIZE = 1                   # integer, faster decoding
# ==================================================

print("Loading Whisper model...")
model = WhisperModel(MODEL_SIZE, device="cpu")  # or "cuda" for GPU
print("Model loaded successfully!\n")

audio_q = queue.Queue()
running = False

def audio_callback(indata, frames, time_info, status):
    if status:
        print("", status)
    audio_q.put(indata.copy())

def is_silent(data):
    audio_np = np.frombuffer(data, dtype=np.int16)
    return np.abs(audio_np).mean() < SILENCE_THRESHOLD

def recognition_worker():
    global running
    buffer = np.zeros((0,), dtype=np.int16)
    silence_start = None
    last_final_text = ""

    while running:
        try:
            data = audio_q.get(timeout=0.1)
        except queue.Empty:
            continue

        buffer = np.concatenate((buffer, data.flatten()))

        if not is_silent(data):
            silence_start = None
        else:
            if silence_start is None:
                silence_start = time.time()
            elif time.time() - silence_start > SILENCE_LIMIT:
                if len(buffer) > SAMPLE_RATE // 4:  # at least 0.25 sec audio
                    audio_float = buffer.astype(np.float32) / 32768.0
                    # transcribe
                    segments, _ = model.transcribe(
                        audio_float,
                        beam_size=BEAM_SIZE,
                        language="ar",
                        vad_filter=True
                    )
                    text = " ".join([segment.text for segment in segments]).strip()
                    if text and text != last_final_text:
                        pyperclip.copy(text + " ")
                        pyautogui.hotkey("ctrl", "v")
                        print(" ", text)
                        last_final_text = text
                    buffer = np.zeros((0,), dtype=np.int16)
                silence_start = None

def start_listening():
    global running, audio_stream, recog_thread
    running = True
    print("Listening...\n")

    audio_stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        blocksize=CHUNK,
        dtype="int16",
        channels=1,
        callback=audio_callback,
        device=MIC_DEVICE_INDEX
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

# ================= MAIN LOOP =================
start_listening()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    stop_listening()
    print("Exiting...")  



#   pyinstaller --noconsole --onefile --hidden-import=numpy --hidden-import=sounddevice --hidden-import=pyautogui --hidden-import=pyperclip --hidden-import=scipy.special._cdflib --hidden-import=torch.utils.tensorboard --add-binary "C:/Users/pc/AppData/Local/Programs/Python/Python312/Lib/site-packages/tbb/bin/tbb12.dll;." whisper.py

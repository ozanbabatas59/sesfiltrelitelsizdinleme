import sounddevice as sd
import numpy as np
import time
import wave
import os
import json
from datetime import datetime
from scipy.signal import butter, sosfilt, sosfilt_zi
from collections import deque
import threading

# ================================================================
#          TYT TH-UV88 RX - GELİŞMİŞ ALICI SES İŞLEMCİ
# ================================================================
# Özellikler:
# - Temiz, stabil ses işleme
# - Otomatik kayıt (sadece anlaşılır ses)
# - Whisper ile metin dönüşümü
# - Tarih/saat loglaması
# ================================================================

FS = 48000
BLOCKSIZE = 512

# ================== RX FİLTRE AYARLARI ==================
RX_HP_CUT = 280
RX_LP_CUT = 3100
RX_FILTER_ORDER = 4

DEEMPH_TC = 50e-6
RX_GAIN = 1.0

# ================== AGC AYARLARI ==================
RX_TARGET = 0.20
RX_AGC_ATTACK = 0.012
RX_AGC_RELEASE = 0.00015
RX_GAIN_MIN = 0.1
RX_GAIN_MAX = 25.0

# ================== SQUELCH AYARLARI ==================
SQUELCH_THRESHOLD = 0.004
SQUELCH_HOLD_TIME = 0.4
SQUELCH_ENABLED = True

# ================== NOISE GATE AYARLARI ==================
RX_GATE_THRESHOLD = 0.003
RX_GATE_HOLD_TIME = 0.5
RX_GATE_FLOOR = 0.015

# ================== COMPRESSOR AYARLARI ==================
RX_COMP_THRESHOLD = 0.14
RX_COMP_RATIO = 2.5
RX_COMP_ATTACK = 0.004
RX_COMP_RELEASE = 0.08
RX_COMP_KNEE = 0.04

# ================== LIMITER ==================
RX_LIMIT = 0.88

# ================== NOISE REDUCTION ==================
NR_ENABLED = True
NR_THRESHOLD = 0.005
NR_RATIO = 0.4

# ================== KAYIT AYARLARI ==================
RECORD_ENABLED = True
RECORD_DIR = "rx_recordings"
RECORD_MIN_DURATION = 1.5
RECORD_SILENCE_TIMEOUT = 2.0
RECORD_THRESHOLD = 0.008

# ================== TRANSKRİPSİYON AYARLARI ==================
TRANSCRIBE_ENABLED = True
WHISPER_MODEL = "base"
LOG_FILE = "rx_log.json"
LOG_TXT_FILE = "rx_log.txt"

# ================== VU METER ==================
VU_WIDTH = 45
VU_UPDATE = 0.06

# ================== PRESET AYARLAR ==================
PRESETS = {
    "clean": {
        "name": "🔊 Temiz Ses",
        "RX_HP_CUT": 280, "RX_LP_CUT": 3100,
        "NR_ENABLED": True, "NR_RATIO": 0.4,
        "RX_COMP_RATIO": 2.5
    },
    "dx": {
        "name": "📡 DX (Uzak Sinyal)",
        "RX_HP_CUT": 200, "RX_LP_CUT": 3300,
        "RX_GAIN_MAX": 35.0, "NR_ENABLED": True, "NR_RATIO": 0.5,
        "RX_AGC_RELEASE": 0.0001
    },
    "noisy": {
        "name": "🔇 Gürültülü Ortam",
        "RX_HP_CUT": 350, "RX_LP_CUT": 2800,
        "NR_ENABLED": True, "NR_RATIO": 0.6,
        "SQUELCH_THRESHOLD": 0.008
    },
    "flat": {
        "name": "📊 Düz (İşlemsiz)",
        "RX_HP_CUT": 100, "RX_LP_CUT": 4000,
        "NR_ENABLED": False, "RX_COMP_RATIO": 1.5
    }
}

# ================== SINIFLAR ==================
def butter_bandpass_sos(low, high, fs, order):
    return butter(order, [low / (fs/2), high / (fs/2)], btype="band", output='sos')

class DCBlocker:
    def __init__(self, alpha=0.997):
        self.alpha = alpha
        self.x_prev = 0.0
        self.y_prev = 0.0
    
    def process(self, audio):
        out = np.zeros_like(audio)
        for i in range(len(audio)):
            out[i] = audio[i] - self.x_prev + self.alpha * self.y_prev
            self.x_prev = audio[i]
            self.y_prev = out[i]
        return out

class DeEmphasis:
    def __init__(self, fs, tc=50e-6):
        self.alpha = 1.0 / (1 + fs * tc)
        self.prev_out = 0.0
    
    def process(self, audio):
        out = np.zeros_like(audio)
        for i in range(len(audio)):
            out[i] = self.alpha * audio[i] + (1 - self.alpha) * self.prev_out
            self.prev_out = out[i]
        return out

class Squelch:
    def __init__(self, threshold, hold_time, fs, blocksize):
        self.threshold = threshold
        self.hold_samples = int(hold_time * fs / blocksize)
        self.hold_counter = 0
        self.is_open = False
        self.enabled = True
    
    def process(self, audio, rms):
        if not self.enabled:
            self.is_open = True
            return audio
        
        if rms > self.threshold:
            self.is_open = True
            self.hold_counter = self.hold_samples
        elif self.hold_counter > 0:
            self.hold_counter -= 1
        else:
            self.is_open = False
        
        if not self.is_open:
            return audio * 0.0
        return audio

class NoiseGate:
    def __init__(self, threshold, hold_time, fs, blocksize, floor=0.02):
        self.threshold = threshold
        self.hold_samples = int(hold_time * fs / blocksize)
        self.hold_counter = 0
        self.is_open = False
        self.floor = floor
    
    def process(self, audio, rms):
        if rms > self.threshold:
            self.is_open = True
            self.hold_counter = self.hold_samples
        elif self.hold_counter > 0:
            self.hold_counter -= 1
        else:
            self.is_open = False
        
        if not self.is_open:
            return audio * self.floor
        return audio

class AGC:
    def __init__(self, target, attack, release, gain_min, gain_max):
        self.target = target
        self.attack = attack
        self.release = release
        self.gain_min = gain_min
        self.gain_max = gain_max
        self.gain = 1.0
    
    def process(self, audio):
        rms = np.sqrt(np.mean(audio ** 2) + 1e-9)
        target_gain = self.target / (rms + 1e-9)
        target_gain = np.clip(target_gain, self.gain_min, self.gain_max)
        
        if target_gain < self.gain:
            speed = self.attack
        else:
            speed = self.release
        
        self.gain += speed * (target_gain - self.gain)
        self.gain = np.clip(self.gain, self.gain_min, self.gain_max)
        
        return audio * self.gain, rms

class NoiseReduction:
    def __init__(self, threshold, ratio):
        self.threshold = threshold
        self.ratio = ratio
        self.enabled = True
        self.noise_floor = 0.0
        self.alpha = 0.008
    
    def process(self, audio, rms):
        if not self.enabled:
            return audio
        
        if rms < self.threshold:
            self.noise_floor = self.noise_floor * (1 - self.alpha) + rms * self.alpha
        
        if rms < self.threshold * 2.5:
            reduction = 1.0 - self.ratio * (1.0 - rms / (self.threshold * 2.5))
            reduction = max(0.08, reduction)
            return audio * reduction
        
        return audio

class Compressor:
    def __init__(self, threshold, ratio, attack, release, knee, fs):
        self.threshold = threshold
        self.ratio = ratio
        self.knee = knee
        self.attack_coef = 1 - np.exp(-1 / (fs * attack))
        self.release_coef = 1 - np.exp(-1 / (fs * release))
        self.envelope = 0.0
    
    def process(self, audio):
        out = np.zeros_like(audio)
        for i, x in enumerate(audio):
            level = abs(x)
            if level > self.envelope:
                self.envelope += self.attack_coef * (level - self.envelope)
            else:
                self.envelope += self.release_coef * (level - self.envelope)
            
            if self.envelope < self.threshold - self.knee:
                gain = 1.0
            elif self.envelope > self.threshold + self.knee:
                gain = self.threshold + (self.envelope - self.threshold) / self.ratio
                gain = gain / (self.envelope + 1e-9)
            else:
                knee_ratio = (self.envelope - self.threshold + self.knee) / (2 * self.knee)
                effective_ratio = 1 + (self.ratio - 1) * knee_ratio
                gain = 1 - (1 - 1/effective_ratio) * knee_ratio
            
            out[i] = x * gain
        return out

class Limiter:
    def __init__(self, limit):
        self.limit = limit
        self.gain = 1.0
    
    def process(self, audio):
        peak = np.max(np.abs(audio)) + 1e-9
        if peak * self.gain > self.limit:
            self.gain = self.limit / peak
        else:
            self.gain = min(1.0, self.gain * 1.003)
        
        return audio * self.gain

# ================== KAYIT YÖNETİCİSİ ==================
class AudioRecorder:
    def __init__(self, fs, record_dir, min_duration, silence_timeout, threshold):
        self.fs = fs
        self.record_dir = record_dir
        self.min_duration = min_duration
        self.silence_timeout = silence_timeout
        self.threshold = threshold
        
        self.is_recording = False
        self.audio_buffer = []
        self.silence_counter = 0
        self.record_start_time = None
        
        os.makedirs(record_dir, exist_ok=True)
    
    def process(self, audio, rms, is_speech):
        if not is_speech or rms < self.threshold:
            if self.is_recording:
                self.silence_counter += 1
                silence_duration = self.silence_counter * BLOCKSIZE / self.fs
                
                if silence_duration >= self.silence_timeout:
                    return self._stop_recording()
                else:
                    self.audio_buffer.append(audio.copy())
            return None
        
        self.silence_counter = 0
        
        if not self.is_recording:
            self.is_recording = True
            self.audio_buffer = []
            self.record_start_time = datetime.now()
        
        self.audio_buffer.append(audio.copy())
        return None
    
    def _stop_recording(self):
        self.is_recording = False
        
        if len(self.audio_buffer) == 0:
            return None
        
        duration = len(self.audio_buffer) * BLOCKSIZE / self.fs
        
        if duration < self.min_duration:
            self.audio_buffer = []
            return None
        
        audio_data = np.concatenate(self.audio_buffer)
        timestamp = self.record_start_time.strftime("%Y%m%d_%H%M%S")
        filename = f"rx_{timestamp}.wav"
        filepath = os.path.join(self.record_dir, filename)
        
        audio_int16 = (audio_data * 32767).astype(np.int16)
        with wave.open(filepath, 'w') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.fs)
            wf.writeframes(audio_int16.tobytes())
        
        self.audio_buffer = []
        
        return {
            "filepath": filepath,
            "filename": filename,
            "duration": duration,
            "start_time": self.record_start_time,
            "end_time": datetime.now()
        }

# ================== TRANSKRİPSİYON YÖNETİCİSİ ==================
class Transcriber:
    def __init__(self, model_name="base", log_file="rx_log.json", log_txt="rx_log.txt"):
        self.model_name = model_name
        self.model = None
        self.log_file = log_file
        self.log_txt = log_txt
        self.enabled = False
        self.queue = deque()
        self.thread = None
        self.running = False
    
    def initialize(self):
        try:
            import whisper
            print(f"\n⏳ Whisper '{self.model_name}' modeli yükleniyor...")
            self.model = whisper.load_model(self.model_name)
            self.enabled = True
            print(f"✅ Whisper hazır!")
            
            self.running = True
            self.thread = threading.Thread(target=self._process_queue, daemon=True)
            self.thread.start()
            
            return True
        except ImportError:
            print("\n⚠️  Whisper yüklü değil. Transkripsiyon devre dışı.")
            print("    Yüklemek için: pip install openai-whisper")
            return False
        except Exception as e:
            print(f"\n❌ Whisper yüklenemedi: {e}")
            return False
    
    def add_to_queue(self, recording_info):
        if self.enabled:
            self.queue.append(recording_info)
    
    def _process_queue(self):
        while self.running:
            if len(self.queue) > 0:
                recording = self.queue.popleft()
                self._transcribe(recording)
            time.sleep(0.5)
    
    def _transcribe(self, recording):
        try:
            filepath = recording["filepath"]
            
            result = self.model.transcribe(
                filepath,
                language="tr",
                fp16=False
            )
            
            text = result["text"].strip()
            
            if text and len(text) > 2:
                log_entry = {
                    "timestamp": recording["start_time"].isoformat(),
                    "end_time": recording["end_time"].isoformat(),
                    "duration": round(recording["duration"], 2),
                    "filename": recording["filename"],
                    "text": text,
                    "language": result.get("language", "tr")
                }
                
                self._append_json_log(log_entry)
                self._append_txt_log(log_entry)
                
                print(f"\n\n📝 [{recording['start_time'].strftime('%H:%M:%S')}] ({recording['duration']:.1f}s)")
                print(f"   {text}")
                print()
        
        except Exception as e:
            print(f"\n❌ Transkripsiyon hatası: {e}")
    
    def _append_json_log(self, entry):
        try:
            logs = []
            if os.path.exists(self.log_file):
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    logs = json.load(f)
            
            logs.append(entry)
            
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(logs, f, ensure_ascii=False, indent=2)
        except:
            pass
    
    def _append_txt_log(self, entry):
        try:
            with open(self.log_txt, 'a', encoding='utf-8') as f:
                f.write(f"[{entry['timestamp'][:19].replace('T', ' ')}] ")
                f.write(f"({entry['duration']}s) ")
                f.write(f"{entry['text']}\n")
        except:
            pass
    
    def stop(self):
        self.running = False

# ================== ANA PROGRAM ==================
print("\n" + "=" * 65)
print("📻 TYT TH-UV88 RX - GELİŞMİŞ ALICI SES İŞLEMCİ")
print("=" * 65)

# Preset seçimi
print("\n🎚️  Ses Profili Seçin:")
print("-" * 40)
for key, preset in PRESETS.items():
    print(f"  [{key}] {preset['name']}")
print("  [custom] Özel ayarlar")

preset_choice = input("\n➡ Profil (varsayılan: clean): ").strip().lower()
if not preset_choice:
    preset_choice = "clean"

if preset_choice in PRESETS:
    preset = PRESETS[preset_choice]
    print(f"\n✅ Profil yüklendi: {preset['name']}")
    
    if "RX_HP_CUT" in preset: RX_HP_CUT = preset["RX_HP_CUT"]
    if "RX_LP_CUT" in preset: RX_LP_CUT = preset["RX_LP_CUT"]
    if "NR_ENABLED" in preset: NR_ENABLED = preset["NR_ENABLED"]
    if "NR_RATIO" in preset: NR_RATIO = preset["NR_RATIO"]
    if "RX_COMP_RATIO" in preset: RX_COMP_RATIO = preset["RX_COMP_RATIO"]
    if "RX_GAIN_MAX" in preset: RX_GAIN_MAX = preset["RX_GAIN_MAX"]
    if "RX_AGC_RELEASE" in preset: RX_AGC_RELEASE = preset["RX_AGC_RELEASE"]
    if "SQUELCH_THRESHOLD" in preset: SQUELCH_THRESHOLD = preset["SQUELCH_THRESHOLD"]

# Kayıt seçeneği
print("\n💾 Kayıt Ayarları:")
rec_choice = input("   Otomatik kayıt aktif mi? (E/h): ").strip().lower()
RECORD_ENABLED = rec_choice != 'h'

if RECORD_ENABLED:
    trans_choice = input("   Whisper transkripsiyon aktif mi? (E/h): ").strip().lower()
    TRANSCRIBE_ENABLED = trans_choice != 'h'
else:
    TRANSCRIBE_ENABLED = False

# Filtreleri oluştur
rx_bp_sos = butter_bandpass_sos(RX_HP_CUT, RX_LP_CUT, FS, RX_FILTER_ORDER)
rx_filter_zi = None

# Instance'ları oluştur
rx_dc_blocker = DCBlocker()
rx_de_emph = DeEmphasis(FS, DEEMPH_TC)
rx_squelch = Squelch(SQUELCH_THRESHOLD, SQUELCH_HOLD_TIME, FS, BLOCKSIZE)
rx_squelch.enabled = SQUELCH_ENABLED
rx_noise_gate = NoiseGate(RX_GATE_THRESHOLD, RX_GATE_HOLD_TIME, FS, BLOCKSIZE, floor=RX_GATE_FLOOR)
rx_agc = AGC(RX_TARGET, RX_AGC_ATTACK, RX_AGC_RELEASE, RX_GAIN_MIN, RX_GAIN_MAX)
rx_noise_reduction = NoiseReduction(NR_THRESHOLD, NR_RATIO)
rx_noise_reduction.enabled = NR_ENABLED
rx_compressor = Compressor(RX_COMP_THRESHOLD, RX_COMP_RATIO, RX_COMP_ATTACK, RX_COMP_RELEASE, RX_COMP_KNEE, FS)
rx_limiter = Limiter(RX_LIMIT)

recorder = AudioRecorder(FS, RECORD_DIR, RECORD_MIN_DURATION, RECORD_SILENCE_TIMEOUT, RECORD_THRESHOLD) if RECORD_ENABLED else None
transcriber = Transcriber(WHISPER_MODEL, LOG_FILE, LOG_TXT_FILE) if TRANSCRIBE_ENABLED else None

if transcriber:
    transcriber.initialize()

def process_rx(audio, rms):
    global rx_filter_zi
    
    audio = rx_dc_blocker.process(audio)
    
    if rx_filter_zi is None:
        rx_filter_zi = sosfilt_zi(rx_bp_sos) * audio[0]
    audio, rx_filter_zi = sosfilt(rx_bp_sos, audio, zi=rx_filter_zi)
    
    audio = rx_de_emph.process(audio)
    audio = rx_noise_reduction.process(audio, rms)
    audio = rx_compressor.process(audio)
    audio = rx_limiter.process(audio)
    
    return audio

last_vu_time = 0
def vu_meter_rx(level, gain, squelch_open, recording):
    global last_vu_time
    now = time.time()
    if now - last_vu_time < VU_UPDATE:
        return
    last_vu_time = now
    
    bars = int(np.clip(level * 7, 0, 1) * VU_WIDTH)
    bar = "█" * bars + "░" * (VU_WIDTH - bars)
    
    if not squelch_open:
        signal = "⬛ SQL  "
    elif level > 0.15:
        signal = "🔴 S9+  "
    elif level > 0.08:
        signal = "🟡 S5-8 "
    elif level > 0.03:
        signal = "🟢 S1-4 "
    else:
        signal = "⚪ S0   "
    
    rec_indicator = "🔴REC" if recording else "     "
    
    print(f"\r🔊 RX |{bar}| {level:.3f} | G:{gain:>4.1f}x | {signal} {rec_indicator}", end="", flush=True)

# Cihaz seçimi
print("\n" + "-" * 65)
print("🎧 INPUT (UV88 Ses Çıkışı / VB-Cable) Cihazları:")
print("-" * 65)
devices = sd.query_devices()
for i, d in enumerate(devices):
    if d["max_input_channels"] > 0:
        print(f"  [{i}] {d['name']}")

IN_DEV = int(input("\n➡ Giriş numarası (UV88 RX): "))

print("\n🔊 OUTPUT (Hoparlör / Kulaklık) Cihazları:")
print("-" * 65)
for i, d in enumerate(devices):
    if d["max_output_channels"] > 0:
        print(f"  [{i}] {d['name']}")

OUT_DEV = int(input("\n➡ Çıkış numarası: "))

# Ayarları göster
print("\n" + "=" * 65)
print("📻 RX AYARLARI")
print("=" * 65)
print(f"  Sample Rate:    {FS} Hz")
print(f"  Filtre:         {RX_HP_CUT}-{RX_LP_CUT} Hz")
print(f"  De-emphasis:    {DEEMPH_TC*1e6:.0f}µs")
print(f"  AGC:            {RX_GAIN_MIN}x - {RX_GAIN_MAX}x")
print(f"  Squelch:        {'AÇIK' if SQUELCH_ENABLED else 'KAPALI'} @ {SQUELCH_THRESHOLD}")
print(f"  Noise Red.:     {'AÇIK' if NR_ENABLED else 'KAPALI'} ({NR_RATIO*100:.0f}%)")
print(f"  Compressor:     {RX_COMP_RATIO}:1")
print(f"  Kayıt:          {'AÇIK' if RECORD_ENABLED else 'KAPALI'} → {RECORD_DIR}/")
print(f"  Transkripsiyon: {'AÇIK' if TRANSCRIBE_ENABLED else 'KAPALI'}")
print("=" * 65)
print("\n📻 UV88 RX → Squelch → AGC → Filtre → De-emph → NR → Comp → 🔊")
if RECORD_ENABLED:
    print(f"            ↓")
    print(f"         💾 Kayıt → {'📝 Whisper' if TRANSCRIBE_ENABLED else ''}")
print()

def callback_rx(indata, outdata, frames, time_info, status):
    if status:
        print(f"\n⚠️ {status}")
    
    audio_in = indata[:, 0].copy()
    audio_in, rms = rx_agc.process(audio_in)
    audio_in = rx_squelch.process(audio_in, rms)
    
    is_recording = recorder.is_recording if recorder else False
    vu_meter_rx(rms, rx_agc.gain, rx_squelch.is_open, is_recording)
    
    if not rx_squelch.is_open:
        outdata[:, 0] = 0
        if recorder:
            recorder.process(audio_in, rms, False)
        return
    
    audio_in = rx_noise_gate.process(audio_in, rms)
    audio = process_rx(audio_in, rms)
    outdata[:, 0] = audio * RX_GAIN
    
    if recorder:
        result = recorder.process(audio, rms, rx_squelch.is_open)
        if result and transcriber:
            transcriber.add_to_queue(result)

try:
    with sd.Stream(
        samplerate=FS,
        channels=1,
        device=(IN_DEV, OUT_DEV),
        callback=callback_rx,
        blocksize=BLOCKSIZE,
        latency='low'
    ):
        print("✅ RX Aktif. Çıkmak için Ctrl+C\n")
        while True:
            time.sleep(0.5)
except KeyboardInterrupt:
    print("\n\n🛑 RX Kapatıldı.")
    if transcriber:
        transcriber.stop()
except Exception as e:
    print(f"\n❌ Hata: {e}")
    if transcriber:
        transcriber.stop()

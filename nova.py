import os
import contextlib
os.add_dll_directory(r"C:\ffmpeg\bin")
import sys
import warnings
import re
import json
import random
import threading
import queue
import wave
import tkinter as tk

warnings.filterwarnings("ignore")

import speech_recognition as sr
from faster_whisper import WhisperModel
import pygame
import requests

import nova_memory as memory

# --- 設定 ---
WAKE_WORD    = "hey nova"

# --- GPT-SoVITS API 設定 ---
GPT_SOVITS_URL   = "http://127.0.0.1:9880"
# 參考音檔（Reference audio）與其對應的逐字稿，兩者必須完全match，
# 音質與情緒會直接影響合成結果，換聲音就是換這兩個值。
REF_AUDIO_PATH   = r"C:\Users\jimli\Downloads\GPT-SoVITS-v2pro-20250604\GPT-SoVITS-v2pro-20250604\output\slicer_opt\mix_1m23s (audio-joiner.com).wav_0002401920_0002526720.wav"
REF_PROMPT_TEXT  = "技術，一切都在飛速進化。Nova"
REF_PROMPT_LANG  = "zh"       # 參考音檔的語言："zh" / "en" / "auto" 等
TEXT_LANG        = "zh"       # 要合成文字的語言，中英夾雜可以先試 "zh"，效果不理想再試 "auto"
GPT_SOVITS_FALLBACK_SR = 32000  # v2pro 預設輸出取樣率，失敗時拿來寫靜音檔用

# Ann 這個聲音模型的權重路徑，nova.py 啟動時會自動呼叫 API 切過去，
# 不用每次重開 API 伺服器都手動在瀏覽器貼一次 set_gpt_weights / set_sovits_weights
GPT_WEIGHTS_PATH   = r"C:\Users\jimli\Downloads\GPT-SoVITS-v2pro-20250604\GPT-SoVITS-v2pro-20250604\GPT_weights_v2ProPlus\Ann-e15.ckpt"
SOVITS_WEIGHTS_PATH = r"C:\Users\jimli\Downloads\GPT-SoVITS-v2pro-20250604\GPT-SoVITS-v2pro-20250604\SoVITS_weights_v2ProPlus\Ann_e8_s184.pth"

# 對應 WebUI「推理設置」裡調過的那組參數，讓 API 合成結果跟你測試時聽到的一致
GPT_SOVITS_PARAMS = {
    "top_k": 15,
    "top_p": 1,
    "temperature": 1,
    "repetition_penalty": 1.35,
    "batch_size": 20,
    "speed_factor": 1,
    "fragment_interval": 0.3,
    "text_split_method": "cut1",   # 對應「湊四句一切」
    "parallel_infer": True,
    "split_bucket": True,
    "seed": -1,                    # -1 = 保持隨機
    "media_type": "wav",
    "streaming_mode": False,
}

SHUTDOWN_PHRASES = ["關閉nova", "關掉nova", "關機", "晚安nova", "掰掰nova", "shutdownnova"]
CONFIRM_PHRASES  = ["確定", "對", "是的", "沒錯", "yes", "確認"]

WAIT_FOR_TEXT_KEYWORDS = ["分析", "拆解", "解釋", "說明", "這個句子", "這句", "幫我看", "幫我分析"]

text_input_queue = queue.Queue()

class ShutdownRequested(Exception):
    pass

def _is_shutdown_request(text):
    normalized = text.replace(" ", "").replace("，", "").replace("。", "").lower()
    return any(phrase in normalized for phrase in SHUTDOWN_PHRASES)

def _is_confirmation(text):
    normalized = text.replace(" ", "").lower()
    return any(phrase in normalized for phrase in CONFIRM_PHRASES)

def _needs_typed_input(voice_text):
    return any(kw in voice_text for kw in WAIT_FOR_TEXT_KEYWORDS)

def _clean_text_for_tts(text):
    """將模型產生的文字清洗為 GPT-SoVITS 讀得懂、乾淨的格式（保留英文，交給 GPT-SoVITS 的中英夾雜能力處理）"""
    if not text:
        return ""

    # 1. 移除括號內的動作或顏文字 (如 (•́ω•`) 或 (*笑*) )
    text = re.sub(
    r'[^\u4e00-\u9fff\sA-Za-z0-9，。！？、：；「」『』《》〈〉【】,.!?]', '', text)
    text = re.sub(r'[\*\`\#]', '', text)

    # 2. 標點轉換
    text = text.replace("——", "，").replace("……", "。").replace("...", "。")
    text = re.sub(r'[～〜~]+', '，', text)
    text = re.sub(r'[！!]{2,}', '！', text)
    text = re.sub(r'[？?]{2,}', '？', text)
    text = re.sub(r'[-－—]{2,}', '，', text)

    # 3. 過濾特殊 Unicode 符號，但保留中文、英文字母、數字與基本標點
    #    （不再暴力刪除英文字母，讓 GPT-SoVITS 自己處理中英夾雜）
    text = re.sub(
        r'[^\w\s，。！？、：；「」『』《》〈〉【】,.!?\u4e00-\u9fffA-Za-z0-9]', '', text
    )

    # 如果濾完變空白（例如整段只是顏文字/裝飾符號），
    # 就保持空字串，交給呼叫端（_tts_worker 的 `if not sentence: continue`）
    # 直接跳過，不要合成、也不要講任何填充音
    text = text.strip()

    return text

_SENTENCE_END = re.compile(r'[。！？!?]+')
_MAX_CHUNK_LEN = 80

def _write_silence(path, sample_rate=GPT_SOVITS_FALLBACK_SR, seconds=0.5):
    with wave.open(path, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00" * int(sample_rate * seconds * 2))


def _tts_to_file(text, path, timeout=120):
    """呼叫本地 GPT-SoVITS API (api_v2.py) 生成聲音檔，並防禦空聲音崩潰"""
    payload = {
        "text": text,
        "text_lang": TEXT_LANG,
        "ref_audio_path": REF_AUDIO_PATH,
        "prompt_text": REF_PROMPT_TEXT,
        "prompt_lang": REF_PROMPT_LANG,
        **GPT_SOVITS_PARAMS,
    }
    try:
        resp = requests.post(f"{GPT_SOVITS_URL}/tts", json=payload, timeout=timeout)
        if resp.status_code == 200 and resp.content and len(resp.content) > 44:
            with open(path, 'wb') as f:
                f.write(resp.content)
            return
        else:
            print(f"\n⚠️ GPT-SoVITS 回應異常（status={resp.status_code}）：{resp.text[:200]}")
    except Exception as e:
        print(f"\n⚠️ 呼叫 GPT-SoVITS API 失敗: {e}")

    # 終極防呆檢查：伺服器沒回東西 / 回傳太小，就補一段靜音，避免整個播放流程卡死
    print("\n⚠️ 警告：GPT-SoVITS 未成功合成語音！已自動替換為靜音。")
    _write_silence(path)

def _tts_worker(tts_queue, interrupt_event=None):
    idx = 0
    while True:
        sentence = tts_queue.get()
        if sentence is None:
            break
            
        if interrupt_event and interrupt_event.is_set():
            _drain_queue(tts_queue)
            break
            
        sentence = _clean_text_for_tts(sentence)
        if not sentence:
            continue
            
        tmp = f"_tts_{idx}.wav"
        idx += 1
        try:
            _tts_to_file(sentence, tmp)
            pygame.mixer.music.load(tmp)
            pygame.mixer.music.play()
            
            while pygame.mixer.music.get_busy():
                if interrupt_event and interrupt_event.is_set():
                    pygame.mixer.music.stop()
                    pygame.mixer.music.unload()
                    _drain_queue(tts_queue)
                    return
                pygame.time.Clock().tick(20)
            pygame.mixer.music.unload()
        except Exception as e:
            print(f"⚠️ 語音播放失敗（{sentence[:20]}）：{e}")
        finally:
            try:
                os.remove(tmp)
            except Exception:
                pass

def _drain_queue(q):
    while True:
        try:
            q.get_nowait()
        except queue.Empty:
            break

def speak(text):
    if not text or not text.strip():
        return
    tts_q = queue.Queue()
    worker = threading.Thread(target=_tts_worker, args=(tts_q, None), daemon=True)
    worker.start()
    parts = re.split(r'(?<=[。！？!?])', text)
    for p in parts:
        p = p.strip()
        if p:
            tts_q.put(p)
    tts_q.put(None)
    worker.join()

def speak_cached(text, filename):
    if not text:
        return
    os.makedirs("cache", exist_ok=True)
    
    if filename.endswith(".mp3"):
        filename = filename.replace(".mp3", ".wav")
        
    cache_path = os.path.join("cache", filename)
    
    # 檢查快取檔案是否健康
    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 44:
        try:
            pygame.mixer.music.load(cache_path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            pygame.mixer.music.unload()
            return 
        except pygame.error:
            print(f"\n⚠️ 發現損壞的快取檔案 {filename}，正在重新生成...")
            try:
                os.remove(cache_path)
            except Exception:
                pass
                
    print(f"\n[系統] 建立預設語音快取「{text}」中...")
    try:
        _tts_to_file(text, cache_path)
        pygame.mixer.music.load(cache_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        pygame.mixer.music.unload()
    except Exception as e:
        print(f"⚠️ 語音快取合成失敗：{e}")

_FILLER_PHRASES = [
    "嗯……",
    "讓我想一下……",
    "好，我看看……",
    "稍等……",
    "嗯，好……",
    "我想一下……",
    "嗯，讓我看看……",
]

def _interrupt_listener(recognizer, source, interrupt_event, capture_queue,
                         energy_threshold, min_energy_frames=3):
    FRAME_DURATION = 0.05
    consecutive = 0

    try:
        while not interrupt_event.is_set():
            try:
                frame = recognizer.record(source, duration=FRAME_DURATION)
            except Exception:
                break
            
            try:
                import audioop
                raw = frame.get_raw_data()
                rms_val = audioop.rms(raw, frame.sample_width)
            except Exception:
                rms_val = 0

            if rms_val > energy_threshold:
                consecutive += 1
                if consecutive >= min_energy_frames:
                    interrupt_event.set()
                    print("\n\n⚡ [打斷] 吉姆開口了，Nova 停止說話")
                    try:
                        audio = recognizer.listen(
                            source, timeout=2, phrase_time_limit=15
                        )
                        with open("_interrupt.wav", "wb") as f:
                            f.write(audio.get_wav_data())
                        capture_queue.put("_interrupt.wav")
                    except sr.WaitTimeoutError:
                        capture_queue.put(None)
                    break
            else:
                consecutive = 0
    except Exception as e:
        print(f"⚠️ [打斷監聽] 執行緒異常（不影響主流程）：{e}")

def think_and_speak(text, recognizer=None, mic_source=None):
    if not text.strip():
        return "", None

    print("\n[Nova 思考中...]")
    url = "http://localhost:11434/api/generate"
    context_block = memory.build_context_block()
    full_prompt = f"{context_block}Jim 現在說：{text}" if context_block else text
    
    # "think": False 直接關掉內心戲推論，大幅降低首字延遲
    payload = {"model": "nova", "prompt": full_prompt, "stream": True, "keep_alive": -1, "think": False}

    interrupt_event = threading.Event()
    capture_queue   = queue.Queue()

    tts_q  = queue.Queue()
    worker = threading.Thread(
        target=_tts_worker, args=(tts_q, interrupt_event), daemon=True
    )
    worker.start()

    listener_thread = None
    if recognizer is not None and mic_source is not None:
        listener_thread = threading.Thread(
            target=_interrupt_listener,
            args=(recognizer, mic_source, interrupt_event,
                  capture_queue, recognizer.energy_threshold),
            daemon=True,
        )
        listener_thread.start()

    filler = random.choice(_FILLER_PHRASES)
    tts_q.put(filler)

    full_reply = []
    buf = ""
    print("\nNova：", end="", flush=True)

    try:
        with requests.post(url, json=payload, stream=True) as resp:
            for line in resp.iter_lines():
                if interrupt_event.is_set():
                    break
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except Exception:
                    continue

                token = chunk.get("response", "")
                if token:
                    print(token, end="", flush=True)
                    full_reply.append(token)
                    buf += token

                    if _SENTENCE_END.search(buf) or len(buf) >= _MAX_CHUNK_LEN:
                        segment = buf.strip()
                        if segment:
                            tts_q.put(segment)
                        buf = ""

                if chunk.get("done", False):
                    break

    except Exception as e:
        tts_q.put(None)
        worker.join()
        if listener_thread:
            listener_thread.join(timeout=1)
        return f"連線大腦失敗：{e}", None

    if not interrupt_event.is_set() and buf.strip():
        tts_q.put(buf.strip())

    tts_q.put(None)
    print()

    worker.join()

    if listener_thread:
        listener_thread.join(timeout=2)

    interrupted_wav = None
    if interrupt_event.is_set():
        try:
            interrupted_wav = capture_queue.get_nowait()
        except queue.Empty:
            interrupted_wav = None

    reply = "".join(full_reply)
    memory.save_turn("user", text)
    memory.save_turn("nova", reply)
    memory.summarize_and_extract_if_needed()
    return reply, interrupted_wav

def think(text, recognizer=None, mic_source=None):
    reply, _ = think_and_speak(text, recognizer=recognizer, mic_source=mic_source)
    return reply

def record_clip(recognizer, source, timeout, phrase_time_limit=20):
    try:
        audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
    except sr.WaitTimeoutError:
        return None
    with open("temp.wav", "wb") as f:
        f.write(audio.get_wav_data())
    return "temp.wav"

def wait_for_wake_word(recognizer, source):
    print(f'\n🔵 [待機中] 請說 "{WAKE_WORD.capitalize()}" 喚醒我...')
    while True:
        wav_path = record_clip(recognizer, source, timeout=None, phrase_time_limit=6)
        if wav_path is None:
            continue
        segments, _ = wake_model.transcribe(
            wav_path, language="en",
            initial_prompt="Hey Nova. Hey Nova, are you there?",
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
            condition_on_previous_text=False,
        )
        text = "".join(seg.text for seg in segments if seg.no_speech_prob < 0.6).strip().lower()
        if any(v in text for v in ["hey nova", "hey, nova", "he nova", "a nova"]):
            return

def launch_input_window():
    def on_submit(event=None):
        text = entry.get().strip()
        if text:
            text_input_queue.put(text)
            entry.delete(0, tk.END)
            log.config(state=tk.NORMAL)
            log.insert(tk.END, f"你：{text}\n")
            log.config(state=tk.DISABLED)
            log.see(tk.END)

    root = tk.Tk()
    root.title("Nova 打字輸入")
    root.geometry("500x320")
    root.resizable(False, False)
    tk.Label(root, text="輸入句子後按 Enter 送給 Nova：", anchor="w", font=("Arial", 11)).pack(fill="x", padx=12, pady=(10, 0))
    log = tk.Text(root, height=10, state=tk.DISABLED, wrap=tk.WORD, font=("Arial", 10))
    log.pack(fill="both", expand=True, padx=12, pady=6)
    entry = tk.Entry(root, font=("Arial", 13))
    entry.pack(fill="x", padx=12, pady=(0, 12))
    entry.bind("<Return>", on_submit)
    entry.focus()
    root.mainloop()

CONVERSATION_IDLE_TIMEOUT = 30

def _handle_voice_input(r, source, voice_text):
    if _is_shutdown_request(voice_text):
        speak_cached("你確定要我關機嗎？確定的話說確定，不然我就繼續待命。", "shutdown_confirmation.wav")
        confirm_wav = record_clip(r, source, timeout=8, phrase_time_limit=6)
        confirm_text = ""
        if confirm_wav:
            segs, _ = command_model.transcribe(
                confirm_wav, language=None, vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
                condition_on_previous_text=False,
            )
            confirm_text = "".join(s.text for s in segs if s.no_speech_prob < 0.6)
        if _is_confirmation(confirm_text):
            speak_cached("好的，晚安 吉姆。", "goodnight.wav")
            raise ShutdownRequested()
        else:
            speak_cached("好，我會繼續待命。", "continue_waiting.wav")
            return None, None

    if _needs_typed_input(voice_text):
        speak_cached("好，把句子打給我。", "send_text.wav")
        print("⏳ [等待打字輸入...]")
        try:
            typed_text = text_input_queue.get(timeout=30)
            print(f"⌨️ [收到打字] {typed_text}")
            return think_and_speak(f"{voice_text}：{typed_text}")
        except queue.Empty:
            print("⚠️ 等待逾時，直接回應語音")
            return think_and_speak(voice_text, recognizer=r, mic_source=source)

    return think_and_speak(voice_text, recognizer=r, mic_source=source)

def _transcribe_wav(wav_path):
    segments, _ = command_model.transcribe(
        wav_path, language=None, vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=300),
        condition_on_previous_text=False,
    )
    return "".join(seg.text for seg in segments if seg.no_speech_prob < 0.6).strip()

def _conversation_loop(r, source, prefill_wav=None):
    next_wav = prefill_wav

    while True:
        try:
            typed_text = text_input_queue.get_nowait()
            print(f"\n⌨️ [打字輸入] {typed_text}")
            think(typed_text)
            next_wav = None
            continue
        except queue.Empty:
            pass

        if next_wav is not None:
            wav_path  = next_wav
            next_wav  = None
            print("⚡ [使用打斷截錄的音訊]")
        else:
            wav_path = record_clip(r, source, timeout=CONVERSATION_IDLE_TIMEOUT, phrase_time_limit=20)

        if wav_path is None:
            print(f'\n🔵 [對話結束，回到待機] 請說 "{WAKE_WORD.capitalize()}" 喚醒我...')
            return

        voice_text = _transcribe_wav(wav_path)

        if len(voice_text) < 2:
            continue

        print(f"你說：{voice_text}")
        _, interrupted_wav = _handle_voice_input(r, source, voice_text)

        if interrupted_wav:
            next_wav = interrupted_wav

def voice_loop(r, source):
    while True:
        try:
            typed_text = text_input_queue.get_nowait()
            print(f"\n⌨️ [打字輸入] {typed_text}")
            think(typed_text)
            _conversation_loop(r, source)
            continue
        except queue.Empty:
            pass

        wait_for_wake_word(r, source)
        print("\n🟢 [偵測到喚醒詞] 我在聽...")
        speak_cached("我在", "i_am.wav")

        wav_path = record_clip(r, source, timeout=60, phrase_time_limit=20)
        if wav_path is None:
            print(f'\n🔵 [對話結束，回到待機] 請說 "{WAKE_WORD.capitalize()}" 喚醒我...')
            continue

        voice_text = _transcribe_wav(wav_path)

        if len(voice_text) < 2:
            continue

        print(f"你說：{voice_text}")
        _, interrupted_wav = _handle_voice_input(r, source, voice_text)

        _conversation_loop(r, source, prefill_wav=interrupted_wav)

# --- 系統初始化 ---
print("========================================")
print("正在啟動 Nova 本地端系統...")

print(f"檢查 GPT-SoVITS API 伺服器（{GPT_SOVITS_URL}）...")
try:
    _health = requests.get(f"{GPT_SOVITS_URL}/", timeout=5)
    print("✅ GPT-SoVITS API 有回應。")
except Exception as e:
    print(f"❌ 連不到 GPT-SoVITS API：{e}")
    print(f"請先啟動 api.bat（或 go-api.bat），等它顯示 {GPT_SOVITS_URL} 準備就緒後再執行本程式。")
    sys.exit(1)

if not os.path.exists(REF_AUDIO_PATH):
    print(f"❌ 找不到參考音檔：{REF_AUDIO_PATH}")
    print("請把 REF_AUDIO_PATH 換成你實際的參考音檔路徑。")
    sys.exit(1)

print("切換到 Ann 聲音模型...")
try:
    _r1 = requests.get(f"{GPT_SOVITS_URL}/set_gpt_weights", params={"weights_path": GPT_WEIGHTS_PATH}, timeout=60)
    _r2 = requests.get(f"{GPT_SOVITS_URL}/set_sovits_weights", params={"weights_path": SOVITS_WEIGHTS_PATH}, timeout=60)
    if _r1.status_code == 200 and _r2.status_code == 200:
        print("✅ 模型權重切換成功。")
    else:
        print(f"⚠️ 模型權重切換可能失敗：GPT={_r1.status_code} {_r1.text[:200]}，SoVITS={_r2.status_code} {_r2.text[:200]}")
except Exception as e:
    print(f"⚠️ 切換模型權重時發生錯誤（將沿用 API 目前載入的模型）：{e}")

print("載入聽覺神經網路 (Whisper) 中...")
wake_model    = WhisperModel("base",  device="cuda", compute_type="float16")
command_model = WhisperModel("small", device="cuda", compute_type="float16")

pygame.mixer.init()
memory.init_db()

print("正在預熱 GPT-SoVITS（第一次合成通常較慢，先跑一次墊底）...")
_warmup_path = os.path.join("cache", "_warmup.wav")
os.makedirs("cache", exist_ok=True)
_tts_to_file("你好。", _warmup_path)
try:
    os.remove(_warmup_path)
except Exception:
    pass

print("✅ 系統啟動完成！使用 GPT-SoVITS API 合成引擎。")
print("========================================")

def main():
    r = sr.Recognizer()
    r.dynamic_energy_threshold = False
    r.pause_threshold = 0.8

    with sr.Microphone() as source:
        print("🎙️ 校準環境噪音中，請保持安靜約 2 秒...")
        r.adjust_for_ambient_noise(source, duration=2)
        r.energy_threshold = max(r.energy_threshold * 1.5, 300)
        print(f"✅ 校準完成，目前音量門檻：{r.energy_threshold:.0f}")

        gui_thread = threading.Thread(target=launch_input_window, daemon=True)
        gui_thread.start()
        print("💬 打字視窗已開啟，可同時用語音或打字與 Nova 對話。")

        voice_loop(r, source)

if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, ShutdownRequested):
        print("\n\n🔴 [系統關閉] Nova 進入休眠狀態。")
        try:
            export_path = memory.export_full_memory()
            print(f"💾 記憶已自動匯出：{export_path}")
        except Exception as e:
            pass
        sys.exit(0)

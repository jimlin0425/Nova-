# --- ffmpeg 路徑（Windows 上 faster-whisper 需要用到） ---
FFMPEG_DLL_DIR = r"C:\ffmpeg\bin"

# --- GPT-SoVITS API 設定 ---
GPT_SOVITS_URL = "http://127.0.0.1:9880"

# 參考音檔與其逐字稿，兩者必須完全match
REF_AUDIO_PATH  = r"C:\path\to\your\ref_audio.wav"
REF_PROMPT_TEXT = "參考音檔裡實際說的那句話"
REF_PROMPT_LANG = "zh"     # 參考音檔的語言："zh" / "en" / "auto" 等
TEXT_LANG       = "zh"     # 要合成文字的語言

# 你訓練出來的聲音模型權重路徑
GPT_WEIGHTS_PATH    = r"C:\path\to\your\GPT_weights\model.ckpt"
SOVITS_WEIGHTS_PATH = r"C:\path\to\your\SoVITS_weights\model.pth"

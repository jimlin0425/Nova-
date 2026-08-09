# -*- coding: utf-8 -*-
"""
config.example.py

這是設定檔範本。使用方式：
1. 複製這個檔案，改名為 config.py（跟 nova.py 放同一層）
2. 把下面的路徑換成你自己電腦上的實際路徑
3. config.py 已經被 .gitignore 排除，不會被上傳到 GitHub，放心填真實路徑

⚠️ 不要把改好的 config.py 上傳到公開的 GitHub repo：
   - REF_AUDIO_PATH 等路徑會透露你的資料夾結構跟使用者名稱
   - 如果你用 GPT-SoVITS 複製了自己的聲音，模型權重與參考音檔本身
     等於是「你的聲音」，公開分享有被拿去冒充你聲音的風險
"""

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

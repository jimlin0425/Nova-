# Nova — 本地端語音 AI 助理

Nova 是一個完全在本機運行、離線的語音 AI 助理，靈感來自鋼鐵人的 Jarvis。

## 架構

| 模組 | 技術 |
|---|---|
| 大腦（LLM） | [Ollama](https://ollama.com/) + Qwen3:8b（見 `Modelfile`） |
| 耳朵（STT） | [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) |
| 嘴巴（TTS） | [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) v2Pro，透過本地 API 呼叫 |
| 記憶 | `nova_memory.py`：SQLite 三層記憶（逐字對話 / 使用者事實 / 摘要） |
| 介面 | Tkinter 打字輸入視窗 + Pygame 播放語音 |

## 前置需求

- **作業系統**：Windows
- **GPU**：需要 NVIDIA 顯示卡，並裝好對應版本的驅動程式與 CUDA / cuDNN（Faster-Whisper 靠這個跑 CUDA 加速）。VRAM 建議 8GB 以上；若同時要跑 LLM（Ollama）、兩個 Whisper 模型、GPT-SoVITS，8GB 屬於偏緊的配置，第一次啟動或對話時反應可能較慢（見下方「已知限制」）。
- **Python**：建議 3.12，用 venv 或 conda 建立獨立環境

## 安裝

### 1. Python 依賴

```bash
pip install -r requirements.txt
```

另外需要：
- [ffmpeg](https://ffmpeg.org/)（Windows 請下載並記下 `bin` 資料夾路徑）
- [Ollama](https://ollama.com/)，並用 `Modelfile` 建立 `nova` 模型：
  ```bash
  ollama create nova -f Modelfile
  ```
- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)（建議用官方整合包），並依照官方說明訓練/準備好你自己的聲音模型

### 2. 設定檔

```bash
cp config.example.py config.py
```

打開 `config.py`，填入你自己的：
- ffmpeg 路徑
- GPT-SoVITS 參考音檔路徑與逐字稿
- 你訓練出來的 GPT / SoVITS 模型權重路徑

`config.py` 已列在 `.gitignore`，不會被提交，可以放心填寫真實路徑。

### 3. 啟動順序

1. 啟動 GPT-SoVITS API 伺服器（在 GPT-SoVITS 資料夾內）：
   ```
   runtime\python.exe -I api_v2.py -a 127.0.0.1 -p 9880 -c GPT_SoVITS/configs/tts_infer.yaml
   ```
   （或使用附上的 `go-api.bat`，需放在 GPT-SoVITS 根目錄）
2. 確認 Ollama 服務正在跑（`ollama serve` 或背景常駐）
3. 執行：
   ```bash
   python nova.py
   ```

啟動後對麥克風說 **"Hey Nova"** 喚醒，開始對話。

## 已知限制

- **TTS 合成偶爾偏慢（第一句話尤其明顯）**：若你的 GPU VRAM 較吃緊（例如 8GB），Ollama、Faster-Whisper、GPT-SoVITS 同時載入模型時可能發生資源競爭，合成速度會變慢。這不代表設定錯誤，屬於已知的效能待優化項目。
- **LLM 偶爾會用英文回覆**，而非 `Modelfile` 設定的繁體中文，原因尚在排查中。

## ⚠️ 隱私與安全注意事項

這個 repo **不包含**：
- 任何人的聲音模型權重或參考音檔（GPT-SoVITS 語音複製模型等同「數位聲音」，公開分享有被冒用風險）
- `nova_memory.db` / `nova_memory_export.json`（含實際對話記錄與個人資訊）

如果你要 fork 這個專案訓練自己的聲音，**強烈建議不要把訓練好的模型權重或參考音檔上傳到公開 repo**，請自行保管在本機或私有雲端空間。

## 檔案說明

| 檔案 | 用途 |
|---|---|
| `nova.py` | 主程式：語音辨識、喚醒詞偵測、LLM 對話、TTS 播放 |
| `nova_memory.py` | 記憶模組，也可獨立執行做記憶管理（見 `python nova_memory.py --help`） |
| `config.example.py` | 設定檔範本，複製為 `config.py` 後填入真實路徑 |
| `Modelfile` | Ollama 自訂模型的系統提示詞設定 |
| `go-api.bat` | GPT-SoVITS API 伺服器啟動捷徑（放在 GPT-SoVITS 根目錄使用） |

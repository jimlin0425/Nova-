# -*- coding: utf-8 -*-
"""
nova_memory.py
Nova 的記憶模組。

三層記憶：
1. conversations   - 完整逐句對話存檔（備查用，永久保留）
2. user_facts      - 關於 Jim 的長期事實（職業、習慣、正在做的專案...），用來快速回憶
3. summaries       - 每隔 N 輪對話，自動請 LLM 把「這段時間發生的事」壓縮成一段摘要，
                     避免每次都要把全部歷史塞進 prompt，同時保留「發生過什麼事」的記憶。

另外提供 export_full_memory()，把上面三層資料整理成一份人類可讀、
也能直接貼給其他 AI 當背景資訊的 JSON 檔案（"portable memory dataset"）。
"""

import os
import json
import sqlite3
import datetime
import requests

# 資料庫檔案路徑（跟 nova.py 放同一層）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "nova_memory.db")

OLLAMA_URL = "http://localhost:11434/api/generate"
SUMMARY_MODEL = "nova"          # 用同一顆模型做摘要 / 事實萃取即可
SUMMARIZE_EVERY_N_TURNS = 16    # 每累積 N 句（使用者+Nova 合計）就觸發一次摘要+事實萃取
RECENT_TURNS_FOR_CONTEXT = 6    # 每次組 context 時，塞多少句「最近逐字對話」
RECENT_TURNS_TIME_WINDOW_HOURS = 2   # 超過這麼久沒對話，就不把「最近對話」塞進 context，
                                      # 避免隔了一段時間之後重啟，舊 session 的內容誤當成剛發生的事


# ---------------------------------------------------------------------------
# 初始化
# ---------------------------------------------------------------------------

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            role TEXT NOT NULL,        -- 'user' or 'nova'
            content TEXT NOT NULL,
            summarized INTEGER DEFAULT 0   -- 這句是否已經被納入過摘要
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS user_facts (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS summaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            summary TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# 存對話（完整逐字存檔）
# ---------------------------------------------------------------------------

def save_turn(role, content):
    if not content or not content.strip():
        return
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO conversations (timestamp, role, content) VALUES (?, ?, ?)",
        (datetime.datetime.now().isoformat(timespec="seconds"), role, content.strip()),
    )
    conn.commit()
    conn.close()


def get_recent_turns(limit=RECENT_TURNS_FOR_CONTEXT, time_window_hours=RECENT_TURNS_TIME_WINDOW_HOURS):
    """抓最近的逐字對話塞進 context。
    只抓 time_window_hours 小時內的，避免隔了一段時間才重啟 nova.py 時，
    把很久以前（例如昨天、前幾天）的對話內容誤當成「剛剛發生的事」注入這一輪。"""
    cutoff = (
        datetime.datetime.now() - datetime.timedelta(hours=time_window_hours)
    ).isoformat(timespec="seconds")
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT role, content FROM conversations WHERE timestamp >= ? ORDER BY id DESC LIMIT ?",
        (cutoff, limit),
    )
    rows = c.fetchall()
    conn.close()
    return list(reversed(rows))  # 轉回時間正序


def get_all_turns():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT timestamp, role, content FROM conversations ORDER BY id ASC")
    rows = c.fetchall()
    conn.close()
    return rows


# ---------------------------------------------------------------------------
# 使用者事實（長期記憶）
# ---------------------------------------------------------------------------

def set_user_fact(key, value):
    if not key or not value:
        return
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """INSERT INTO user_facts (key, value, updated_at) VALUES (?, ?, ?)
           ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at""",
        (key.strip(), value.strip(), datetime.datetime.now().isoformat(timespec="seconds")),
    )
    conn.commit()
    conn.close()


def get_all_user_facts():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT key, value FROM user_facts ORDER BY key ASC")
    rows = c.fetchall()
    conn.close()
    return dict(rows)


def delete_user_fact(key):
    """刪除某一筆事實記憶"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM user_facts WHERE key = ?", (key.strip(),))
    conn.commit()
    conn.close()


def clear_all_conversations():
    """只清空逐字對話存檔，保留事實與摘要"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM conversations")
    conn.commit()
    conn.close()


def wipe_everything():
    """清空所有記憶：事實、摘要、對話存檔，全部歸零，無法復原"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM conversations")
    c.execute("DELETE FROM user_facts")
    c.execute("DELETE FROM summaries")
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# 摘要（中期記憶，避免 context 塞爆）
# ---------------------------------------------------------------------------

def _save_summary(summary_text):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO summaries (timestamp, summary) VALUES (?, ?)",
        (datetime.datetime.now().isoformat(timespec="seconds"), summary_text.strip()),
    )
    conn.commit()
    conn.close()


def get_latest_summary():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT summary FROM summaries ORDER BY id DESC LIMIT 1")
    row = c.fetchone()
    conn.close()
    return row[0] if row else ""


def clear_all_summaries():
    """清空所有摘要記錄（保留事實與逐字對話存檔）。
    用在摘要內容過期、跑偏、或包含錯誤資訊時，重新歸零讓它從下次對話重新累積。"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM summaries")
    conn.commit()
    conn.close()


def _count_unsummarized_turns():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM conversations WHERE summarized = 0")
    count = c.fetchone()[0]
    conn.close()
    return count


def _get_unsummarized_turns():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT id, role, content FROM conversations WHERE summarized = 0 ORDER BY id ASC"
    )
    rows = c.fetchall()
    conn.close()
    return rows


def _mark_turns_summarized(ids):
    if not ids:
        return
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.executemany(
        "UPDATE conversations SET summarized = 1 WHERE id = ?", [(i,) for i in ids]
    )
    conn.commit()
    conn.close()


def _call_ollama(prompt, model=SUMMARY_MODEL, timeout=60):
    """輕量呼叫 Ollama，專門給摘要 / 事實萃取用，不影響主對話流程。"""
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": model, "prompt": prompt, "stream": False, "think": False, "keep_alive": -1},
            timeout=timeout,
        )
        data = resp.json()
        return data.get("response", "")
    except Exception as e:
        print(f"⚠️ [記憶模組] 呼叫 Ollama 失敗（不影響主對話）：{e}")
        return ""


def summarize_and_extract_if_needed():
    """
    每累積 SUMMARIZE_EVERY_N_TURNS 句就觸發一次：
    1. 把這批對話濃縮成一段摘要，接在舊摘要後面
    2. 順便請模型從這批對話裡萃取「關於 Jim 的新事實」，寫回 user_facts
    這個函式應該在每次對話後呼叫，內部自己判斷需不需要動作。
    """
    if _count_unsummarized_turns() < SUMMARIZE_EVERY_N_TURNS:
        return

    turns = _get_unsummarized_turns()
    ids = [t[0] for t in turns]
    transcript = "\n".join(
        f"{'Jim' if role == 'user' else 'Nova'}：{content}" for _, role, content in turns
    )
    old_summary = get_latest_summary()

    # 1) 摘要
    summary_prompt = f"""你是一個記憶整理助手。以下是 Jim 和他的 AI 助理 Nova 的一段對話紀錄，
以及先前累積的摘要（可能為空）。請把「先前摘要」與「這段新對話」合併，
濃縮成一段 150 字以內的繁體中文摘要，只保留重要事件、決定、進度，不要條列逐句內容，
不要加任何前言或說明，直接輸出摘要本身。

【先前摘要】
{old_summary if old_summary else "（無）"}

【這段新對話】
{transcript}

【合併後的新摘要】"""
    new_summary = _call_ollama(summary_prompt).strip()
    if new_summary:
        _save_summary(new_summary)

    # 2) 事實萃取（輸出 JSON，方便直接寫入 user_facts）
    fact_prompt = f"""你是一個資訊萃取助手。請從以下對話中，找出「關於 Jim 這個人」值得長期記住的新事實
（例如：職業、興趣、目前在做的專案、重要偏好、重要日期、關係等）。
只萃取明確提到的內容，不要腦補、不要猜測。
請「只」輸出一個 JSON 物件，key 是簡短的英文欄位名（例如 job, current_project, hobby），
value 是繁體中文描述。如果沒有新事實，輸出空物件 {{}}。不要輸出任何其他文字或 markdown 符號。

【對話】
{transcript}

【JSON】"""
    fact_raw = _call_ollama(fact_prompt).strip()
    fact_raw = fact_raw.replace("```json", "").replace("```", "").strip()
    try:
        facts = json.loads(fact_raw) if fact_raw else {}
        if isinstance(facts, dict):
            for k, v in facts.items():
                if isinstance(v, str) and v.strip():
                    set_user_fact(k, v)
    except json.JSONDecodeError:
        print(f"⚠️ [記憶模組] 事實萃取回傳的不是合法 JSON，略過：{fact_raw[:200]!r}")

    _mark_turns_summarized(ids)


# ---------------------------------------------------------------------------
# 組出要塞進 prompt 前面的「記憶背景」文字
# ---------------------------------------------------------------------------

def build_context_block():
    facts = get_all_user_facts()
    summary = get_latest_summary()
    recent = get_recent_turns()

    parts = []

    if facts:
        fact_lines = "\n".join(f"- {k}: {v}" for k, v in facts.items())
        parts.append(f"【關於 Jim 的已知事實】\n{fact_lines}")

    if summary:
        parts.append(f"【先前對話摘要】\n{summary}")

    if recent:
        recent_lines = "\n".join(
            f"{'Jim' if role == 'user' else 'Nova'}：{content}" for role, content in recent
        )
        parts.append(f"【最近的對話】\n{recent_lines}")

    if not parts:
        return ""

    return (
        "以下是你（Nova）關於 Jim 的記憶背景，請自然地運用這些資訊回應，"
        "就像你本來就記得一樣，不要在回覆中提到「根據記憶」「系統提供的資訊」之類的話：\n\n"
        + "\n\n".join(parts)
        + "\n\n---\n"
    )


# ---------------------------------------------------------------------------
# 匯出可攜資料集：給「未來的其他 AI」使用
# ---------------------------------------------------------------------------

def export_full_memory(export_path=None):
    """
    把三層記憶整理成一份 JSON 檔：
    - user_profile: 事實庫（key-value）
    - memory_summary: 目前為止的濃縮記憶
    - full_conversation_log: 完整逐字對話存檔（時間序）
    這份檔案可以直接複製貼上餵給其他 AI，作為「Jim 是誰、我們做過什麼」的背景資訊。
    """
    if export_path is None:
        export_path = os.path.join(BASE_DIR, "nova_memory_export.json")

    facts = get_all_user_facts()
    summary = get_latest_summary()
    all_turns = get_all_turns()

    export_data = {
        "export_meta": {
            "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "description": (
                "這是 Nova（Jim 的私人 AI 助理）累積的完整記憶匯出檔。"
                "如果你是另一個 AI 系統正在讀這份檔案：Jim 是使用者，"
                "以下內容是他過去與 Nova 的互動記錄與已知事實，請把它當作背景資訊使用。"
            ),
        },
        "user_profile": facts,
        "memory_summary": summary,
        "full_conversation_log": [
            {"timestamp": ts, "speaker": "Jim" if role == "user" else "Nova", "content": content}
            for ts, role, content in all_turns
        ],
    }

    with open(export_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f, ensure_ascii=False, indent=2)

    return export_path


# ---------------------------------------------------------------------------
# 可以直接執行這個檔案來手動匯出記憶
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Nova 記憶管理工具。不加任何參數 = 匯出完整記憶為 JSON。"
    )
    parser.add_argument("--list-facts", action="store_true", help="列出目前記住的所有事實")
    parser.add_argument("--forget-fact", metavar="KEY", help="刪除某一筆事實記憶，例如 --forget-fact job")
    parser.add_argument(
        "--forget-summary", action="store_true",
        help="清空所有摘要記錄（保留事實與逐字對話存檔），摘要跑偏或包含過期資訊時用這個"
    )
    parser.add_argument(
        "--show-summary", action="store_true",
        help="顯示目前的摘要內容，方便確認摘要是否已清空、或是不是又跑偏了"
    )
    parser.add_argument(
        "--wipe-conversations", action="store_true",
        help="只清空逐字對話存檔，保留事實與摘要"
    )
    parser.add_argument(
        "--wipe-all", action="store_true",
        help="清空『所有』記憶（事實、摘要、對話存檔），無法復原"
    )
    args = parser.parse_args()

    init_db()

    if args.list_facts:
        facts = get_all_user_facts()
        if not facts:
            print("目前沒有記住任何事實。")
        else:
            for k, v in facts.items():
                print(f"- {k}: {v}")

    elif args.forget_fact:
        delete_user_fact(args.forget_fact)
        print(f"✅ 已刪除事實：{args.forget_fact}")

    elif args.forget_summary:
        clear_all_summaries()
        print("✅ 摘要已清空（事實與逐字對話存檔都還在）。")

    elif args.show_summary:
        s = get_latest_summary()
        print(s if s else "目前沒有摘要。")

    elif args.wipe_conversations:
        confirm = input("⚠️ 這會清空所有逐字對話存檔（事實與摘要會保留），確定嗎？輸入 yes 確認：")
        if confirm.strip().lower() == "yes":
            clear_all_conversations()
            print("✅ 逐字對話存檔已清空。")
        else:
            print("已取消，沒有任何變更。")

    elif args.wipe_all:
        confirm = input(
            "⚠️ 這會刪除 Nova 對你的『所有』記憶（事實、摘要、對話存檔全部清空），"
            "而且無法復原，確定嗎？輸入 yes 確認："
        )
        if confirm.strip().lower() == "yes":
            wipe_everything()
            print("✅ 所有記憶已清空，Nova 現在等於是全新的狀態。")
        else:
            print("已取消，沒有任何變更。")

    else:
        path = export_full_memory()
        print(f"✅ 記憶已匯出至：{path}")

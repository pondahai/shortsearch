# shortsearch.py
# Lightweight news/web summarizer for Open-WebUI with hard budget + cache
# Python 3.11+

import re
import time
import json
import sqlite3
import hashlib
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional

import httpx
import feedparser
import trafilatura
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import unicodedata

# === Config ===
DB = "shortsearch.db"
# 建議先放你常看的 RSS；之後可再擴充
SOURCES = [
    "https://feedx.net/rss/cna.xml",            # 中央社 (總覽)
    "https://www.ithome.com.tw/rss",              # iThome IT 新聞
    "https://technews.tw/feed/"                   # TechNews 科技新報
    # 你也可加入在地/教育類 RSS
]
QUERY_TTL_SEC = 8 * 3600      # 查詢快取 8 小時
URL_TTL_SEC   = 24 * 3600     # 單頁快取 24 小時

# === App ===
app = FastAPI(
    title="ShortSearch API",
    version="0.2.0",
    openapi_url="/openapi.json",
    docs_url="/docs"
)

# 允許 Open-WebUI 讀 openapi.json 與呼叫工具
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # 如需更嚴謹：改填你的 Open-WebUI 來源
    allow_methods=["GET", "OPTIONS"],
    allow_headers=["*"],
    allow_credentials=False
)

# === DB ===
def init_db():
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS urlcache (
            url TEXT PRIMARY KEY,
            title TEXT,
            content TEXT,
            ts INTEGER
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS qcache (
            q TEXT PRIMARY KEY,
            payload TEXT,   -- JSON string
            ts INTEGER
        )
    """)
    conn.commit()
    conn.close()

init_db()

def now_ts() -> int:
    return int(time.time())

# === Utils ===

def normalize_text(s: str) -> str:
    # 全半形/組合字正規化、去空白、轉小寫
    s = unicodedata.normalize("NFKC", s or "").lower().strip()
    # 把連續空白壓一個
    s = re.sub(r"\s+", " ", s)
    return s

def match_query(q: str, title: str, content: str) -> bool:
    """
    比對邏輯：
    1) 先看標題是否含任一關鍵字
    2) 再看正文是否含任一關鍵字（若有抽到）
    3) 支援多詞：用空白分詞，任一命中即通過
    """
    q_norm = normalize_text(q)
    if not q_norm:
        return True  # 無關鍵字時不過濾

    toks = [t for t in q_norm.split(" ") if t]  # "AI 教育 部" → ["ai","教育","部"]
    if not toks:
        return True

    title_norm = normalize_text(title)
    content_norm = normalize_text(content)

    # 先看標題（最可靠）
    if any(t in title_norm for t in toks):
        return True
    # 再看正文
    if content_norm and any(t in content_norm for t in toks):
        return True

    return False

def clean_text(x: Optional[str]) -> str:
    x = x or ""
    x = re.sub(r"\s+", " ", x)
    return x.strip()

def summarize(txt: str, max_sent=2, lang="zh") -> str:
    """極簡 TextRank 摘要（sumy）；失敗則取前 120 字。"""
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.summarizers.text_rank import TextRankSummarizer

    t = clean_text(txt)
    if not t:
        return ""
    try:
        parser = PlaintextParser.from_string(
            t, Tokenizer("chinese" if lang.startswith("zh") else "english")
        )
        summarizer = TextRankSummarizer()
        sents = [str(s) for s in summarizer(parser.document, max_sent)]
        if not sents:
            sents = [t[:120]]
        return clean_text(" ".join(sents))[:240]
    except Exception:
        return t[:120]

def fetch_readable(url: str) -> Dict[str, str]:
    """讀取單頁並以 trafilatura 抽正文，帶 24h 快取。"""
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute("SELECT title, content, ts FROM urlcache WHERE url=?", (url,))
    row = cur.fetchone()
    if row and now_ts() - row[2] < URL_TTL_SEC:
        conn.close()
        return {"title": row[0] or "", "content": row[1] or ""}

    downloaded = trafilatura.fetch_url(url, no_ssl=True)
    title = ""
    content = ""
    if downloaded:
        try:
            meta = trafilatura.extract_metadata(downloaded)
            title = clean_text(meta.title if meta else "")
        except Exception:
            title = ""
        try:
            doc = trafilatura.extract(
                downloaded,
                include_comments=False,
                include_formatting=False
            )
            content = clean_text(doc or "")
        except Exception:
            content = ""

    try:
        cur.execute(
            "REPLACE INTO urlcache(url,title,content,ts) VALUES(?,?,?,?)",
            (url, title, content, now_ts())
        )
        conn.commit()
    finally:
        conn.close()
    return {"title": title, "content": content}

def build_bullets(txt: str) -> List[str]:
    """規則抓 1~3 個關鍵詞片段，保持極短。"""
    txt = clean_text(txt)
    cands = re.findall(
        r"(?:\d{1,2}日|\d{1,2}月|\d{4}年|\d+%|政策|公告|調整|新增|上線|下架|起實施|研議|修正|通過)",
        txt
    )[:3]
    return [clean_text(c)[:28] for c in cands]

def simhash_of(title: str, brief: str) -> int:
    from simhash import Simhash
    return Simhash((title + " " + brief).lower()).value

def hard_budget(items: List[Dict[str, Any]], budget=1200) -> List[Dict[str, Any]]:
    """對整體 payload 進行硬性字元預算裁切。"""
    out = []
    total = 0
    for it in items:
        it["t"] = it.get("t", "")[:60]
        it["brief"] = it.get("brief", "")[:120]
        it["bul"] = [b[:28] for b in (it.get("bul") or [])][:3]
        approx = len(it["t"]) + len(it["brief"]) + sum(len(b) for b in it["bul"]) + 40
        if total + approx <= budget:
            out.append(it)
            total += approx
        else:
            break
    return out

def score_item(dts: str, src_weight=1.0) -> float:
    """新鮮度（7 天內衰減）+ 來源權重。"""
    try:
        # 盡量解析 ISO 或 'YYYY-MM-DD'
        dt = datetime.fromisoformat(dts)
    except Exception:
        try:
            dt = datetime.strptime(dts, "%Y-%m-%d")
        except Exception:
            dt = datetime.now(timezone.utc) - timedelta(days=3)
    dt = dt.replace(tzinfo=timezone.utc)
    age_days = max(0.0, (datetime.now(timezone.utc) - dt).total_seconds() / 86400)
    freshness = max(0.0, 1.0 - age_days / 7.0)
    return 0.6 * freshness + 0.4 * src_weight

def pick_sources(q: str) -> List[str]:
    """MVP：直接回全列表；可依 q 做分類/白名單。"""
    return SOURCES

# === API ===
@app.get("/search_news")
async def search_news(
    q: str = Query(..., description="關鍵字"),
    n: int = Query(5, ge=1, le=10, description="最多要幾則（≤8）"),
    lang: str = Query("zh", description="語言"),
    budget: int = Query(1200, ge=100, le=4000, description="總字元上限")
):
    # 1) Query cache：若命中且 items>0 則直接回
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute("SELECT payload, ts FROM qcache WHERE q=?", (q,))
    row = cur.fetchone()
    if row and now_ts() - row[1] < QUERY_TTL_SEC:
        try:
            payload = json.loads(row[0]) if isinstance(row[0], str) else row[0]
        except Exception:
            payload = None
        if payload and isinstance(payload, dict) and len(payload.get("items", [])) > 0:
            conn.close()
            return JSONResponse(content=payload)
    conn.close()

    # 2) 抓 RSS + 正文抽取 + 摘要
    items_scored: List[tuple[float, Dict[str, Any]]] = []
    srcs = pick_sources(q)

    async with httpx.AsyncClient(timeout=10) as client:
        for rss in srcs:
            try:
                r = await client.get(rss, follow_redirects=True)
                fp = feedparser.parse(r.text)
                for e in fp.entries[:10]:  # 每源最多 10 篇
                    url = e.get("link") or ""
                    if not url:
                        continue

                    meta = fetch_readable(url)
                    title = clean_text(e.get("title") or meta.get("title") or "")
                    if not title:
                        continue

                    content = meta.get("content", "")
                    # MVP 的關鍵字過濾（標題+正文）
#                     hay = (title + " " + content).lower()
#                     if q and q.lower() not in hay:
#                         continue
                    if not match_query(q, title, content):
                        continue
                    
                    brief = summarize(content, max_sent=2, lang=lang)
                    if not brief:
                        continue

                    d_raw = e.get("published") or e.get("updated") or ""
                    d = clean_text(d_raw)[:10] or datetime.utcnow().date().isoformat()
                    src = ""
                    if e.get("source") and isinstance(e.get("source"), dict):
                        src = clean_text(e.get("source", {}).get("title"))
                    if not src:
                        src = clean_text(e.get("author") or "rss")

                    item = {
                        "t": title,
                        "u": url,
                        "d": d,
                        "src": src,
                        "brief": brief,
                        "bul": build_bullets(content),
                        "ents": [],
                        "cat": ""
                    }
                    score = score_item(d, 1.0)
                    items_scored.append((score, item))
            except Exception:
                # 某源失敗就跳過，不中斷整體
                continue

    # 3) 排序、去重、取前 n*2 再做硬預算裁切
    items_scored.sort(key=lambda x: x[0], reverse=True)
    uniq: List[Dict[str, Any]] = []
    seen: set[int] = set()
    for _, it in items_scored:
        key = simhash_of(it["t"], it["brief"]) >> 8  # 粗粒度避免重複
        if key in seen:
            continue
        seen.add(key)
        uniq.append(it)
        if len(uniq) >= max(3, n * 2):
            break

    # 若完全沒有命中，放寬條件：只看標題是否包含
    if len(uniq) == 0 and q:
        # 重新掃一輪，以「標題包含」為準（不看正文）
        tmp = []
        seen2 = set()
        for _, it in items_scored:
            title_norm = normalize_text(it["t"])
            if normalize_text(q) in title_norm:
                key = simhash_key(it["t"], it["brief"])
                if key in seen2: 
                    continue
                seen2.add(key)
                tmp.append(it)
            if len(tmp) >= max(3, n):
                break
        uniq = tmp

    # 若還是 0，最後退回「取來源最高分前 n 則」（確保不會 0 筆）
    if len(uniq) == 0:
        uniq = [it for _, it in items_scored[:max(3, n)]]
        
    uniq = uniq[:max(3, n)]
    uniq = hard_budget(uniq, budget=budget)

    payload = {
        "q": q,
        "generated_at": datetime.now().astimezone().isoformat(),
        "items": uniq,
        "note": f"共 {len(uniq)} 則；已去重、壓縮；budget={budget}"
    }

    # 4) 快取寫入：只在 items > 0 才寫；否則刪除舊空值/不寫入
    if len(uniq) > 0:
        try:
            conn = sqlite3.connect(DB)
            cur = conn.cursor()
            cur.execute(
                "REPLACE INTO qcache(q,payload,ts) VALUES(?,?,?)",
                (q, json.dumps(payload, ensure_ascii=False), now_ts())
            )
            conn.commit()
        finally:
            conn.close()
    else:
        # 確保不留空結果在快取
        try:
            conn = sqlite3.connect(DB)
            cur = conn.cursor()
            cur.execute("DELETE FROM qcache WHERE q=?", (q,))
            conn.commit()
        finally:
            conn.close()

    return JSONResponse(content=payload)

@app.get("/fetch")
async def fetch(
    url: str = Query(..., description="要摘要的網址"),
    budget: int = Query(800, ge=200, le=2000, description="總字元上限"),
    lang: str = Query("zh", description="語言")
):
    meta = fetch_readable(url)
    brief = summarize(meta.get("content", ""), max_sent=2, lang=lang)
    # 套用 micro hard-limit
    t = clean_text(meta.get("title", ""))[:60]
    brief = clean_text(brief)[: min(240, budget - 100)]
    bul = build_bullets(meta.get("content", ""))[:3]
    return JSONResponse(content={"t": t, "u": url, "brief": brief, "bul": bul})

@app.get("/health")
def health():
    return {"ok": True}

# === Entrypoint ===
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("shortsearch:app", host="127.0.0.1", port=8777, log_level="info")

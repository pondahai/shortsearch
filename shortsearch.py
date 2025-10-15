# shortsearch.py
import re
import time
import json
import sqlite3
import unicodedata
from urllib.parse import quote_plus
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional

import feedparser
import httpx
import trafilatura
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from simhash import Simhash

# =========================
# 配置區（可依需求修改）
# =========================

DB = "shortsearch.db"

# 固定 RSS 來源（無查詢參數）
SOURCES = [
    "https://feedx.net/rss/cna.xml",  # 中央社（轉載聚合）
    "https://www.ithome.com.tw/rss",  # iThome
    "https://technews.tw/feed/",      # TechNews 科技新報
]

# 啟用「Google News RSS 搜尋」作為動態來源
USE_GOOGLE_NEWS = True
GOOGLE_NEWS_PARAMS = {
    "hl": "zh-TW",          # 介面語言
    "gl": "TW",             # 地區
    "ceid": "TW:zh-Hant",   # 內容版位
}

# 站台白名單（可留空）。若非空，會在 Google News 查詢中加上 site: 限縮
# 例：["ithome.com.tw", "technews.tw"]
SITE_FILTERS: List[str] = []

# 查詢快取時間（秒）
Q_CACHE_TTL = 8 * 3600
# 單頁內容快取時間（秒）
URL_CACHE_TTL = 24 * 3600


# =========================
# FastAPI + CORS
# =========================

app = FastAPI(
    title="ShortSearch API",
    version="0.3.0",
    openapi_url="/openapi.json",
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # 若需嚴格限制，改成你的 Open-WebUI 網址
    allow_methods=["GET", "OPTIONS"],
    allow_headers=["*"],
    allow_credentials=False,
)


# =========================
# DB 初始化
# =========================

def init_db():
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute(
        "CREATE TABLE IF NOT EXISTS urlcache (url TEXT PRIMARY KEY, title TEXT, content TEXT, ts INTEGER)"
    )
    cur.execute(
        "CREATE TABLE IF NOT EXISTS qcache (q TEXT PRIMARY KEY, payload TEXT, ts INTEGER)"
    )
    conn.commit()
    conn.close()

init_db()


# =========================
# 工具函式 utils
# =========================

from urllib.parse import urlparse
from html import unescape

def is_gnews_url(url: str) -> bool:
    try:
        u = urlparse(url)
        return u.netloc in ("news.google.com", "www.news.google.com") and "/articles/" in u.path
    except Exception:
        return False

async def resolve_final_url(client: httpx.AsyncClient, url: str) -> str:
    """
    解析 Google News 轉址為出版方最終 URL；若解析失敗就回原 URL。
    """
    try:
        r = await client.get(url, follow_redirects=True, timeout=10)
        return str(r.url)
    except Exception:
        return url

def strip_html(x: str) -> str:
    # 用於從 RSS 的 <description>/<summary> 擷取純文字
    x = unescape(x or "")
    x = re.sub(r"<[^>]+>", " ", x)        # 去標籤
    x = re.sub(r"\s+", " ", x).strip()
    return x

def now_ts() -> int:
    return int(time.time())

def clean_text(x: Optional[str]) -> str:
    x = x or ""
    x = re.sub(r"\s+", " ", x)
    return x.strip()

def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "").lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s

def simple_sent_split(txt: str) -> List[str]:
    parts = re.split(r"[。！？!?\.\n\r]+", txt)
    sents = [clean_text(p) for p in parts if clean_text(p)]
    return sents

def summarize_text(txt: str, max_sent: int = 2) -> str:
    txt = clean_text(txt)
    if not txt:
        return ""
    try:
        from sumy.parsers.plaintext import PlaintextParser
        from sumy.nlp.tokenizers import Tokenizer
        from sumy.summarizers.text_rank import TextRankSummarizer
        parser = PlaintextParser.from_string(txt, Tokenizer("english"))
        summarizer = TextRankSummarizer()
        sents = [str(s) for s in summarizer(parser.document, max_sent)]
        s = clean_text(" ".join(sents))
        if s:
            return s[:240]
    except Exception:
        pass
    sents = simple_sent_split(txt)
    return clean_text(" ".join(sents[:max_sent]))[:240]

def build_bullets(txt: str) -> List[str]:
    cands = re.findall(
        r"(?:\d{1,2}日|\d{1,2}月|\d{4}年|\d+%|政策|公告|調整|新增|上線|下架|修正|開放|延長)",
        txt,
    )[:3]
    return [clean_text(c)[:28] for c in cands]

def simhash_key(title: str, brief: str) -> int:
    return Simhash((title + " " + brief).lower()).value >> 8

def hard_budget(items: List[Dict[str, Any]], budget: int = 1200) -> List[Dict[str, Any]]:
    out = []
    total = 0
    for it in items:
        it["t"] = it["t"][:60]
        it["brief"] = it["brief"][:120]
        it["bul"] = [b[:28] for b in (it.get("bul") or [])][:3]
        approx = len(it["t"]) + len(it["brief"]) + sum(len(b) for b in it["bul"]) + 40
        if total + approx <= budget:
            out.append(it)
            total += approx
        else:
            break
    return out

def score_item(dts: str, src_weight: float = 1.0) -> float:
    try:
        dt = datetime.fromisoformat(dts)
    except Exception:
        try:
            dt = datetime.strptime(dts, "%Y-%m-%d")
        except Exception:
            dt = datetime.now(timezone.utc) - timedelta(days=3)
    age_days = max(
        0.0,
        (datetime.now(timezone.utc) - dt.replace(tzinfo=timezone.utc)).total_seconds()
        / 86400,
    )
    freshness = max(0.0, 1.0 - age_days / 7.0)
    return 0.6 * freshness + 0.4 * src_weight

def match_query(q: str, title: str, content: str) -> bool:
    q_norm = normalize_text(q)
    if not q_norm:
        return True
    toks = [t for t in q_norm.split(" ") if t]
    if not toks:
        return True
    title_norm = normalize_text(title)
    content_norm = normalize_text(content)
    if any(t in title_norm for t in toks):
        return True
    if content_norm and any(t in content_norm for t in toks):
        return True
    return False


# =========================
# 來源選擇（含 Google News）
# =========================

def build_gnews_url(q: str, site: Optional[str] = None) -> str:
    """
    產生 Google News RSS URL，支援：
    - q：原始關鍵字
    - site：單一站台過濾（臨時）
    - SITE_FILTERS：全域站台白名單（多個用 OR）
    """
    parts = []

    q = (q or "").strip()
    if q:
        parts.append(q)

    # 單次臨時 site 過濾
    if site:
        parts.append(f"site:{site.strip()}")

    # 全域站台白名單
    if SITE_FILTERS:
        if len(SITE_FILTERS) == 1:
            parts.append(f"site:{SITE_FILTERS[0]}")
        else:
            or_part = " OR ".join([f"site:{d}" for d in SITE_FILTERS])
            parts.append(f"({or_part})")

    q_combined = " ".join(parts).strip()
    q_enc = quote_plus(q_combined) if q_combined else ""

    base = "https://news.google.com/rss/search"
    params = f"?q={q_enc}"
    params += f"&hl={GOOGLE_NEWS_PARAMS['hl']}&gl={GOOGLE_NEWS_PARAMS['gl']}&ceid={GOOGLE_NEWS_PARAMS['ceid']}"
    return base + params

def pick_sources(q: str, site: Optional[str] = None) -> List[str]:
    """
    有關鍵字時：若開啟 USE_GOOGLE_NEWS，將 Google News RSS 放在最前面
    無關鍵字時：只用固定來源
    """
    base = list(SOURCES)
    if USE_GOOGLE_NEWS and q and q.strip():
        gn = build_gnews_url(q, site=site)
        return [gn] + base
    return base


# =========================
# 內容抓取 + 快取
# =========================

def fetch_readable(url: str) -> Dict[str, str]:
    # cache hit?
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute("SELECT title, content, ts FROM urlcache WHERE url=?", (url,))
    row = cur.fetchone()
    if row and now_ts() - row[2] < URL_CACHE_TTL:
        conn.close()
        return {"title": row[0] or "", "content": row[1] or ""}

    downloaded = trafilatura.fetch_url(url, no_ssl=True)
    if not downloaded:
        conn.close()
        return {"title": "", "content": ""}

    title = ""
    try:
        meta = trafilatura.extract_metadata(downloaded)
        if meta and getattr(meta, "title", None):
            title = clean_text(meta.title)
    except Exception:
        title = ""

    try:
        doc = trafilatura.extract(
            downloaded, include_comments=False, include_formatting=False
        )
    except Exception:
        doc = ""
    content = clean_text(doc or "")

    cur.execute(
        "REPLACE INTO urlcache(url,title,content,ts) VALUES(?,?,?,?)",
        (url, title, content, now_ts()),
    )
    conn.commit()
    conn.close()
    return {"title": title, "content": content}


# =========================
# API
# =========================

@app.get("/health")
def health():
    return {"ok": True, "use_google_news": USE_GOOGLE_NEWS, "site_filters": SITE_FILTERS}

@app.get("/search_news")
async def search_news(
    q: str = Query(..., description="關鍵字"),
    n: int = Query(5, ge=1, le=10, description="最多要幾則（≤10）"),
    lang: str = Query("zh", description="語言"),
    budget: int = Query(1200, ge=300, le=4000, description="總字元上限"),
    site: Optional[str] = Query(None, description="只查某站（如：ithome.com.tw）"),
):
    # 伺服器側再夾一次，保險
    n = max(1, min(n, 10))
    budget = max(300, min(budget, 4000))

    # ---- Query Cache（命中且 items>0 才使用） ----
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cache_key = f"{q}|n={n}|lang={lang}|site={site or ''}"
    cur.execute("SELECT payload, ts FROM qcache WHERE q=?", (cache_key,))
    row = cur.fetchone()
    if row and now_ts() - row[1] < Q_CACHE_TTL:
        try:
            payload = json.loads(row[0]) if isinstance(row[0], str) else row[0]
        except Exception:
            payload = None
        if payload and isinstance(payload, dict) and len(payload.get("items", [])) > 0:
            conn.close()
            return JSONResponse(content=payload)
    conn.close()
    # 未命中或 items==0 → 繼續抓

    items_scored: List[tuple[float, Dict[str, Any]]] = []
    srcs = pick_sources(q, site=site)

    async with httpx.AsyncClient(timeout=12) as client:
        for rss in srcs:
            try:
                r = await client.get(rss, follow_redirects=True)
                fp = feedparser.parse(r.text)
            except Exception:
                continue

            for e in fp.entries[:12]:  # 每源最多取 12 篇
                url = e.get("link") or ""
                if not url:
                    continue

                # ★ 先處理 Google News /articles/... 的轉址 → 出版方真網址
                if is_gnews_url(url):
                    url = await resolve_final_url(client, url)

                # 先抓 RSS 自帶的 summary/description，當抽不到正文時的後備摘要
                rss_text = ""
                if "summary" in e and e.get("summary"):
                    rss_text = strip_html(e.get("summary"))
                elif "description" in e and e.get("description"):
                    rss_text = strip_html(e.get("description"))

                # 抽正文（可能會抽不到）
                meta = fetch_readable(url)
                title = clean_text(e.get("title") or meta["title"] or "")
                if not title:
                    continue
                content = meta["content"]

                # 關鍵字匹配：標題優先，再退正文；正文空也用 rss_text 兜著比
                if not match_query(q, title, content or rss_text):
                    continue

                # 摘要：正文優先，沒有就用 RSS 文本；再不行用標題兜底
                base_for_summary = content if content else rss_text
                brief = summarize_text(base_for_summary, max_sent=2)
                if not brief:
                    brief = title[:120]

                d = (e.get("published") or e.get("updated") or "")[:10]
                src_name = ""
                if e.get("source") and isinstance(e.get("source"), dict):
                    src_name = clean_text(e.get("source", {}).get("title"))
                if not src_name:
                    src_name = clean_text(e.get("author")) if e.get("author") else "rss"

                item = {
                    "t": title[:60],
                    "u": url,
                    "d": d or datetime.utcnow().date().isoformat(),
                    "src": src_name or "rss",
                    "brief": brief[:120],
                    "bul": build_bullets(base_for_summary),
                    "ents": [],
                    "cat": "",
                }
                items_scored.append((score_item(item["d"], 1.0), item))

    # 排序
    items_scored.sort(key=lambda x: x[0], reverse=True)

    # 第一次去重
    uniq: List[Dict[str, Any]] = []
    seen = set()
    for _, it in items_scored:
        key = simhash_key(it["t"], it["brief"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(it)
        if len(uniq) >= max(3, n * 2):
            break

    # Fallback 1：若沒有命中且 q 存在，放寬到只看標題包含
    if len(uniq) == 0 and q:
        tmp: List[Dict[str, Any]] = []
        seen2 = set()
        q_norm = normalize_text(q)
        for _, it in items_scored:
            title_norm = normalize_text(it["t"])
            if q_norm in title_norm:
                key2 = simhash_key(it["t"], it["brief"])
                if key2 in seen2:
                    continue
                seen2.add(key2)
                tmp.append(it)
            if len(tmp) >= max(3, n):
                break
        uniq = tmp

    # Fallback 2：若仍為 0，取最高分前 n 則（確保不會 0 筆）
    if len(uniq) == 0:
        uniq = [it for _, it in items_scored[:max(3, n)]]

    # 先粗砍到 n，再做硬性字元預算
    uniq = uniq[:max(3, n)]
    uniq = hard_budget(uniq, budget=budget)

    payload = {
        "q": q,
        "site": site,
        "generated_at": datetime.now().astimezone().isoformat(),
        "items": uniq,
        "note": f"共 {len(uniq)} 則；已去重、壓縮；budget={budget}"
                 + (f"；use_gnews=1" if (USE_GOOGLE_NEWS and q.strip()) else "；use_gnews=0")
    }

    # 只有 items>0 才寫入快取；否則刪除舊條目
    if len(uniq) > 0:
        s = json.dumps(payload, ensure_ascii=False)
        conn = sqlite3.connect(DB)
        cur = conn.cursor()
        cur.execute("REPLACE INTO qcache(q,payload,ts) VALUES(?,?,?)",
                    (cache_key, s, now_ts()))
        conn.commit()
        conn.close()
    else:
        try:
            conn = sqlite3.connect(DB)
            cur = conn.cursor()
            cur.execute("DELETE FROM qcache WHERE q=?", (cache_key,))
            conn.commit()
        finally:
            conn.close()

    return JSONResponse(content=payload)


@app.get("/fetch")
async def fetch(url: str, budget: int = 800, lang: str = "zh"):
    budget = max(300, min(budget, 4000))
    meta = fetch_readable(url)
    brief = summarize_text(meta["content"], max_sent=2)[: max(0, min(240, budget - 100))]
    item = {
        "t": meta["title"][:60],
        "u": url,
        "brief": brief,
        "bul": build_bullets(meta["content"])[:3],
    }
    return JSONResponse(content=item)


# =========================
# Main
# =========================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "shortsearch:app",
        host="127.0.0.1",
        port=8777,
        log_level="info",
        reload=False,
    )


# ==============================================================================
# shortsearch_improved.py (v1.2 - Complete and Standalone)
# 
# 這是一個完整、獨立且修正了所有先前問題的版本。
# 它包含了所有必要的輔助函式、完整的快取邏輯以及詳細的日誌記錄。
# 請使用此檔案進行測試。
# ==============================================================================

# --- Python Standard Libraries ---
import os
import re
import time
import json
import sqlite3
import logging
import unicodedata
import asyncio
from html import unescape
from urllib.parse import quote_plus, urlparse
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

# --- Third-party Libraries ---
import feedparser
import httpx
import trafilatura
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from simhash import Simhash
from pydantic_settings import BaseSettings

# --- Selenium for advanced scraping ---
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import WebDriverException

# --- OpenAI for LLM summarization ---
import openai

# ==============================================================================
# --- 1. 設定管理 (使用 Pydantic-Settings) ---
# ==============================================================================
class Settings(BaseSettings):
    DB_PATH: str = "shortsearch.db"
    Q_CACHE_TTL: int = 8 * 3600
    URL_CACHE_TTL: int = 24 * 3600
    USE_GOOGLE_NEWS: bool = True
    SITE_FILTERS: List[str] = []
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"
    OPENAI_MODEL: str = "gpt-4o-mini"
    SELENIUM_HEADLESS: bool = True
    MIN_CONTENT_LEN_FOR_SELENIUM: int = 200
    DEBUG_MODE: bool = False

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()

# ==============================================================================
# --- 2. 日誌與 FastAPI 初始化 ---
# ==============================================================================
# 強制使用 DEBUG 級別並覆蓋 Uvicorn 的預設值，以進行詳細診斷
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] [%(name)s:%(lineno)d] %(message)s",
    force=True
)
LOG = logging.getLogger("shortsearch_improved")
LOG.info("日誌系統已配置為 DEBUG 級別。")

app = FastAPI(
    title="ShortSearch API (Improved & Complete)",
    version="1.2.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["GET", "OPTIONS"], allow_headers=["*"],
)

# --- 全域資源 ---
REQ_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"}
executor = ThreadPoolExecutor(max_workers=4)

# ==============================================================================
# --- 3. 資料庫初始化 ---
# ==============================================================================
def init_db():
    with sqlite3.connect(settings.DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("CREATE TABLE IF NOT EXISTS urlcache (url TEXT PRIMARY KEY, title TEXT, content TEXT, ts INTEGER)")
        cur.execute("CREATE TABLE IF NOT EXISTS qcache (q TEXT PRIMARY KEY, payload TEXT, ts INTEGER)")
        conn.commit()
init_db()

# ==============================================================================
# --- 4. 核心輔助函式 (從原 shortsearch.py 移植) ---
# ==============================================================================
def now_ts() -> int:
    return int(time.time())

def clean_text(x: Optional[str]) -> str:
    x = x or ""
    x = re.sub(r"\s+", " ", x)
    return x.strip()

def strip_html(x: str) -> str:
    x = unescape(x or "")
    x = re.sub(r"<[^>]+>", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x

def is_gnews_url(url: str) -> bool:
    try:
        u = urlparse(url)
        return u.netloc.endswith("news.google.com") and ("/articles/" in u.path or "/rss/articles/" in u.path)
    except Exception:
        return False

def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "").lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s

def match_query_mode(q: str, title: str, content: str, mode: str = "auto") -> bool:
    mode = (mode or "auto").lower()
    if mode in ("any", "off"): return True
    
    q_norm = normalize_text(q)
    if not q_norm: return True
    toks = [t for t in q_norm.split(" ") if t]
    if not toks: return True

    if mode == "title":
        return any(t in normalize_text(title) for t in toks)
    
    # default: auto
    title_norm = normalize_text(title)
    content_norm = normalize_text(content)
    return any(t in title_norm for t in toks) or any(t in content_norm for t in toks)

def build_gnews_url(q: str, site: Optional[str] = None) -> str:
    q_enc = quote_plus(f"{q} site:{site}" if site else q)
    return f"https://news.google.com/rss/search?q={q_enc}&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"

def pick_sources(q: str, site: Optional[str] = None) -> List[Dict[str, Any]]:
    sources: List[Dict[str, Any]] = []
    if settings.USE_GOOGLE_NEWS and q and q.strip():
        sources.append({"url": build_gnews_url(q, site=site), "is_search": True})
    return sources

def fmt_feed_date(e: Dict[str, Any]) -> str:
    for key in ["published_parsed", "updated_parsed"]:
        if e.get(key):
            return time.strftime("%Y-%m-%d", e[key])
    return datetime.utcnow().date().isoformat()

def score_item(dts: str) -> float:
    try:
        dt = datetime.fromisoformat(dts).replace(tzinfo=timezone.utc)
    except Exception:
        dt = datetime.now(timezone.utc) - timedelta(days=3)
    age_days = (datetime.now(timezone.utc) - dt).total_seconds() / 86400.0
    return max(0.0, 1.0 - age_days / 7.0)

def simhash_key(title: str, brief: str) -> int:
    return Simhash((title + " " + brief).lower()).value

def build_bullets(txt: str) -> List[str]:
    cands = re.findall(r"(?:\d{1,2}日|\d{4}年|\d+%|政策|公告|調整|新增|修正|開放)", txt)[:3]
    return [clean_text(c)[:28] for c in cands]

def hard_budget(items: List[Dict[str, Any]], budget: int = 1200) -> List[Dict[str, Any]]:
    out, total = [], 0
    for it in items:
        approx = len(it.get("t", "")) + len(it.get("brief", "")) + sum(len(b) for b in it.get("bul", []))
        if total + approx <= budget:
            out.append(it)
            total += approx
    return out

# ==============================================================================
# --- 5. 內容抓取模組 (融合 Selenium) ---
# ==============================================================================
def get_selenium_driver() -> webdriver.Chrome:
    chrome_options = Options()
    if settings.SELENIUM_HEADLESS:
        chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.page_load_strategy = "eager"
    return webdriver.Chrome(options=chrome_options)

def _scrape_with_selenium(url: str) -> str:
    LOG.info(f"[Selenium] Fallback triggered for URL: {url[:80]}")
    html_content = ""
    try:
        with get_selenium_driver() as driver:
            driver.set_page_load_timeout(45)
            driver.get(url)
            time.sleep(5) 
            html_content = driver.page_source
    except WebDriverException as e:
        LOG.error(f"[Selenium] WebDriver error for {url}: {e}", exc_info=True)
    except Exception as e:
        LOG.error(f"[Selenium] General error for {url}: {e}", exc_info=True)
    return html_content

def _fetch_content_blocking(url: str) -> Dict[str, str]:
    content, title = "", ""
    try:
        downloaded = trafilatura.fetch_url(url, no_ssl=True, headers=REQ_HEADERS)
    except TypeError:
        LOG.warning("trafilatura version might be old, fetching without headers.")
        downloaded = trafilatura.fetch_url(url, no_ssl=True)
        
    if downloaded:
        content = trafilatura.extract(downloaded) or ""
    
    if len(content) < settings.MIN_CONTENT_LEN_FOR_SELENIUM:
        LOG.warning(f"Content from trafilatura is too short ({len(content)} chars). Trying Selenium.")
        selenium_html = _scrape_with_selenium(url)
        if selenium_html:
            new_content = trafilatura.extract(selenium_html) or ""
            if len(new_content) > len(content):
                LOG.info(f"Selenium extracted more content ({len(new_content)} chars).")
                content = new_content
    
    try:
        meta = trafilatura.extract_metadata(downloaded or selenium_html)
        if meta and meta.title:
             title = meta.title
    except Exception:
        pass
    
    return {"title": clean_text(title), "content": clean_text(content)}

async def fetch_full_content(url: str) -> Dict[str, str]:
    loop = asyncio.get_running_loop()

    # 線程安全的資料庫讀取
    def _db_read():
        with sqlite3.connect(settings.DB_PATH) as conn:
            cur = conn.cursor()
            cur.execute("SELECT title, content, ts FROM urlcache WHERE url=?", (url,))
            return cur.fetchone()

    row = await loop.run_in_executor(executor, _db_read)
    if row and now_ts() - row[2] < settings.URL_CACHE_TTL:
        return {"title": row[0] or "", "content": row[1] or ""}

    # 快取未命中，執行抓取
    result = await loop.run_in_executor(executor, _fetch_content_blocking, url)

    # 線程安全的資料庫寫入
    def _db_write():
        with sqlite3.connect(settings.DB_PATH) as conn:
            conn.execute("REPLACE INTO urlcache(url,title,content,ts) VALUES(?,?,?,?)",
                         (url, result["title"], result["content"], now_ts()))
            conn.commit()
    
    await loop.run_in_executor(executor, _db_write)
    return result


# ==============================================================================
# --- 6. 摘要模組 (本地 + LLM) ---
# ==============================================================================
def summarize_text_local(txt: str, max_sent: int = 2) -> str:
    txt = clean_text(txt)
    if not txt: return ""
    try:
        from sumy.parsers.plaintext import PlaintextParser
        from sumy.nlp.tokenizers import Tokenizer
        from sumy.summarizers.lsa import LsaSummarizer
        parser = PlaintextParser.from_string(txt, Tokenizer("chinese"))
        summarizer = LsaSummarizer()
        # 使用傳入的 max_sent 參數
        sents = [str(s) for s in summarizer(parser.document, max_sent)]
        return " ".join(sents).strip()[:240 * (max_sent // 2)] # 動態調整最大長度
    except Exception:
        parts = re.split(r"[。！？!?.\n]+", txt)
        # 使用傳入的 max_sent 參數
        return " ".join(p.strip() for p in parts[:max_sent] if p.strip())[:240 * (max_sent // 2)]

LLM_SUMMARY_PROMPTS = {
    "default": (
        "你是一位資深新聞編輯，請將以下新聞內文濃縮成一段不超過150字的客觀、精簡中文摘要。"
        "摘要應包含最關鍵的人物、事件和結論。請直接輸出摘要，不要任何開場白。"
    ),
    "bullet": (
        "你是一位高效的資訊分析師。請從以下新聞內文中，提煉出 3 到 5 個最關鍵的重點，並以條列式(bullet points)呈現。"
        "每個重點都應該簡潔明瞭。請直接輸出條列內容，格式為：\n- 重點一\n- 重點二"
    ),
    "qa": (
        "你是一位新聞問答專家。請閱讀以下新聞內文，並提出三個關鍵問題以及它們的簡潔答案。"
        "這能幫助讀者快速抓住文章核心。請直接輸出問答，格式為：\nQ1: ...\nA1: ...\nQ2: ..."
    )
}

async def summarize_with_llm(text_content: str, title: str, prompt_type: str = "default") -> str:
    if not settings.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not configured.")
    if not text_content: return title

    client = openai.AsyncOpenAI(api_key=settings.OPENAI_API_KEY, base_url=settings.OPENAI_BASE_URL)
#     prompt = "你是一位資深新聞編輯，請將以下新聞內文濃縮成一段不超過150字的客觀、精簡中文摘要。請直接輸出摘要，不要任何開場白。"
    prompt = LLM_SUMMARY_PROMPTS.get(prompt_type, LLM_SUMMARY_PROMPTS["default"])
    
    try:
        res = await client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"標題：{title}\n\n內文：{text_content[:6000]}"}
            ],
            max_tokens=500, temperature=0.2, timeout=90.0
        )
        summary = res.choices[0].message.content.strip()
        LOG.info(f"LLM summary (type: {prompt_type}) generated for '{title[:30]}...', length: {len(summary)}")
        return summary
    except Exception as e:
        LOG.error(f"LLM summarization failed: {e}")
        return f"摘要生成失敗：{e}"

# ==============================================================================
# --- 7. 核心 API 端點 (完整版,支持自訂摘要規則) ---
# ==============================================================================
@app.get("/search_news")
async def search_news(
    q: str = Query(..., description="關鍵字"),
    n: int = Query(5, ge=1, le=10, description="最多幾則"),
    budget: int = Query(1200, ge=300, le=4000, description="總字元上限"),
    site: Optional[str] = Query(None, description="只查某站"),
    match: str = Query("auto", description="關鍵字匹配模式: auto/title/any"),
    summary_mode: str = Query("local", description="摘要模式: local/llm"),
    # --- 新增的参数 ---
    summary_sentences: int = Query(2, ge=1, le=5, description="[本地摘要] 摘要的句子數量 (1-5)"),
    summary_prompt_type: str = Query("default", description="[LLM摘要] Prompt類型: default/bullet/qa")
):
    LOG.debug(f"收到請求: q='{q}', n={n}, summary_mode='{summary_mode}', sentences='{summary_sentences}', prompt='{summary_prompt_type}'")
    
    if summary_mode == "llm" and not settings.OPENAI_API_KEY:
        raise HTTPException(status_code=400, detail="LLM summary requested, but OPENAI_API_KEY is not configured.")

    # --- 快取逻辑（保持不变）---
    loop = asyncio.get_running_loop()
    cache_key = f"{q}|n={n}|site={site or ''}|budget={budget}|match={match}|smode={summary_mode}|ssen={summary_sentences}|spt={summary_prompt_type}"
    
    # ... (之前的快取讀取邏輯不變) ...
    def _cache_read():
        with sqlite3.connect(settings.DB_PATH) as conn:
            cur = conn.cursor()
            cur.execute("SELECT payload, ts FROM qcache WHERE q=?", (cache_key,))
            return cur.fetchone()
    row = await loop.run_in_executor(executor, _cache_read)
    if row and now_ts() - row[1] < settings.Q_CACHE_TTL:
        try:
            payload = json.loads(row[0])
            if payload and isinstance(payload, dict) and len(payload.get("items", [])) > 0:
                LOG.info(f"[CACHE HIT] For key: {cache_key[:80]}")
                return JSONResponse(content=payload)
        except Exception as e:
            LOG.warning(f"Cache decode error: {e}")
    LOG.info(f"[CACHE MISS] For key: {cache_key[:80]}")


    # --- 核心抓取逻辑 ---
    items_scored: List[Tuple[float, Dict[str, Any]]] = []
    srcs = pick_sources(q, site=site)
    LOG.debug(f"產生的 RSS 來源: {[s['url'] for s in srcs]}")

    async with httpx.AsyncClient(timeout=15, headers=REQ_HEADERS) as client:
        rss_tasks = [client.get(src["url"]) for src in srcs]
        rss_responses = await asyncio.gather(*rss_tasks, return_exceptions=True)
        LOG.debug(f"已完成 {len(rss_responses)} 個 RSS 來源的抓取。")

        content_fetch_tasks = []
        for i, res in enumerate(rss_responses):
            if isinstance(res, Exception):
                LOG.error(f"Failed to fetch RSS {srcs[i]['url']}: {res}")
                continue
            
            feed = feedparser.parse(res.text)
            LOG.debug(f"正在處理來源: {srcs[i]['url']}，找到 {len(feed.entries)} 個新聞條目。")
            is_search_source = srcs[i].get("is_search", False)

            for entry in feed.entries[:3]:
                url = entry.get("link")
                if not url: continue
                
                # ==========================================================
                # --- START: 這是本次修改的核心部分 ---
                # ==========================================================
                
                # 定义一个清晰的、接收所有参数的内部处理函数
                async def process_entry(
                    entry_data: Dict, 
                    entry_url: str, 
                    is_search_src: bool,
                    s_mode: str,          # 明确传入 summary_mode
                    s_sentences: int,     # 明确传入 summary_sentences
                    s_prompt_type: str    # 明确传入 summary_prompt_type
                ):
                    LOG.info(f"--- [START] Processing entry: {entry_data.get('title', 'NO TITLE')[:50]} ---")
                    
                    if is_gnews_url(entry_url):
                        try:
                            r = await client.get(entry_url, follow_redirects=True, timeout=10)
                            original_url, entry_url = entry_url, str(r.url)
                            LOG.info(f"[Redirect] Resolved {original_url[:50]} -> {entry_url[:50]}")
                        except Exception:
                            LOG.warning(f"Failed to resolve redirect for {entry_url}")
                    
                    meta = await fetch_full_content(entry_url)
                    LOG.info(f"[Fetch] Got title: '{meta.get('title', '')[:30]}', content len: {len(meta.get('content', ''))} for URL: {entry_url[:50]}")
                    
                    title = clean_text(entry_data.get("title") or meta.get("title"))
                    if not title:
                        LOG.warning(f"[FILTERED] No title found for URL: {entry_url}")
                        return None
                        
                    content = meta["content"]
                    rss_text = strip_html(entry_data.get("summary", ""))
                    
                    if not is_search_src and not match_query_mode(q, title, content or rss_text, mode=match):
                        LOG.debug(f"[FILTERED] Non-match for q='{q}' in title='{title[:30]}...'")
                        return None
                        
                    base_for_summary = content if content else rss_text
                    
                    # --- 在这里使用明确传入的参数来调用摘要函数 ---
                    if s_mode == 'llm':
                        brief = await summarize_with_llm(base_for_summary, title, prompt_type=s_prompt_type)
                    else:
                        brief = summarize_text_local(base_for_summary, max_sent=s_sentences)

                    if not brief: brief = title[:120]

                    return {
                        "t": title[:60], "u": entry_url, "d": fmt_feed_date(entry_data),
                        "src": clean_text(entry_data.get("source", {}).get("title", "rss")),
                        "brief": brief, "bul": build_bullets(base_for_summary),
                    }

                # 创建任务时，将所有需要的参数都传入
                content_fetch_tasks.append(process_entry(
                    entry, 
                    url, 
                    is_search_source,
                    summary_mode, 
                    summary_sentences, 
                    summary_prompt_type
                ))

                # ==========================================================
                # --- END: 核心修改部分结束 ---
                # ==========================================================

        processed_items = await asyncio.gather(*content_fetch_tasks, return_exceptions=True)
        
        for item in processed_items:
            if isinstance(item, Exception):
                LOG.error(f"Error during process_entry: {item}", exc_info=False)
            elif item:
                items_scored.append((score_item(item["d"]), item))

    # --- 结果处理与返回（保持不变）---
    LOG.debug(f"成功處理了 {len(items_scored)} 篇文章，準備去重和排序。")
    items_scored.sort(key=lambda x: x[0], reverse=True)
    uniq: List[Dict[str, Any]] = []
    seen = set()
    for _, it in items_scored:
        key = simhash_key(it["t"], it["brief"])
        if key not in seen:
            seen.add(key)
            uniq.append(it)
    
    final_items = hard_budget(uniq[:n], budget=budget)
    LOG.info(f"最終返回 {len(final_items)} 篇文章。")

    payload = {
        "q": q, "generated_at": datetime.now().astimezone().isoformat(), "items": final_items,
        "note": f"Returned {len(final_items)} items. Summary mode: {summary_mode}."
    }
    
    # ... (之前的快取写入逻辑不变) ...
    if len(final_items) > 0:
        s = json.dumps(payload, ensure_ascii=False)
        def _cache_write():
            with sqlite3.connect(settings.DB_PATH) as conn:
                conn.execute("REPLACE INTO qcache(q,payload,ts) VALUES(?,?,?)", (cache_key, s, now_ts()))
                conn.commit()
        await loop.run_in_executor(executor, _cache_write)
        LOG.info(f"Result for key '{cache_key[:80]}' saved to cache.")

    return JSONResponse(content=payload)

# ==============================================================================
# --- 8. 應用啟動 ---
# ==============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8777)

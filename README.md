# ShortSearch — 超輕量新聞/網頁搜尋摘要伺服器

一個面向 **Open-WebUI** 的本機 FastAPI 服務：
先在工具端做**去噪、去重、摘要與硬性字元預算**，再把**極短 JSON** 回傳給 LLM，徹底降低 token 開銷。

---

## ✨ 重點更新（相對於上一版）

* **來源帶屬性：** `pick_sources()` 回傳 `{"url", "is_search"}`；
  **Google News RSS（is_search=True）不再做關鍵字二次過濾**，固定來源（is_search=False）才本地過濾。
* **偵錯輸出：** `debug=1` 或環境變數 `SHORTSEARCH_DEBUG=1` 會附帶 `diagnostics`（來源清單、計數、耗時等），並把空 feed 落地到 `./_debug/`。
* **匹配模式：** `match=auto|title|any|off`（預設 `auto`）。

  > 只有 **非搜尋來源** 會套用匹配模式；搜尋來源（GNews）直接通過。
* **Google News 轉址解析更穩：** 辨識 `news.google.com/(rss/)articles/…`，並以標準 UA 追跳轉。
* **CJK 友善摘要：** 中文/日文/韓文直接用簡易句切，摘要更貼近原文。
* **快取鍵更新：** `qcache` 納入 `budget` 與 `match`，避免「不同預算/模式命中舊結果」。

---

## 目錄結構

```
shortsearch.py
shortsearch.db          # 首次啟動後產生
_debug/                 # debug=1 時可能生成（RSS 原文、log）
```

---

## 系統需求

* Python 3.11+（建議）
* 可用網路（首次安裝依賴）
* 可選：WSL/Ubuntu

---

## 安裝

```bash
# 建議：建立虛擬環境
python -m venv .venv && source .venv/bin/activate

# 基本依賴
pip install fastapi uvicorn httpx trafilatura simhash feedparser

# 可選：提升英文摘要品質（TextRank）
pip install sumy
```

---

## 啟動

**方式 A**（直接跑檔案）

```bash
python shortsearch.py
```

**方式 B**（uvicorn）

```bash
uvicorn shortsearch:app --host 127.0.0.1 --port 8777 --log-level info
```

健康檢查：

```
http://127.0.0.1:8777/health
```

OpenAPI：

```
http://127.0.0.1:8777/openapi.json
```

---

## 與 Open-WebUI 整合

1. Settings → Tools → **Add custom tool**
2. **Name**：`ShortSearch`
3. **OpenAPI URL**：`http://127.0.0.1:8777/openapi.json`
4. 儲存（必要時 **Refresh**）後可使用 `search_news` 與 `fetch`

> 建議在工具 Default Args：`n=5`、`budget=1000~1200`。

---

## 來源選擇與流程

* **有關鍵字 `q`** 且 `USE_GOOGLE_NEWS=True`：
  來源為 **[Google News(搜尋) + 固定 RSS 列表]**。

  * Google News URL 由 `build_gnews_url(q, site)` 組成
  * `SITE_FILTERS`（全域白名單）會以 `site:` OR 條件加入搜尋
* **沒有關鍵字**（或關閉 `USE_GOOGLE_NEWS`）：
  僅使用 **固定 RSS 列表**。

**重要：**

* **搜尋來源（is_search=True）**：**不再做關鍵字過濾**（因為 GNews 已經是搜尋結果）。
* **固定來源（is_search=False）**：才會做標題/內文的本地關鍵字匹配。

每則條目都會（在 `urlcache` 未命中時）進一步抓目標頁面，用 `trafilatura` 抽正文，並做摘要、去重與排序。

---

## API

### `GET /health`

回傳服務健康狀態與來源設定。

```json
{
  "ok": true,
  "use_google_news": true,
  "site_filters": []
}
```

### `GET /search_news`

以關鍵字搜尋來源 RSS，去噪、去重並在 `budget` 內回傳極短摘要。

**Query 參數**

* `q` *(必填)*：關鍵字（空白分詞；固定來源會套用匹配）
* `n` *(選填)*：最多返回篇數，**1–10**（預設 5；伺服器端夾範圍）
* `lang` *(選填)*：`zh` / `en`（影響摘要策略；預設 `zh`）
* `budget` *(選填)*：總字元上限，**300–4000**（預設 1200）
* `site` *(選填)*：僅搜尋特定站台（只影響 **Google News**）
* `match` *(選填)*：`auto|title|any|off`（**只對非搜尋來源生效**；預設 `auto`）
* `debug` *(選填)*：`1` 開啟診斷輸出（回傳 `diagnostics`，並將空 feed 存到 `_debug/`）

**回應範例**

```json
{
  "q": "AI 教育",
  "site": null,
  "match": "auto",
  "generated_at": "2025-10-17T09:30:00+08:00",
  "items": [
    {
      "t": "教育部推 AI 課程試辦",
      "u": "https://example.com/a1",
      "d": "2025-10-16",
      "src": "中央社",
      "brief": "教育部宣布高中階段試辦 AI 課程，重點在運算思維與倫理。",
      "bul": ["2025年10月", "試辦", "課綱調整"],
      "ents": [],
      "cat": ""
    }
  ],
  "note": "共 3 則；已去重、壓縮；budget=1200；use_gnews=1"
}
```

**匹配與 fallback**

* 非搜尋來源：`match=auto` 時需在**標題或正文**命中任一 token 才保留。
* 若最終 0 筆：

  1. 退而只看「標題包含整段 `q`」；再不行
  2. 取最高分前 n 則，**保證不回 0**。

**快取規則**

* `urlcache`（單頁正文）：TTL 24h
* `qcache`（查詢結果）：TTL 8h

  * **快取鍵**含：`q/n/lang/site/budget/match`
  * 只在 `items>0` 時寫入；`items==0` 會刪除舊條目（視為未命中）

### `GET /fetch`

摘要單一 URL（適合在對話中貼新聞/文章）。

**Query 參數**

* `url` *(必填)*
* `budget` *(選填)*（預設 800）
* `lang` *(選填)*：`zh` / `en`

**回應範例**

```json
{
  "t": "新聞標題",
  "u": "https://example.com/news/123",
  "brief": "文章正文明確兩句摘要……",
  "bul": ["2025年10月", "公告", "調整"]
}
```

---

## 參數建議（搭配 Open-WebUI）

* `n=5`、`budget=1000~1200`：新聞彙整的體感最佳
* 提示詞請**直接使用工具輸出**（已極短 & 去重），避免再次讓模型上網抓文
* 若要做二次整合與重點歸納，建議把輸入控制在 **600–800 字**

---

## 調整來源

在 `shortsearch.py` 上方調整 `SOURCES`：

```python
SOURCES = [
    "https://feedx.net/rss/cna.xml",
    "https://www.ithome.com.tw/rss",
    "https://technews.tw/feed/",
]
```

`SITE_FILTERS` 可限制 Google News 只搜某些域名（多個以 OR）：

```python
SITE_FILTERS = ["ithome.com.tw", "technews.tw"]
```

---

## 偵錯與診斷

* 單次開啟：

  ```bash
  curl -s --get "http://127.0.0.1:8777/search_news" \
    --data-urlencode "q=3i atlas" \
    --data-urlencode "n=10" \
    --data-urlencode "debug=1" | jq .
  ```
* 全域開啟：

  ```bash
  export SHORTSEARCH_DEBUG=1
  export SHORTSEARCH_DEBUG_DIR="./_debug"
  uvicorn shortsearch:app --host 127.0.0.1 --port 8777 --reload
  ```

**會得到什麼？**

* `diagnostics.sources[]`：每個 RSS 的 `status_code / entries / elapsed_ms / is_search`
* `counters`：`entries_total / skipped_no_url / skipped_empty_title / skipped_nonmatch / gnews_resolved / exceptions`
* 若某 feed 無 entries，原文會存成 `_debug/rss_empty_<ts>.xml`

---

## 常見問題（FAQ / Troubleshooting）

**Q1. 為什麼 GNews 也被過濾到 0？**
A：已改為 **不對 GNews 做本地關鍵字過濾**。若仍 0，請用 `debug=1` 檢查 `diagnostics.sources[].entries` 是否為 0（可能是暫時 403 或查無相關），或檢查 `site/SITE_FILTERS` 是否過度限縮。

**Q2. 很慢或偶爾拿不到正文？**
A：正文抽取失敗會退回 RSS 摘要。若想更快，可改成只用 RSS 摘要（我可以提供 `fetch_mode=rss_only` 的小 patch）。

**Q3. 一直看到舊結果？**
A：新版本的快取鍵已含 `budget/match`。若你之前手動改過 DB 或有奇怪行為，可先刪除 `shortsearch.db` 讓它重建。

**Q4. Open-WebUI 工具 schema 舊？**
A：在 Tools 重新 **Refresh**；不行就刪掉重加，或清除瀏覽器快取。

---

## 安全與隱私

* 只連你列出的 RSS 與其對應的文章頁面
* 本地 SQLite 快取，不外傳
* 若需嚴格 CORS，請把 `allow_origins=["*"]` 換成你的 Open-WebUI 網址

---

## 一鍵啟動（可選）

`run-shortsearch.sh`：

```bash
#!/usr/bin/env bash
set -euo pipefail

if ! command -v uvicorn >/dev/null 2>&1; then
  python -m pip install --upgrade pip
  pip install fastapi uvicorn httpx trafilatura simhash feedparser sumy
fi

exec uvicorn shortsearch:app --host 127.0.0.1 --port 8777 --log-level info
```

```bash
chmod +x run-shortsearch.sh
./run-shortsearch.sh
```

---

## 提示詞範例（給 LLM）

> 工具回傳的 JSON 已經是極短摘要。請不要自行瀏覽或再次抓取網頁。
> 只需列出每則「標題｜來源｜日期」，並把 `brief/bul` 合併成不超過 3 句的中文摘要。
> 總回覆限制在 500 字；若超過，優先減少篇數到 3。

---

## 版本

* **v0.5.0**

  * 來源帶屬性（`is_search`），GNews 不再本地過濾
  * `debug=1` 診斷輸出（含落地空 feed）
  * `match=auto|title|any|off`，僅對非搜尋來源生效
  * CJK 友善摘要、標準 UA、GNews 轉址更穩
  * `qcache` 鍵納入 `budget/match`

* v0.2.0（舊）

  * 基礎 CORS、快取、去重與 fallback

---

## 授權

MIT（建議在專案加入 LICENSE 檔）。

---

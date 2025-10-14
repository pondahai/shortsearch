# ShortSearch — 超輕量新聞/網頁搜尋摘要伺服器

一個面向 **Open-WebUI** 的本機 FastAPI 服務：
先在工具端做**去噪、去重、摘要與硬性字數預算**，再把**極短 JSON**回傳給 LLM，徹底降低 token 開銷。

---

## ✨ 功能重點

* **極短輸出（budget）**：每次呼叫按 `budget`（字元上限）裁切，避免爆上下文
* **來源整潔**：RSS 抓取 → Readability 正文抽取（trafilatura）→ **TextRank/前幾句**摘要
* **相似去重**：SimHash 群集，保留代表篇
* **查詢匹配更穩**：標題/內文正規化比對（NFKC、大小寫、空白），並有**兩段 fallback**
* **快取**：`url → 正文`（24h）、`q → 結果`（8h）；**0 筆不寫入快取**
* **CORS 已開**：Open-WebUI 能載入 `/openapi.json`（避免 OPTIONS 405）

---

## 目錄結構

```
shortsearch.py
shortsearch.db          # 首次啟動後產生
```

---

## 系統需求

* Python 3.11+（建議）
* pip 可用的網路環境（首次安裝依賴）
  -（可選）WSL/Ubuntu

---

## 安裝

```bash
# 建議：建立虛擬環境
python -m venv .venv && source .venv/bin/activate

# 基本依賴
pip install fastapi uvicorn httpx trafilatura simhash feedparser

# 可選：更佳的摘要品質（TextRank）
pip install sumy
```

---

## 啟動

方式 A（直接跑檔案）：

```bash
python shortsearch.py
```

方式 B（使用 uvicorn）：

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

1. 打開 **Settings → Tools → Add custom tool**
2. **Name**：`ShortSearch`
3. **OpenAPI URL**：`http://127.0.0.1:8777/openapi.json`
4. 儲存後（必要時按 **Refresh**），工具會註冊 `search_news` 與 `fetch` 兩個端點

> 建議在工具的 Default Args 裡**不要**預設 `n` 太大；推薦 `n=5`、`budget=1000~1200`。

---

## API 一覽

### `GET /health`

回傳服務健康狀態。

```json
{"ok": true}
```

### `GET /search_news`

以關鍵字搜尋來源 RSS，去噪、去重並在 `budget` 內回傳極短摘要。

**Query 參數**

* `q` *(必填)*：關鍵字（支援空白分詞，任一命中即通過）
* `n` *(選填)*：最多返回篇數，**1–8**（預設 5；伺服器端會再夾到 8）
* `lang` *(選填)*：`zh` / `en`（目前僅用於摘要與分句策略，預設 `zh`）
* `budget` *(選填)*：總字元上限，**300–4000**（預設 1200）

**回應範例**

```json
{
  "q": "AI 教育",
  "generated_at": "2025-10-14T14:20:00+08:00",
  "items": [
    {
      "t": "教育部推AI課程試辦",
      "u": "https://example.com/a1",
      "d": "2025-10-14",
      "src": "中央社",
      "brief": "教育部宣布高中階段試辦AI課程，重點在運算思維與倫理。",
      "bul": ["2025年10月", "試辦", "課綱調整"],
      "ents": [],
      "cat": ""
    }
  ],
  "note": "共 3 則；已去重、壓縮；budget=1000"
}
```

**注意**

* 若 `q` 有值但過濾後 **0 筆**：
  會自動 fallback（只看標題匹配）→ 再不行取最高分前 n 則，確保不會回 0。
* **快取規則**：

  * 命中且 `items>0` → 直接回快取
  * `items==0` → 視為未命中，**重新抓**
  * 只在 `items>0` 時寫入快取；否則清除舊條目

### `GET /fetch`

摘要單一 URL（適合你在對話中貼單篇新聞/文章）。
**Query 參數**

* `url` *(必填)*：網頁連結
* `budget` *(選填)*：總字元上限（預設 800）
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
* 只讓 LLM「列出標題/來源/日期 + 2–3 句摘要」，不要再叫模型自己去抓網頁
* 需要「跨多篇深度歸納」時，再用小模型做二次摘要（輸入控制在 600–800 字）

---

## 調整來源

在 `shortsearch.py` 上方的 `SOURCES` 陣列修改你的 RSS 清單，例如：

```python
SOURCES = [
    "https://feedx.net/rss/cna.xml",            # 中央社 (總覽)
    "https://www.ithome.com.tw/rss",              # iThome IT 新聞
    "https://technews.tw/feed/"                   # TechNews 科技新報
    # 你也可加入在地/教育類 RSS
]
```

> 先以 **少量可信來源** 上線，效果通常比大量泛抓更穩、更省 token。

---

## 常見問題（FAQ / Troubleshooting）

**Q1. Open-WebUI 載入 `/openapi.json` 回 405？**
A：已內建 CORS。若還見到 405，請確認你正在跑的是這個版本，或重啟服務。
（`allow_origins=["*"]`、`allow_methods=["GET","OPTIONS"]` 已配置）

**Q2. 工具報 `n` 超過上限（例如 n=10）？**
A：Open-WebUI 可能快取了舊 schema 或工具預設寫了 10。
→ 在 Tools 介面 **Refresh/Reload schema** 或刪除後重加；並把 Default Args 的 `n` 改 ≤ 8。
→ 伺服器端也會強制夾到 8。

**Q3. `search_news` 帶 `q` 總是 0 筆，但不帶 `q` 卻有？**
A：已修正匹配邏輯＆新增 fallback。若仍然全 0：

* 嘗試其他關鍵字（確認來源 RSS 是否真的有該主題）
* 檢查 `SOURCES` 的 RSS 是否暫時無法連線
* 看 `shortsearch.db` 是否可寫（WSL 權限）

**Q4. Open-WebUI 總是讀到舊的 schema？**
A：重啟 shortsearch、瀏覽 `http://127.0.0.1:8777/openapi.json` 確認內容；
在 Tools 裡 **Refresh**，或刪掉重加；必要時清空瀏覽器快取。

**Q5. 需要對外網提供服務？**
A：把啟動改成 `--host 0.0.0.0`，並調整防火牆與 `allow_origins` 白名單。

---

## 安全與隱私

* 僅使用你列的 RSS 來源，不會主動連大型商業搜尋 API
* 本地 SQLite 快取，資料不離機器
* 若需要更嚴格的 CORS，請把 `allow_origins` 換成你的 Open-WebUI 網址

---

## 一鍵啟動（可選）

建立 `run-shortsearch.sh`：

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

* `v0.2.0`

  * CORS、JSONResponse
  * 快取：`items==0` 視為未命中；只在 `items>0` 時寫入
  * 查詢匹配正規化＋雙重 fallback
  * `n` 夾限至 8、`budget` 夾限 300–4000

---

## 授權

MIT（若需加入專案 LICENSE，建議使用 MIT/Apache-2.0 其中之一）。

---

需要我把 README 另外存成檔案或幫你做個最小的 ZIP（含 `shortsearch.py`、README、run 腳本）嗎？

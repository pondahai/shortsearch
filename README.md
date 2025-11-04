
# ShortSearch Improved API

**ShortSearch Improved** 是一個高效、強大的新聞搜尋與摘要 API。它融合了傳統網頁抓取、模擬瀏覽器深度抓取 (Selenium) 以及大型語言模型 (LLM) 摘要等多種技術，旨在為 RAG (Retrieval-Augmented Generation) 應用、聊天機器人或其他自動化工作流程提供高品質、客製化的即時新聞資訊。

此專案基於原始的 `shortsearch` API，並參考了 `line_bot_news` 專案中經過實戰驗證的強大抓取與處理邏輯，進行了全面升級。

## ✨ 功能亮點

- **🚀 現代化 API 框架**: 使用 **FastAPI** 構建，提供高效能的異步處理能力與自動生成的互動式 API 文件 (Swagger UI)。
- **🕷️ 雙層抓取策略**:
  - **`trafilatura` 高速抓取**: 快速解析靜態網頁內容。
  - **`Selenium` 深度後備**: 當靜態抓取失敗或內容不足時，自動啟用模擬瀏覽器進行深度抓取，大幅提高對動態 JavaScript 網站的內容提取成功率。
- **🧠 智慧摘要引擎**:
  - **本地摘要 (`local`)**: 透過 `sumy` 函式庫提供快速的 extractive 摘要，可自訂摘要句子數量。
  - **LLM 摘要 (`llm`)**: 整合 OpenAI 相容的 API (如 Groq, OpenAI)，提供多種風格的 abstractive 摘要，包括：
    - 預設摘要 (`default`)
    - 重點條列 (`bullet`)
    - 問答格式 (`qa`)
- **⚡ 高效能快取**: 內建兩級 SQLite 快取機制，對已完成的**查詢結果**和**單篇文章內容**進行快取，大幅提升重複請求的回應速度。
- **🛡️ 智慧去重**: 使用 `Simhash` 演算法對文章進行指紋比對，有效去除內容高度相似的重複新聞。
- **⚙️ 靈活配置**: 所有關鍵參數（如 API Keys, Selenium 設定）均可透過 `.env` 檔案進行配置，無需修改程式碼。

## 🔧 環境設定與安裝

### 1. 前置需求

- Python 3.8+
- [Google Chrome](https://www.google.com/chrome/) 瀏覽器
- [ChromeDriver](https://googlechromelabs.github.io/chrome-for-testing/)
  - **重要**: 下載的 ChromeDriver 版本必須與您安裝的 Chrome 瀏覽器版本**完全匹配**。
  - 將下載的 `chromedriver.exe` (或 `chromedriver` for Linux/macOS) 放置在系統的 `PATH` 環境變數所包含的路徑下（例如 `C:\Windows` 或 `/usr/local/bin`）。

### 2. 安裝 Python 依賴套件

將專案 clone 到本地後，在專案根目錄下執行以下指令安裝所有必要的函式庫：

```bash
pip install "fastapi[all]" httpx feedparser trafilatura simhash selenium pydantic-settings openai python-dotenv beautifulsoup4 lxml sumy
```

### 3. 配置環境變數

在專案根目錄下建立一個名為 `.env` 的檔案，並填入以下內容。請根據您的需求修改設定值。

```env
# --- LLM 設定 (可使用 OpenAI, Groq 或其他相容 API) ---
# 範例使用 Groq API
OPENAI_API_KEY="gsk_YourGroqApiKeyHere"
OPENAI_BASE_URL="https://api.groq.com/openai/v1"
OPENAI_MODEL="llama3-8b-8192"

# --- Selenium 設定 ---
# 在伺服器上部署時建議設為 "true"
SELENIUM_HEADLESS="true"

# --- (可選) 偵錯模式 ---
# DEBUG_MODE="true"
```

## 🚀 啟動 API 伺服器

在專案根目錄下，執行以下指令啟動伺服器：

```bash
python shortsearch_improved.py
```

伺服器成功啟動後，您會看到類似以下的訊息：
```
INFO:     Uvicorn running on http://0.0.0.0:8777 (Press CTRL+C to quit)
```

現在，您可以開始使用 API 了。

## 📚 API 使用說明

伺服器啟動後，您可以透過瀏覽器訪問 [http://127.0.0.1:8777/docs](http://127.0.0.1:8777/docs) 來查看並使用自動生成的 Swagger UI 互動式文件。

### 端點: `GET /search_news`

#### 基礎查詢

- **查詢 "AI晶片" 的相關新聞 (使用預設設定)**
  ```
  http://127.0.0.1:8777/search_news?q=AI晶片
  ```

#### 進階查詢 (摘要規則)

- **使用本地摘要，並指定摘要 3 句話**
  ```
  http://127.0.0.1:8777/search_news?q=AI晶片&summary_mode=local&summary_sentences=3
  ```

- **使用 LLM 摘要，並要求條列式重點 (`bullet`)**
  ```
  http://127.0.0.1:8777/search_news?q=Nvidia財報&summary_mode=llm&summary_prompt_type=bullet
  ```

- **使用 LLM 摘要，並要求問答格式 (`qa`)**
  ```
  http://127.0.0.1:8777/search_news?q=量子計算突破&summary_mode=llm&summary_prompt_type=qa
  ```

#### 查詢參數詳解

| 參數 | 類型 | 預設值 | 說明 |
| :--- | :--- | :--- | :--- |
| `q` | `string` | **(必填)** | 您想搜尋的新聞關鍵字。 |
| `n` | `integer` | `5` | 希望返回的最大新聞則數 (1-10)。 |
| `budget` | `integer` | `1200` | 回傳內容的總字元數上限。 |
| `site` | `string` | `null` | (可選) 將搜尋範圍限制在特定網站 (例如 `ithome.com.tw`)。 |
| `summary_mode` | `string` | `local` | 摘要模式。可選值: `local` (本地快速摘要) 或 `llm` (大型語言模型摘要)。 |
| `summary_sentences` | `integer` | `2` | **[本地摘要模式]** 指定摘要的句子數量 (1-5)。 |
| `summary_prompt_type`| `string` | `default` | **[LLM摘要模式]** 指定摘要風格。可選值: `default`, `bullet`, `qa`。 |

### 範例回傳結果

```json
{
  "q": "Nvidia財報",
  "generated_at": "2023-11-04T12:30:00.123456+08:00",
  "items": [
    {
      "t": "Nvidia 財報再超預期，資料中心業務成主要引擎",
      "u": "https://technews.tw/2023/11/04/nvidia-earnings-report/",
      "d": "2023-11-04",
      "src": "TechNews 科技新報",
      "brief": "- Nvidia 公布最新季度財報，營收與利潤均大幅超越分析師預期。\n- 資料中心業務營收同比增長超過 150%，成為公司最主要的增長動力。\n- 公司對下一季度的展望保持樂觀，預計 AI 晶片需求將持續強勁。",
      "bul": [
        "財報",
        "資料中心",
        "AI 晶片"
      ]
    }
  ],
  "note": "Returned 1 items. Summary mode: llm."
}
```

---

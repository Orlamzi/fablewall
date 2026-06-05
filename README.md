# VerifyIt — News Fact-Checking and Authenticity Verification Platform

A web-based platform that allows users to verify news claims by cross-referencing them against real fact-check databases maintained by credible organisations worldwide. Built for the Final Year Project, Department of Information Technology, University of Ilorin.

---

## How It Works

1. User pastes a news headline or claim into the platform
2. The backend queries the **Google Fact Check Tools API** — a database of verified fact-checks from organisations like PolitiFact, Snopes, AFP Fact Check, Reuters Fact Check, and others
3. The backend simultaneously queries **NewsAPI** to show whether the story appears in credible news outlets
4. The frontend displays the real verdicts from human fact-checkers, along with source links
5. An overall credibility label (Likely True / Likely False / Mixed / Unverified) is derived from the returned ratings

**Important:** This system does NOT use any AI or language model to generate verdicts. All results come from human fact-checkers at credible organisations.

---

## Project Structure

```
verifyit/
├── app.py              # Flask backend server
├── index.html          # Frontend (open directly in browser)
├── requirements.txt    # Python dependencies
├── .env.example        # API key template
├── .env                # Your actual API keys (DO NOT commit this)
└── README.md
```

---

## Getting Your API Keys

### 1. Google Fact Check Tools API (Free)

1. Go to [https://console.cloud.google.com](https://console.cloud.google.com)
2. Create a new project (or select an existing one)
3. In the left menu, go to **APIs & Services → Library**
4. Search for **"Fact Check Tools API"** and click **Enable**
5. Go to **APIs & Services → Credentials**
6. Click **Create Credentials → API Key**
7. Copy the generated key

### 2. NewsAPI Key (Free)

1. Go to [https://newsapi.org](https://newsapi.org)
2. Click **Get API Key** and sign up for a free account
3. Your API key will be shown on your dashboard
4. Copy it

---

## Setup and Running

### Step 1 — Clone or download the project files

Place all files in one folder on your computer.

### Step 2 — Create your .env file

Copy `.env.example` to a new file called `.env` and fill in your keys:

```
GOOGLE_API_KEY=paste_your_google_key_here
NEWS_API_KEY=paste_your_newsapi_key_here
```

### Step 3 — Install Python dependencies

Make sure you have Python 3.8+ installed. Then run:

```bash
pip install -r requirements.txt
```

### Step 4 — Start the backend server

```bash
python app.py
```

You should see:
```
* Running on http://127.0.0.1:5000
```

Leave this terminal open while using the system.

### Step 5 — Open the frontend

Open `index.html` in your browser. You can do this by:
- Double-clicking the file, OR
- Dragging it into your browser window

The frontend will automatically connect to `http://localhost:5000`.

---

## Testing the System

Try these sample claims to verify the system works:

- `"COVID-19 vaccines contain microchips"`
- `"Climate change is not caused by humans"`
- `"The moon landing was faked"`
- `"5G causes coronavirus"`

These are well-known claims that have been fact-checked by multiple organisations and should return results.

---

## Deployment

### Frontend — Vercel (Free)

1. Go to [https://vercel.com](https://vercel.com) and sign up
2. Upload the `index.html` file or connect your GitHub repo
3. Before deploying, update the `API_BASE` constant in `index.html` from `http://localhost:5000` to your deployed backend URL

### Backend — Render (Free)

1. Go to [https://render.com](https://render.com) and sign up
2. Create a new **Web Service** and connect your GitHub repo
3. Set **Build Command** to: `pip install -r requirements.txt`
4. Set **Start Command** to: `python app.py`
5. Add your environment variables (`GOOGLE_API_KEY` and `NEWS_API_KEY`) in the Render dashboard under **Environment**

---

## API Endpoint Reference

### POST /verify

**Request body:**
```json
{
  "claim": "The news headline or claim to verify"
}
```

**Response:**
```json
{
  "claim": "The original claim submitted",
  "credibility": {
    "label": "Likely False",
    "description": "Fact-checkers have rated this claim as false or misleading.",
    "color": "red"
  },
  "fact_checks": [
    {
      "claim_text": "The exact claim that was checked",
      "claimant": "Who made the claim",
      "claim_date": "2024-01-15T00:00:00Z",
      "publisher": "PolitiFact",
      "rating": "False",
      "title": "Title of the fact-check article",
      "url": "https://link-to-full-fact-check.com"
    }
  ],
  "news_coverage": [
    {
      "title": "Article headline",
      "source": "Reuters",
      "url": "https://link-to-article.com",
      "published_at": "2024-01-16T10:30:00Z",
      "description": "Article summary"
    }
  ],
  "total_fact_checks": 3
}
```

### GET /health

Returns `{"status": "ok"}` — used to confirm the server is running.

---

## Credibility Labels Explained

| Label | Meaning |
|---|---|
| **Likely False** | At least one fact-checker has rated the claim as false, misleading, or fabricated |
| **Likely True** | Fact-checkers have confirmed the claim as accurate or mostly accurate |
| **Mixed or Disputed** | Fact-checkers have given conflicting or nuanced ratings |
| **Unverified** | No matching fact-checks found in the database |

---

## Limitations

- The system can only verify claims that have already been fact-checked by a credible organisation and indexed by Google
- Very recent or local news claims may not yet be in the fact-check database
- The free NewsAPI plan only returns results for articles from the past month
- The system currently supports English-language claims only

---

## Built With

- **Frontend:** HTML5, CSS3, Vanilla JavaScript
- **Backend:** Python, Flask, Flask-CORS
- **Primary Data Source:** Google Fact Check Tools API
- **Secondary Data Source:** NewsAPI
- **Deployment:** Vercel (frontend) + Render (backend)

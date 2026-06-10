from flask import Flask, request, jsonify, render_template_string, redirect, session
from flask_cors import CORS
import requests
import os
import json
import pickle
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pymongo import MongoClient
from datetime import datetime
from functools import wraps

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "fablewall-secret-2025")
CORS(app)

GOOGLE_API_KEY    = os.getenv("GOOGLE_API_KEY")
NEWS_API_KEY      = os.getenv("NEWS_API_KEY")
GEMINI_API_KEY    = os.getenv("GEMINI_API_KEY")
MONGO_URI         = os.getenv("MONGO_URI")
ADMIN_PASSWORD    = os.getenv("ADMIN_PASSWORD", "fablewall2025")

FALSE_KEYWORDS = [
    "false", "misleading", "pants on fire", "mostly false",
    "incorrect", "fabricated", "inaccurate", "fake", "wrong",
    "disputed", "debunked", "unfounded", "unsubstantiated"
]
TRUE_KEYWORDS = [
    "true", "mostly true", "correct", "accurate", "verified",
    "confirmed", "factual", "legitimate", "supported"
]

# MongoDB connection
mongo_client = None
db           = None
claims_col   = None

def connect_mongodb():
    global mongo_client, db, claims_col
    try:
        mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        mongo_client.server_info()
        db         = mongo_client["fablewall"]
        claims_col = db["claims"]
        print("MongoDB connected successfully.")
        seed_initial_claims()
    except Exception as e:
        print(f"MongoDB connection failed: {e}. Falling back to local JSON.")
        mongo_client = None

def seed_initial_claims():
    if claims_col.count_documents({}) == 0:
        try:
            db_path = os.path.join(os.path.dirname(__file__), "local_claims.json")
            if os.path.exists(db_path):
                with open(db_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                claims = data.get("claims", [])
                if claims:
                    claims_col.insert_many(claims)
                    print(f"Seeded {len(claims)} claims from local_claims.json to MongoDB.")
        except Exception as e:
            print(f"Could not seed initial claims: {e}")

def load_local_claims():
    if claims_col is not None:
        try:
            claims = list(claims_col.find({}, {"_id": 0}))
            return claims
        except Exception as e:
            print(f"MongoDB read error: {e}")
    try:
        db_path = os.path.join(os.path.dirname(__file__), "local_claims.json")
        with open(db_path, "r", encoding="utf-8") as f:
            return json.load(f).get("claims", [])
    except Exception as e:
        print(f"Could not load local claims: {e}")
        return []

def save_claim_to_db(new_claim):
    if claims_col is not None:
        try:
            claims_col.insert_one({**new_claim, "_id": None})
            claims_col.update_one(
                {"claim_text": new_claim["claim_text"]},
                {"$set": new_claim},
                upsert=True
            )
            return True
        except Exception as e:
            print(f"MongoDB write error: {e}")
    try:
        db_path = os.path.join(os.path.dirname(__file__), "local_claims.json")
        with open(db_path, "r", encoding="utf-8") as f:
            db_data = json.load(f)
        db_data["claims"].append(new_claim)
        with open(db_path, "w", encoding="utf-8") as f:
            json.dump(db_data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"JSON fallback write error: {e}")
        return False

# ML Models
tfidf_vectorizer     = None
naive_bayes_pipeline = None
ml_trained           = False
model_accuracy       = None
MODEL_PATH           = os.path.join(os.path.dirname(__file__), "fablewall_model.pkl")

def train_ml_models():
    global tfidf_vectorizer, naive_bayes_pipeline, ml_trained, model_accuracy

    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "rb") as f:
                saved = pickle.load(f)
            tfidf_vectorizer     = saved["tfidf"]
            naive_bayes_pipeline = saved["pipeline"]
            model_accuracy       = saved["accuracy"]
            ml_trained           = True
            print(f"ML model loaded from file. Accuracy: {model_accuracy}%")
            return
        except Exception as e:
            print(f"Could not load saved model: {e}. Retraining.")

    texts  = []
    labels = []

    dataset_path = os.path.join(os.path.dirname(__file__), "WELFake_Dataset.csv")
    if os.path.exists(dataset_path):
        try:
            print("Loading WELFake dataset...")
            df = pd.read_csv(dataset_path)
            df = df.dropna(subset=["label", "title", "text"])
            df["content"] = df["title"].fillna("") + " " + df["text"].fillna("")
            df = df[df["content"].str.strip() != ""]
            df["label"] = df["label"].astype(int)
            df = df[df["label"].isin([0, 1])]
            if len(df) > 20000:
                df = df.sample(n=20000, random_state=42)
            texts  = df["content"].tolist()
            labels = df["label"].tolist()
            print(f"WELFake loaded: {len(texts)} articles.")
        except Exception as e:
            print(f"Could not load WELFake: {e}")

    local_claims = load_local_claims()
    local_texts  = []
    local_labels = []
    for entry in local_claims:
        claim_text = entry.get("claim_text", "")
        verdict    = entry.get("verdict", "").lower()
        if not claim_text:
            continue
        if any(kw in verdict for kw in ["true", "genuine", "confirmed"]):
            local_texts.append(claim_text)
            local_labels.append(1)
        elif any(kw in verdict for kw in ["false", "fake", "misleading"]):
            local_texts.append(claim_text)
            local_labels.append(0)

    if local_texts:
        boost = 10
        for _ in range(boost):
            texts  += local_texts
            labels += local_labels
        print(f"Local claims added: {len(local_texts)} entries (boosted x{boost}).")

    if len(texts) < 2:
        print("Not enough data to train ML models.")
        ml_trained = False
        return

    try:
        tfidf_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), stop_words="english", max_features=10000
        )
        tfidf_vectorizer.fit(texts)

        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )
        naive_bayes_pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=10000)),
            ("nb",    MultinomialNB(alpha=0.1))
        ])
        naive_bayes_pipeline.fit(X_train, y_train)
        y_pred         = naive_bayes_pipeline.predict(X_test)
        model_accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)
        ml_trained     = True
        print(f"Naive Bayes trained. Accuracy: {model_accuracy}%")

        with open(MODEL_PATH, "wb") as f:
            pickle.dump({
                "tfidf":    tfidf_vectorizer,
                "pipeline": naive_bayes_pipeline,
                "accuracy": model_accuracy
            }, f)
        print("ML model saved to file.")

    except Exception as e:
        print(f"ML training error: {e}")
        ml_trained = False


def get_tfidf_similarity_matches(claim, threshold=0.55):
    claims = load_local_claims()
    if not claims or not ml_trained or tfidf_vectorizer is None:
        return []
    texts = [entry.get("claim_text", "") for entry in claims]
    if not any(texts):
        return []
    try:
        claim_vec  = tfidf_vectorizer.transform([claim])
        corpus_vec = tfidf_vectorizer.transform(texts)
        scores     = cosine_similarity(claim_vec, corpus_vec)[0]
        matches    = []
        for idx, score in enumerate(scores):
            if score >= threshold:
                entry = claims[idx]
                matches.append({
                    "claim_text":       entry.get("claim_text", ""),
                    "claimant":         "FableWall Local Database",
                    "claim_date":       entry.get("date_added", ""),
                    "publisher":        entry.get("source", "FableWall Local Database"),
                    "publisher_site":   "",
                    "rating":           entry.get("rating", ""),
                    "title":            entry.get("explanation", ""),
                    "url":              entry.get("url", "#"),
                    "review_date":      entry.get("date_added", ""),
                    "source":           "local",
                    "similarity_score": round(float(score), 3)
                })
        matches.sort(key=lambda x: x["similarity_score"], reverse=True)
        return matches
    except Exception as e:
        print(f"TF-IDF error: {e}")
        return []


def get_naive_bayes_prediction(claim):
    if not naive_bayes_pipeline or not ml_trained:
        return None, None
    try:
        proba      = naive_bayes_pipeline.predict_proba([claim])[0]
        prediction = naive_bayes_pipeline.predict([claim])[0]
        confidence = round(float(max(proba)) * 100, 1)
        label      = "Likely Genuine" if prediction == 1 else "Likely Fake"
        return label, confidence
    except Exception as e:
        print(f"Naive Bayes error: {e}")
        return None, None


def determine_credibility(fact_checks, local_matches, nb_prediction=None, nb_confidence=None):
    if local_matches:
        ratings     = [fc.get("rating", "").lower() for fc in local_matches]
        false_count = sum(1 for r in ratings if any(kw in r for kw in FALSE_KEYWORDS))
        true_count  = sum(1 for r in ratings if any(kw in r for kw in TRUE_KEYWORDS))
        if true_count > 0 and false_count == 0:
            return {"label": "Likely Genuine", "description": "Verified by FableWall local database using TF-IDF similarity matching.", "color": "green"}
        elif false_count > 0 and true_count == 0:
            return {"label": "Likely Fake", "description": "Flagged by FableWall local database using TF-IDF similarity matching.", "color": "red"}
        else:
            return {"label": "Unverified", "description": "Found in local database but carries no definitive rating.", "color": "grey"}

    if fact_checks:
        ratings     = [fc.get("rating", "").lower() for fc in fact_checks]
        false_count = sum(1 for r in ratings if any(kw in r for kw in FALSE_KEYWORDS))
        true_count  = sum(1 for r in ratings if any(kw in r for kw in TRUE_KEYWORDS))
        if false_count > true_count:
            return {"label": "Likely Fake",       "description": "Fact-checkers have rated this claim as false or misleading.",                  "color": "red"}
        elif true_count > false_count:
            return {"label": "Likely Genuine",    "description": "Fact-checkers have verified this claim as accurate or mostly accurate.",        "color": "green"}
        elif false_count > 0 or true_count > 0:
            return {"label": "Mixed or Disputed", "description": "Fact-checkers have given this claim conflicting or mixed ratings.",             "color": "orange"}

    if nb_prediction and nb_confidence and float(nb_confidence) >= 70.0:
        return {
            "label":       nb_prediction,
            "description": f"No fact-check records found. Naive Bayes classifier predicted {nb_prediction} with {nb_confidence}% confidence based on text patterns.",
            "color":       "green" if nb_prediction == "Likely Genuine" else "red"
        }

    return {"label": "Unverified", "description": "No fact-check records were found for this claim in any database. This does not mean the claim is false. It may not yet have been formally investigated.", "color": "grey"}


def generate_ai_summary(claim, verdict_label, fact_checks, local_matches):
    all_checks = local_matches + fact_checks
    if not all_checks:
        return None, "No evidence available to summarise."

    evidence_lines = []
    for fc in all_checks[:5]:
        publisher = fc.get("publisher", "Unknown")
        rating    = fc.get("rating", "")
        claim_txt = fc.get("claim_text", "")
        source    = "FableWall local database" if fc.get("source") == "local" else publisher
        evidence_lines.append(f"- {source} rated this as '{rating}': \"{claim_txt}\"")

    evidence_text = "\n".join(evidence_lines)
    prompt = f"""You are a fact-checking assistant helping general users understand news verification results.

A user submitted this claim: "{claim}"
The system verdict: {verdict_label}
Retrieved fact-check records:
{evidence_text}

Write a clear, plain-English summary (3 to 5 sentences) explaining what the evidence says.
Only summarise the retrieved evidence. Do not form your own verdict.
Mention organisations by name where relevant.
Keep language simple and accessible.
Do not start with Based on or According to.
Do not add AI disclaimers."""

    if GEMINI_API_KEY:
        try:
            url     = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "safetySettings": [
                    {"category": "HARM_CATEGORY_HARASSMENT",        "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH",       "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
                ],
                "generationConfig": {"maxOutputTokens": 300, "temperature": 0.3}
            }
            response = requests.post(url, headers={"Content-Type": "application/json"}, json=payload, timeout=60)
            data     = response.json()
            if not data.get("promptFeedback", {}).get("blockReason"):
                candidates = data.get("candidates", [])
                if candidates and candidates[0].get("finishReason") != "SAFETY":
                    summary = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()
                    if summary:
                        return summary, None
        except Exception:
            pass

    return generate_rule_based_summary(claim, verdict_label, all_checks), None


def generate_rule_based_summary(claim, verdict_label, all_checks):
    publishers = list({
        fc.get("publisher", "") for fc in all_checks
        if fc.get("publisher") and fc.get("publisher") not in
        ["Unknown", "Unknown Publisher", "Local Database", "FableWall Local Database"]
    })
    local   = [fc for fc in all_checks if fc.get("source") == "local"]
    ratings = [fc.get("rating", "") for fc in all_checks if fc.get("rating")]
    lines   = []

    if verdict_label == "Likely Fake":
        lines.append(f"This claim has been rated as false or misleading by {', '.join(publishers[:3])}." if publishers else "This claim has been rated as false or misleading by fact-checking organisations.")
        if ratings:
            lines.append(f"Ratings include: {', '.join(list(set(ratings[:3])))}.")
        lines.append("Users are advised to treat this claim with caution and consult the source links for full details.")
    elif verdict_label == "Likely Genuine":
        if local:
            lines.append("This claim has been manually verified and confirmed in the FableWall local database.")
        if publishers:
            lines.append(f"Fact-checking organisations including {', '.join(publishers[:3])} have reviewed and confirmed this claim.")
        if not local and not publishers:
            lines.append("Fact-checkers have reviewed this claim and found it to be accurate or mostly accurate.")
        lines.append("Users can follow the source links below to read the full investigations.")
    elif verdict_label == "Mixed or Disputed":
        lines.append(f"This claim has received mixed ratings from {', '.join(publishers[:3])}." if publishers else "This claim has received mixed ratings from fact-checking organisations.")
        lines.append("Users should read the full source articles to form their own informed judgement.")
    else:
        lines.append("No matching fact-check records were found for this claim in the available databases.")
        lines.append("This does not necessarily mean the claim is false. It may not yet have been formally investigated.")
        lines.append("Users are encouraged to search trusted news sources before sharing this claim.")

    return " ".join(lines)


def parse_fact_checks(data):
    results = []
    for claim in data.get("claims", []):
        for review in claim.get("claimReview", []):
            results.append({
                "claim_text":     claim.get("text", "N/A"),
                "claimant":       claim.get("claimant", "Unknown"),
                "claim_date":     claim.get("claimDate", ""),
                "publisher":      review.get("publisher", {}).get("name", "Unknown Publisher"),
                "publisher_site": review.get("publisher", {}).get("site", ""),
                "rating":         review.get("textualRating", "Unrated"),
                "title":          review.get("title", ""),
                "url":            review.get("url", "#"),
                "review_date":    review.get("reviewDate", ""),
                "source":         "google"
            })
    return results


def parse_news(data):
    results = []
    for article in data.get("articles", [])[:3]:
        results.append({
            "title":        article.get("title", ""),
            "source":       article.get("source", {}).get("name", "Unknown"),
            "url":          article.get("url", "#"),
            "published_at": article.get("publishedAt", ""),
            "description":  article.get("description", "")
        })
    return results


# ── Admin authentication ──
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("admin_logged_in"):
            return redirect("/admin/login")
        return f(*args, **kwargs)
    return decorated


ADMIN_LOGIN_HTML = """
<!DOCTYPE html>
<html>
<head>
  <title>FableWall Admin</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: sans-serif; background: #fdf8f3; display: flex; align-items: center; justify-content: center; min-height: 100vh; }
    .card { background: #fff; border-radius: 16px; padding: 40px; width: 100%; max-width: 400px; box-shadow: 0 8px 32px rgba(59,31,14,0.12); }
    h2 { font-size: 22px; color: #3b1f0e; margin-bottom: 8px; }
    p  { font-size: 14px; color: #9a8a7a; margin-bottom: 24px; }
    input { width: 100%; padding: 12px 16px; border: 2px solid #ede5da; border-radius: 10px; font-size: 14px; margin-bottom: 16px; }
    input:focus { outline: none; border-color: #9b5a2f; }
    button { width: 100%; padding: 14px; background: #3b1f0e; color: #fff; border: none; border-radius: 10px; font-size: 14px; font-weight: 600; cursor: pointer; }
    button:hover { background: #6b3a1f; }
    .error { color: #b83232; font-size: 13px; margin-bottom: 12px; }
  </style>
</head>
<body>
  <div class="card">
    <h2>FableWall Admin</h2>
    <p>Sign in to manage the claims database</p>
    {% if error %}<div class="error">{{ error }}</div>{% endif %}
    <form method="POST">
      <input type="password" name="password" placeholder="Enter admin password" required />
      <button type="submit">Sign In</button>
    </form>
  </div>
</body>
</html>
"""

ADMIN_DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
  <title>FableWall Admin Dashboard</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: sans-serif; background: #fdf8f3; color: #3b1f0e; }
    header { background: #3b1f0e; padding: 16px 24px; display: flex; align-items: center; justify-content: space-between; }
    header span { color: #fff; font-size: 18px; font-weight: 700; }
    header a { color: rgba(255,255,255,0.6); font-size: 13px; text-decoration: none; }
    header a:hover { color: #fff; }
    main { max-width: 900px; margin: 0 auto; padding: 32px 24px; }
    h2 { font-size: 20px; margin-bottom: 20px; }
    .form-card { background: #fff; border-radius: 16px; padding: 32px; box-shadow: 0 4px 16px rgba(59,31,14,0.08); margin-bottom: 32px; border: 1px solid #ede5da; }
    .form-row { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px; }
    label { display: block; font-size: 12px; font-weight: 600; color: #5a4a3a; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px; }
    input, textarea, select { width: 100%; padding: 12px 14px; border: 2px solid #ede5da; border-radius: 10px; font-family: sans-serif; font-size: 14px; color: #3b1f0e; background: #fdf8f3; }
    input:focus, textarea:focus, select:focus { outline: none; border-color: #9b5a2f; background: #fff; }
    textarea { min-height: 80px; resize: vertical; }
    .full { grid-column: 1 / -1; }
    .btn { padding: 14px 28px; background: #3b1f0e; color: #fff; border: none; border-radius: 10px; font-size: 14px; font-weight: 600; cursor: pointer; }
    .btn:hover { background: #6b3a1f; }
    .success { background: #eef7f2; border: 1px solid rgba(42,122,74,0.3); color: #2a7a4a; padding: 12px 16px; border-radius: 8px; font-size: 13px; margin-bottom: 20px; }
    .claims-list { background: #fff; border-radius: 16px; box-shadow: 0 4px 16px rgba(59,31,14,0.08); overflow: hidden; border: 1px solid #ede5da; }
    .claim-item { padding: 18px 24px; border-bottom: 1px solid #f0ebe5; }
    .claim-item:last-child { border-bottom: none; }
    .claim-top { display: flex; justify-content: space-between; align-items: flex-start; gap: 12px; margin-bottom: 6px; }
    .claim-text { font-size: 14px; font-weight: 500; flex: 1; }
    .badge { padding: 3px 10px; border-radius: 100px; font-size: 11px; font-weight: 600; white-space: nowrap; }
    .badge-fake { background: #fdf0f0; color: #b83232; }
    .badge-genuine { background: #eef7f2; color: #2a7a4a; }
    .badge-unverified { background: #f0ebe5; color: #6a5a4a; }
    .claim-meta { font-size: 12px; color: #9a8a7a; }
    .stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-bottom: 32px; }
    .stat-card { background: #fff; border-radius: 12px; padding: 20px; text-align: center; border: 1px solid #ede5da; }
    .stat-num { font-size: 32px; font-weight: 700; color: #3b1f0e; }
    .stat-label { font-size: 12px; color: #9a8a7a; margin-top: 4px; }
    @media (max-width: 600px) { .form-row { grid-template-columns: 1fr; } .stats { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
<header>
  <span>FableWall Admin</span>
  <a href="/admin/logout">Sign Out</a>
</header>
<main>

  {% if success %}
  <div class="success">Claim added successfully to the database.</div>
  {% endif %}

  <div class="stats">
    <div class="stat-card">
      <div class="stat-num">{{ total }}</div>
      <div class="stat-label">Total Claims</div>
    </div>
    <div class="stat-card">
      <div class="stat-num">{{ fake_count }}</div>
      <div class="stat-label">Flagged as Fake</div>
    </div>
    <div class="stat-card">
      <div class="stat-num">{{ genuine_count }}</div>
      <div class="stat-label">Confirmed Genuine</div>
    </div>
  </div>

  <h2>Add New Claim</h2>
  <div class="form-card">
    <form method="POST" action="/admin/add">
      <div class="form-row">
        <div>
          <label>Claim Text</label>
          <input type="text" name="claim_text" placeholder="The full claim as stated" required />
        </div>
        <div>
          <label>Verdict</label>
          <select name="verdict">
            <option value="Likely False">Likely False</option>
            <option value="Likely True">Likely True</option>
            <option value="Unverified">Unverified</option>
            <option value="Mixed or Disputed">Mixed or Disputed</option>
          </select>
        </div>
      </div>
      <div class="form-row">
        <div>
          <label>Rating (short label)</label>
          <input type="text" name="rating" placeholder="e.g. False, Confirmed, Disputed" required />
        </div>
        <div>
          <label>Source</label>
          <input type="text" name="source" placeholder="e.g. BBC News, NBS Nigeria" required />
        </div>
      </div>
      <div class="form-row">
        <div class="full">
          <label>Explanation</label>
          <textarea name="explanation" placeholder="Brief explanation of the finding" required></textarea>
        </div>
      </div>
      <div class="form-row">
        <div>
          <label>Keywords (comma separated)</label>
          <input type="text" name="keywords" placeholder="e.g. buhari, dead, death" required />
        </div>
        <div>
          <label>Source URL (optional)</label>
          <input type="text" name="url" placeholder="https://..." />
        </div>
      </div>
      <button type="submit" class="btn">Add Claim to Database</button>
    </form>
  </div>

  <h2>All Claims ({{ total }})</h2>
  <div class="claims-list">
    {% for claim in claims %}
    <div class="claim-item">
      <div class="claim-top">
        <div class="claim-text">{{ claim.claim_text }}</div>
        <span class="badge {% if 'false' in claim.verdict.lower() or 'fake' in claim.verdict.lower() %}badge-fake{% elif 'true' in claim.verdict.lower() or 'genuine' in claim.verdict.lower() %}badge-genuine{% else %}badge-unverified{% endif %}">
          {{ claim.verdict }}
        </span>
      </div>
      <div class="claim-meta">{{ claim.source }} &middot; Added {{ claim.date_added }}</div>
    </div>
    {% endfor %}
  </div>

</main>
</body>
</html>
"""


@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    error = None
    if request.method == "POST":
        if request.form.get("password") == ADMIN_PASSWORD:
            session["admin_logged_in"] = True
            return redirect("/admin")
        error = "Incorrect password. Please try again."
    return render_template_string(ADMIN_LOGIN_HTML, error=error)


@app.route("/admin/logout")
def admin_logout():
    session.pop("admin_logged_in", None)
    return redirect("/admin/login")


@app.route("/admin")
@login_required
def admin_dashboard():
    claims        = load_local_claims()
    total         = len(claims)
    fake_count    = sum(1 for c in claims if any(kw in c.get("verdict","").lower() for kw in ["false","fake","misleading"]))
    genuine_count = sum(1 for c in claims if any(kw in c.get("verdict","").lower() for kw in ["true","genuine","confirmed"]))
    success       = request.args.get("success")
    return render_template_string(
        ADMIN_DASHBOARD_HTML,
        claims=claims, total=total,
        fake_count=fake_count, genuine_count=genuine_count,
        success=success
    )


@app.route("/admin/add", methods=["POST"])
@login_required
def admin_add_claim():
    keywords_raw = request.form.get("keywords", "")
    keywords     = [k.strip().lower() for k in keywords_raw.split(",") if k.strip()]
    new_claim    = {
        "keywords":    keywords,
        "claim_text":  request.form.get("claim_text", "").strip(),
        "verdict":     request.form.get("verdict", "Unverified"),
        "rating":      request.form.get("rating", "").strip(),
        "explanation": request.form.get("explanation", "").strip(),
        "source":      request.form.get("source", "").strip(),
        "url":         request.form.get("url", "").strip(),
        "date_added":  datetime.now().strftime("%Y-%m-%d")
    }
    save_claim_to_db(new_claim)
    return redirect("/admin?success=1")


@app.route("/verify", methods=["POST"])
def verify():
    data  = request.get_json()
    claim = data.get("claim", "").strip()

    if not claim:
        return jsonify({"error": "No claim provided."}), 400
    if len(claim) < 15:
        return jsonify({"error": "Please enter a full news headline or claim, not just a single word or phrase."}), 400

    local_matches                = get_tfidf_similarity_matches(claim)
    nb_prediction, nb_confidence = get_naive_bayes_prediction(claim)

    fact_check_results = []
    fact_check_error   = None
    try:
        fc_response = requests.get(
            "https://factchecktools.googleapis.com/v1alpha1/claims:search",
            params={"query": claim, "key": GOOGLE_API_KEY, "languageCode": "en"},
            timeout=10
        )
        fc_data = fc_response.json()
        if "error" in fc_data:
            fact_check_error = fc_data["error"].get("message", "Google API error")
        else:
            fact_check_results = parse_fact_checks(fc_data)
    except requests.exceptions.Timeout:
        fact_check_error = "Fact-check API timed out."
    except Exception as e:
        fact_check_error = str(e)

    credibility          = determine_credibility(fact_check_results, local_matches, nb_prediction, nb_confidence)
    ai_summary, ai_error = generate_ai_summary(claim, credibility["label"], fact_check_results, local_matches)

    news_results = []
    news_error   = None
    try:
        news_response = requests.get(
            "https://newsapi.org/v2/everything",
            params={"q": claim[:100], "sortBy": "relevance", "pageSize": 3,
                    "apiKey": NEWS_API_KEY, "language": "en"},
            timeout=20
        )
        news_data = news_response.json()
        if news_data.get("status") == "ok":
            news_results = parse_news(news_data)
        else:
            news_error = news_data.get("message", "NewsAPI error")
    except requests.exceptions.Timeout:
        news_error = "News coverage API timed out."
    except Exception as e:
        news_error = str(e)

    return jsonify({
        "claim":               claim,
        "credibility":         credibility,
        "fact_checks":         fact_check_results,
        "local_matches":       local_matches,
        "nb_prediction":       nb_prediction,
        "nb_confidence":       nb_confidence,
        "ai_summary":          ai_summary,
        "ai_error":            ai_error,
        "news_coverage":       news_results,
        "fact_check_error":    fact_check_error,
        "news_error":          news_error,
        "total_fact_checks":   len(fact_check_results),
        "total_local_matches": len(local_matches),
        "ml_trained":          ml_trained,
        "model_accuracy":      model_accuracy
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":                 "ok",
        "service":                "FableWall Backend",
        "local_database_entries": len(load_local_claims()),
        "ml_trained":             ml_trained,
        "model_accuracy":         model_accuracy,
        "mongodb_connected":      mongo_client is not None
    })

@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status": "awake"})

@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "FableWall backend is running. Use POST /verify to check a claim."})


connect_mongodb()
train_ml_models()

if __name__ == "__main__":
    app.run(debug=True, port=5000)

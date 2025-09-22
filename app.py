# app.py
import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import random
import os

# -------------------------
# Config / paths
# -------------------------
DATA_PATH = "training_data.csv"   # ✅ Load from repo
DB_PATH = "interviewer_logs.db"

# -------------------------
# Helpers
# -------------------------
@st.cache_data
def load_qa_dataframe(path=DATA_PATH):
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        st.error("❌ training_data.csv not found in repo. Please upload below.")
        uploaded = st.file_uploader("Upload training_data.csv", type=["csv"])
        if uploaded is not None:
            df = pd.read_csv(uploaded)
        else:
            st.stop()

    # Split "Question: ... Answer: ..." format
    qa = df["text"].str.extract(r"Question:\s*(.*?)\s*Answer:\s*(.*)")
    qa.columns = ["question", "answer"]
    qa = qa.dropna().reset_index(drop=True)
    return qa

def init_db(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS responses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            question TEXT,
            expected_answer TEXT,
            user_answer TEXT,
            score REAL,
            method TEXT
        )
        """
    )
    conn.commit()
    return conn

def save_response(conn, question, expected_answer, user_answer, score, method="keyword"):
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO responses (timestamp, question, expected_answer, user_answer, score, method) VALUES (?, ?, ?, ?, ?, ?)",
        (datetime.utcnow().isoformat(), question, expected_answer, user_answer, float(score), method),
    )
    conn.commit()

# Normalize text
def normalize_text(s):
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# TF-IDF similarity
def evaluate_similarity(expected, user):
    expected_norm = normalize_text(expected)
    user_norm = normalize_text(user)
    vec = TfidfVectorizer().fit_transform([expected_norm, user_norm])
    score = cosine_similarity(vec[0:1], vec[1:2])[0][0]
    return float(score * 100)  # scale to 0-100

# Keyword overlap
def keyword_overlap_score(expected, user, top_n=8):
    expected_norm = normalize_text(expected)
    tokens = expected_norm.split()
    stop = set(["the","a","an","and","is","are","of","in","to","for","with","that","this"])
    tokens = [t for t in tokens if t not in stop]
    key_terms = list(dict.fromkeys(tokens))[:top_n]
    user_norm = normalize_text(user)
    hits = sum(1 for t in key_terms if t in user_norm)
    if len(key_terms) == 0:
        return 0.0
    return float(hits / len(key_terms) * 100)

# Combined score
def combined_score(expected, user, w_sim=0.7, w_kw=0.3):
    sim = evaluate_similarity(expected, user)
    kw = keyword_overlap_score(expected, user)
    return w_sim * sim + w_kw * kw

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="AI/Data-Science Interviewer Bot", layout="centered")

st.title("🤖 AI & Data Science Interviewer Bot")
st.markdown("Practice mock interview Q&A from your dataset. Scores are estimated using text similarity + keyword overlap.")

# Load dataset
qa = load_qa_dataframe()
conn = init_db()

# Sidebar
st.sidebar.header("⚙️ Settings")
mode = st.sidebar.selectbox("Mode", ["Random questions", "Topic search", "Browse list"])
num_questions = st.sidebar.slider("Number of questions (session)", 1, 10, 3)
evaluation_method = st.sidebar.selectbox("Evaluation method", ["combined", "similarity", "keywords"])
seed = st.sidebar.number_input("Random seed", value=42, step=1)
random.seed(seed)

# Session state
if "remaining_qs" not in st.session_state:
    st.session_state.remaining_qs = []

if mode == "Random questions":
    if st.button("🎲 Start random session"):
        st.session_state.remaining_qs = random.sample(list(qa.index), k=min(num_questions, len(qa)))
elif mode == "Topic search":
    topic = st.sidebar.text_input("Enter keyword (e.g., 'regression')")
    if st.sidebar.button("🔍 Search"):
        matches = qa[qa["question"].str.contains(topic, case=False, na=False) | qa["answer"].str.contains(topic, case=False, na=False)]
        if matches.empty:
            st.sidebar.warning("No matches found.")
        else:
            st.session_state.remaining_qs = list(matches.index[:num_questions])
elif mode == "Browse list":
    pick = st.sidebar.selectbox("Pick a question", qa["question"].tolist())
    if st.sidebar.button("➕ Add to session"):
        idx = qa[qa["question"] == pick].index[0]
        st.session_state.remaining_qs.append(int(idx))

st.markdown(f"**Questions queued:** {len(st.session_state.remaining_qs)}")

# Ask questions
if st.session_state.remaining_qs:
    q_idx = st.session_state.remaining_qs.pop(0)
    q_row = qa.loc[q_idx]
    st.subheader("❓ Question")
    st.write(q_row["question"])
    user_answer = st.text_area("✍️ Your answer:", height=150)

    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        if st.button("✅ Submit Answer"):
            if not user_answer.strip():
                st.warning("Please type an answer.")
            else:
                if evaluation_method == "similarity":
                    score = evaluate_similarity(q_row["answer"], user_answer)
                    method_name = "similarity"
                elif evaluation_method == "keywords":
                    score = keyword_overlap_score(q_row["answer"], user_answer)
                    method_name = "keywords"
                else:
                    score = combined_score(q_row["answer"], user_answer)
                    method_name = "combined"

                save_response(conn, q_row["question"], q_row["answer"], user_answer, score, method_name)
                st.success(f"Score: {score:.1f}/100")
                st.info("Expected answer:")
                st.write(q_row["answer"])

    with col2:
        if st.button("👀 Show model answer"):
            st.info(q_row["answer"])

    with col3:
        if st.button("⏭️ Skip question"):
            st.info("Skipped.")

else:
    st.info("No questions queued. Use sidebar to start session.")

# Logs
if st.checkbox("📊 Show recent attempts"):
    df_logs = pd.read_sql_query("SELECT timestamp, question, user_answer, score, method FROM responses ORDER BY id DESC LIMIT 20", conn)
    if not df_logs.empty:
        st.dataframe(df_logs)
    else:
        st.write("No logs yet.")

if st.button("💾 Export logs"):
    df_logs_all = pd.read_sql_query("SELECT * FROM responses", conn)
    csv_path = "interviewer_logs_export.csv"
    df_logs_all.to_csv(csv_path, index=False)
    st.success(f"Logs exported to `{csv_path}`")

st.markdown("---")
st.markdown("⚡ **Tips**: Use random mode for mocks, topic search for focused prep, and browse list for manual practice.")

conn.close()

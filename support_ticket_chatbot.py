import os
import re
import string
from datetime import datetime
from collections import Counter

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


### Utility helpers 


def highlight_terms(text: str, query: str) -> str:
    for word in re.findall(r"\w+", query):
        text = re.sub(rf"(?i)({re.escape(word)})",
                      r"<mark style='background-color:yellow'>\1</mark>", text)
    return text


def safe_parse_date(val):
    """Parse dates in multiple formats, fallback to pandas."""
    for fmt in (
        "%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y", "%Y/%m/%d",
        "%d/%b/%Y", "%d %b %Y", "%d.%m.%Y", "%m-%d-%Y"
    ):
        try:
            return datetime.strptime(str(val), fmt)
        except Exception:
            continue
    return pd.to_datetime(val, errors="coerce")


### Load Data

st.set_page_config(page_title="💬 Support Ticket Chatbot", layout="wide")
st.title("💬 Support Ticket Chatbot")

csv_file = "support_tickets.csv"
if not os.path.exists(csv_file):
    st.error(" CSV file not found. Please place 'support_tickets.csv' in the app folder.")
    st.stop()

df = pd.read_csv(csv_file)

# Normalize text columns
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].astype(str).str.strip().str.lower()

# Combined text for semantic search
df["combined"] = df.apply(lambda r: " ".join(str(v) for v in r.values), axis=1)
vectorizer = TfidfVectorizer(stop_words="english")
vectorizer.fit(df["combined"])


###Sidebar Filters 

st.sidebar.header("Filters")
top_k = st.sidebar.slider("Number of tickets to show", 1, 1000, 20)

# Priority
apply_priority = st.sidebar.checkbox("Filter by Priority")
priority_filter = None
if apply_priority and "priority" in df.columns:
    p_sel = st.sidebar.selectbox("Select Priority", ["(All)"] + sorted(df["priority"].dropna().unique()))
    priority_filter = None if p_sel == "(All)" else p_sel.strip().lower()

# Status
apply_status = st.sidebar.checkbox("Filter by Status")
status_filter = None
if apply_status and "status" in df.columns:
    s_sel = st.sidebar.selectbox("Select Status", ["(All)"] + sorted(df["status"].dropna().unique()))
    status_filter = None if s_sel == "(All)" else s_sel.strip().lower()

sort_by = st.sidebar.radio("Sort results by", ["Similarity score", "Newest first"])


### Query Handling

def structured_query_handler(query: str):
    q = query.lower()
    result = None

    # Day-of-week queries
    days = {"monday": "mon", "tuesday": "tue", "wednesday": "wed",
            "thursday": "thu", "friday": "fri", "saturday": "sat", "sunday": "sun"}
    for d, short in days.items():
        if d in q and "day_of_week" in df.columns:
            result = df[df["day_of_week"] == short]
            return f"Tickets on {d}: {len(result)}", result

    # Priority queries
    if "high priority" in q:
        result = df[df["priority"] == "high"]
        return f"High priority tickets: {len(result)}", result
    if "medium priority" in q:
        result = df[df["priority"] == "medium"]
        return f"Medium priority tickets: {len(result)}", result
    if "low priority" in q:
        result = df[df["priority"] == "low"]
        return f"Low priority tickets: {len(result)}", result

    return None, None


def retrieve_similar(query: str, top_k: int = 5):
    structured_answer, structured_df = structured_query_handler(query)
    if structured_df is not None:
        return structured_answer, structured_df.head(top_k)

    subset = df.copy()
    if apply_priority and priority_filter:
        subset = subset[subset["priority"] == priority_filter]
    if apply_status and status_filter:
        subset = subset[subset["status"] == status_filter]

    if subset.empty:
        return "No tickets found.", pd.DataFrame()

    q_vec = vectorizer.transform([query])
    scores = cosine_similarity(q_vec, vectorizer.transform(subset["combined"])).flatten()
    subset = subset.assign(Similarity=scores)

    if sort_by == "Similarity score":
        subset = subset.sort_values("Similarity", ascending=False)

    return "", subset.head(top_k)


###Conversation

if "history" not in st.session_state:
    st.session_state.history = []

question = st.text_input("Your question:")
if st.button("Ask") and question.strip():
    with st.spinner("Searching…"):
        ans, results = retrieve_similar(question, top_k)
        if results.empty:
            ans = "No tickets matched."
        st.session_state.history.append((question, ans, results))

st.markdown("### Conversation")
for q, a, res in reversed(st.session_state.history):
    st.markdown(f"**You:** {q}")
    if a:
        st.markdown(f"**Bot:** {a}")
    for _, row in res.iterrows():
        ticket_id = row.get("ticket_id", "N/A")
        snippet = str(row["combined"])[:400]
        st.markdown(
            f"<details><summary>Ticket ID {ticket_id} "
            f"(Similarity {row.get('Similarity',0):.2f})</summary>"
            f"<p>{highlight_terms(snippet, q)}</p></details>",
            unsafe_allow_html=True,
        )
    st.markdown("---")




###Dashboard 

st.markdown("## 📊 Ticket Overview")
cols = st.columns(3)
cols[0].metric("Total Tickets", len(df))
if "status" in df.columns:
    cols[1].metric("Open Tickets", (df["status"] == "open").sum())
if "priority" in df.columns:
    cols[2].metric("High Priority", (df["priority"] == "high").sum())

### Tickets by Priority 
if "priority" in df.columns:
    st.markdown("### 📊 Tickets by Priority")
    priority_counts = df["priority"].value_counts()
    chart_type = st.radio("Choose chart type for Priority", ["Bar", "Pie"], key="priority_chart")
    if chart_type == "Bar":
        st.bar_chart(priority_counts)
    else:
        fig, ax = plt.subplots()
        ax.pie(priority_counts, labels=priority_counts.index, autopct="%1.1f%%", startangle=90)
        ax.axis("equal")
        st.pyplot(fig)

### Tickets by Industry 
if "industry" in df.columns:
    st.markdown("### 📊 Tickets by Industry")
    industry_counts = df["industry"].value_counts()
    chart_type = st.radio("Choose chart type for Industry", ["Bar", "Pie"], key="industry_chart")
    if chart_type == "Bar":
        st.bar_chart(industry_counts)
    else:
        fig, ax = plt.subplots()
        ax.pie(industry_counts, labels=industry_counts.index, autopct="%1.1f%%", startangle=90)
        ax.axis("equal")
        st.pyplot(fig)

### Generic Column Visualizer 

categorical_cols = df.select_dtypes(include="object").columns.tolist()
if categorical_cols:
    st.markdown("### 📊 Explore Any Column")
    col_choice = st.selectbox("Choose a column to visualize:", categorical_cols)

    if col_choice:
        counts = df[col_choice].value_counts()
        chart_type = st.radio("Choose chart type", ["Bar", "Pie"], key=f"{col_choice}_chart")

        if chart_type == "Bar":
            st.bar_chart(counts)
        else:
            fig, ax = plt.subplots()
            ax.pie(counts, labels=counts.index, autopct="%1.1f%%", startangle=90)
            ax.axis("equal")
            st.pyplot(fig)

### Word frequency 
st.markdown("### Top 20 Most Frequent Words")
if not df["combined"].dropna().empty:
    all_text = " ".join(df["combined"].astype(str)).lower()
    all_text = all_text.translate(str.maketrans("", "", string.punctuation))
    words = [w for w in all_text.split() if len(w) > 2]
    word_counts = Counter(words)
    if word_counts:
        labels, values = zip(*word_counts.most_common(20))
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(labels, values, color="steelblue")
        ax.invert_yaxis()
        st.pyplot(fig)
    else:
        st.info(" Not enough text data to compute word frequency.")
else:
    st.info(" No combined text data available.")

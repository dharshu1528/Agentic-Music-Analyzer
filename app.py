import os
import streamlit as st
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv
from music_agent import answer_question
from data_utils import load_or_create_data, DATA_PATH

load_dotenv()
st.set_page_config(page_title="Agentic Music Listening Analyzer", page_icon="🎵", layout="wide")

st.title("🎵 Agentic AI Music Listening Pattern Analyzer")
st.caption("Explore listening habits, discover patterns, and ask questions about your music data.")

df = load_or_create_data()

with st.sidebar:
    st.header("Project status")
    st.success(f"Dataset ready: {len(df):,} listening records")
    st.write(f"Data file: `{DATA_PATH}`")
    st.write("AI mode: " + ("LLM agent enabled" if os.getenv("OPENAI_API_KEY") else "Local demo mode"))
    st.info("Add an OpenAI API key in Replit Secrets to enable natural-language agent responses. The dashboard works without it.")

st.subheader("Your listening overview")
c1, c2, c3 = st.columns(3)
c1.metric("Total plays", f"{len(df):,}")
c2.metric("Unique artists", f"{df['artist'].nunique():,}")
c3.metric("Listening time", f"{df['duration_min'].sum():,.0f} min")

left, right = st.columns(2)
with left:
    genre_counts = df["genre"].value_counts().reset_index()
    genre_counts.columns = ["genre", "plays"]
    st.plotly_chart(px.bar(genre_counts, x="genre", y="plays", title="Listening by genre"), use_container_width=True)
with right:
    hourly = df.groupby("hour", as_index=False).size().rename(columns={"size": "plays"})
    st.plotly_chart(px.line(hourly, x="hour", y="plays", markers=True, title="Listening by hour of day"), use_container_width=True)

st.divider()
st.subheader("Ask your music data agent")
st.write("Try: *Which genre do I listen to most?* or *When do I listen to music most?*")
question = st.text_input("Your question", placeholder="Which artists do I listen to most?")
if st.button("Ask agent", type="primary") and question.strip():
    with st.spinner("Analyzing your listening data..."):
        response = answer_question(question, df)
    st.markdown(response)

with st.expander("Preview dataset"):
    st.dataframe(df.head(20), use_container_width=True)
    st.download_button("Download listening dataset (CSV)", df.to_csv(index=False), "music_listening_history.csv", "text/csv")

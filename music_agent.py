"""A small LangGraph agent with safe, read-only music analytics tools."""
import os
import pandas as pd

def local_answer(question, df):
    """Rule-based fallback so the app can be explored without an API key."""
    q = question.lower()
    if "genre" in q:
        counts = df["genre"].value_counts()
        return f"Your most-played genre is **{counts.index[0]}** with **{counts.iloc[0]} plays**."
    if "hour" in q or "time" in q or "when" in q:
        counts = df.groupby("hour").size()
        h = int(counts.idxmax())
        return f"Your busiest listening hour is **{h:02d}:00–{(h+1)%24:02d}:00**, with **{int(counts.max())} plays**."
    if "artist" in q:
        counts = df["artist"].value_counts().head(5)
        return "Your top artists are:\n" + "\n".join(f"- **{name}**: {count} plays" for name, count in counts.items())
    if "long" in q or "duration" in q or "minutes" in q:
        return f"You listened for approximately **{df['duration_min'].sum():,.1f} minutes** in total."
    counts = df["genre"].value_counts().head(3)
    return "Here is a quick snapshot of your top genres:\n" + "\n".join(f"- {g}: {n} plays" for g, n in counts.items()) + "\n\nTry asking about your favourite genre, top artists, listening time, or total duration."

def answer_question(question, df):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "### Local demo answer\n" + local_answer(question, df) + "\n\n*This is a local demo response. Add an API key to enable the LangGraph-powered AI agent.*"
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.tools import tool
        from langgraph.prebuilt import create_react_agent

        @tool
        def top_genres() -> str:
            """Return the most listened-to music genres and play counts."""
            return df["genre"].value_counts().head(8).to_string()

        @tool
        def top_artists() -> str:
            """Return the most listened-to artists and play counts."""
            return df["artist"].value_counts().head(8).to_string()

        @tool
        def busiest_hours() -> str:
            """Return listening play counts grouped by hour of day."""
            return df.groupby("hour").size().sort_values(ascending=False).head(8).to_string()

        @tool
        def listening_summary() -> str:
            """Return total plays, unique artists and total listening minutes."""
            return (f"Plays: {len(df)}; unique artists: {df['artist'].nunique()}; "
                    f"total minutes: {df['duration_min'].sum():.1f}")

        tools = [top_genres, top_artists, busiest_hours, listening_summary]
        model = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
        agent = create_react_agent(model, tools)
        result = agent.invoke({"messages": [
            ("system", "You are a careful music listening analyst. Use the provided tools for data claims. "
                       "Explain findings in simple language. The dataset is synthetic sample data, not real Spotify data."),
            ("user", question)
        ]})
        return result["messages"][-1].content
    except Exception as exc:
        return ("The AI agent could not run. Check your API key and package setup. "
                f"Technical detail: `{type(exc).__name__}: {exc}`\n\n"
                "Local demo answer:\n" + local_answer(question, df))

from pathlib import Path
import pandas as pd
import numpy as np

DATA_PATH = Path("data/music_listening_history.csv")

def create_sample_data(rows=1200, seed=42):
    """Create clearly labelled synthetic music listening records for development."""
    rng = np.random.default_rng(seed)
    artists_by_genre = {
        "Pop": ["Ava Lane", "The Satellites", "Mira Blue"],
        "Rock": ["Northbound", "Paper Signals", "The Echoes"],
        "Hip-Hop": ["Nova K", "Jay River", "Golden Era"],
        "Classical": ["Lumen Quartet", "Aria Stone", "Orchid Ensemble"],
        "Electronic": ["Pixel Wave", "Neon Harbour", "Circuit Bloom"],
        "Jazz": ["Blue Room Trio", "Miles Garden", "Velvet Keys"],
    }
    genres = list(artists_by_genre)
    genre = rng.choice(genres, size=rows, p=[.28, .18, .17, .10, .17, .10])
    artists = [rng.choice(artists_by_genre[g]) for g in genre]
    start = pd.Timestamp("2025-01-01")
    days = rng.integers(0, 180, size=rows)
    minutes = rng.integers(1, 8 * 60, size=rows)
    timestamps = [start + pd.Timedelta(days=int(d), minutes=int(m)) for d, m in zip(days, minutes)]
    duration = np.round(rng.uniform(2.0, 5.5, size=rows), 2)
    return pd.DataFrame({
        "listening_id": range(1, rows + 1),
        "user_id": rng.choice(["user_001", "user_002", "user_003"], size=rows),
        "played_at": timestamps,
        "artist": artists,
        "genre": genre,
        "track": [f"Track {i:04d}" for i in range(1, rows + 1)],
        "duration_min": duration,
    })

def load_or_create_data():
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH, parse_dates=["played_at"])
    else:
        df = create_sample_data()
        df.to_csv(DATA_PATH, index=False)
    df["played_at"] = pd.to_datetime(df["played_at"])
    df["hour"] = df["played_at"].dt.hour
    df["date"] = df["played_at"].dt.date.astype(str)
    return df

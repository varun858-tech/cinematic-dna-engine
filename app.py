import os
os.environ["PYTHONUTF8"] = "1"  # Force Windows to drop cp1252 and decode UTF-8 globally

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from sklearn.metrics.pairwise import cosine_similarity

# Set modern wide layout configuration with an administrative page header
st.set_page_config(page_title="Cinematic DNA Engine", layout="wide", page_icon="🎬")

# ==========================================
# 🌌 CUSTOM PREMIUM CSS INJECTION LAYER
# ==========================================
st.markdown("""
    <style>
    /* Import modern sleek font profiles from Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    /* Apply uniform premium font tracking globally across the web application wrapper */
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', sans-serif !important;
        background-color: #0d0f12 !important; /* Premium Cinematic Deep Space Dark */
        color: #e2e8f0 !important;
    }
    
    /* Modernize Title Headers */
    h1 {
        font-weight: 700 !important;
        color: #ffffff !important;
        letter-spacing: -0.05em !important;
        background: linear-gradient(45deg, #ff4b4b, #ff7676);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px !important;
    }
    
    /* Modernize Subsection Headers */
    h3, .stSubheader {
        font-weight: 600 !important;
        color: #f1f5f9 !important;
        letter-spacing: -0.02em !important;
        margin-top: 20px !important;
    }
    
    /* Restyle the Selectbox input container box wrapper */
    div[data-baseweb="select"] {
        background-color: #1e222b !important;
        border-radius: 10px !important;
        border: 1px solid #334155 !important;
    }
    
    /* CSS Injection Class for clean, uniform movie card visualization grids */
    .movie-card {
        background-color: #151922;
        border-radius: 12px;
        padding: 0px;
        margin-bottom: 20px;
        border: 1px solid #232936;
        transition: transform 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275), border-color 0.3s ease, box-shadow 0.3s ease;
    }
    
    /* Premium Hover Animation Effects (Smooth lifting depth matching premium apps) */
    .movie-card:hover {
        transform: translateY(-8px);
        border-color: #ff4b4b;
        box-shadow: 0 12px 20px rgba(255, 75, 75, 0.15);
    }
    
    /* Enforce standardized dimensions and rounding constraints onto images/posters */
    .movie-poster {
        border-radius: 12px 12px 0px 0px !important;
        width: 100% !important;
        height: auto !important;
        object-fit: cover !important;
        display: block;
    }
    
    /* Clean internal metadata layout container padding configurations */
    .movie-info {
        padding: 12px 10px 14px 10px;
    }
    
    /* Text layout styling rules for titles */
    .movie-title {
        font-weight: 600 !important;
        font-size: 14px !important;
        color: #ffffff !important;
        line-height: 1.3 !important;
        margin-bottom: 6px !important;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
        height: 36px; /* Strict structural height balancing prevents label offset drifts */
    }
    
    /* Micro-styling layout adjustments for numeric ratings */
    .movie-rating {
        font-size: 12px !important;
        font-weight: 600 !important;
        color: #ffb800 !important; /* Premium Star Gold */
        display: flex;
        align-items: center;
        gap: 4px;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. HARDWARE CACHING & DATA INGESTION
# ==========================================
@st.cache_data
def load_base_data():
    """Loads and caches the cleaned final dataframe."""
    df = pd.read_csv('movies_final_level6.csv')
    return df

@st.cache_resource
def load_vector_matrices():
    """Loads and caches heavy vector spaces safely."""
    with open('movie_embeddings.pkl', 'rb') as f:
        sbert = pickle.load(f, encoding='utf-8')
    with open("movie_tfidf.pkl", "rb") as f:
        tfidf = pickle.load(f, encoding='utf-8')
    return sbert, tfidf

# Initialize data and vector frameworks
movies = load_base_data()
sbert_vectors, tfidf_vectors = load_vector_matrices()

# Global variables for score calculations
min_votes = movies["vote_count"].min()
max_votes = movies["vote_count"].max()

def norm_votes(v):
    return (v - min_votes) / (max_votes - min_votes)

# ==========================================
# 3. THE SHARED API TUNNEL (Your exact stable connection)
# ==========================================
@st.cache_resource
def get_stable_session():
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    return session

session = get_stable_session()
TMDB_ACCESS_TOKEN = "eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiI1NzNlNDNlYmE3MDMxODgzYzVhZTI5ODE4NDgwNDI4MCIsIm5iZiI6MTc3MjEyNzAxOC45MTYsInN1YiI6IjY5YTA4MzJhMTFmNTkxNTk2ZGYyMDY1NSIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ._8rLI9shumkeEqNQkwli8-2C06eoJwW5hjSSPNa6wqw".strip()

def fetch_poster_url_ultimate(raw_id):
    """Your exact stable endpoint connection string converted for layout grids."""
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {TMDB_ACCESS_TOKEN}"
    }
    try:
        clean_id = int(float(str(raw_id).strip()))
        url = f"https://api.themoviedb.org/3/movie/{clean_id}"
        response = session.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            path = response.json().get('poster_path')
            if path:
                return f"https://image.tmdb.org/t/p/w500{path}"
    except Exception:
        pass
    return "placeholder"  # Used as an explicit tracking keyword string for UI skips

# ==========================================
# 4. CORE HYBRID ALGORITHMS
# ==========================================
def recommend(target_idx):
    sim_tfidf = cosine_similarity(tfidf_vectors[target_idx], tfidf_vectors).flatten()
    sim_sbert = cosine_similarity([sbert_vectors[target_idx]], sbert_vectors).flatten()
    
    combined_sim = (0.6 * sim_tfidf) + (0.4 * sim_sbert)
    
    similar_indices = sorted(
        list(enumerate(combined_sim)), 
        reverse=True, 
        key=lambda x: x[1]
    )[1:101]

    rec_list = []

    for i in similar_indices:
        idx_i = i[0]
        sim_score = i[1]
        temp_df = movies.iloc[idx_i]

        if temp_df["rating"] < 6.5:
            continue

        rating_score = temp_df["rating"] / 10
        popularity_score = norm_votes(temp_df["vote_count"])

        final_score = (
            0.82 * sim_score + 
            0.13 * rating_score + 
            0.05 * popularity_score
        )

        rec_list.append({
            "id": str(temp_df["id"]),
            "title": temp_df["title"],
            "rating": temp_df["rating"],
            "score": final_score
        })

    final_recommendations = sorted(rec_list, key=lambda x: x["score"], reverse=True)
    return final_recommendations

def director_spotlight_by_idx(target_idx, n=5):
    filter_crew = movies.loc[target_idx, "crew"]
    display_director_name = movies.loc[target_idx, "director"]
    
    director_movies = movies[(movies["crew"] == filter_crew) & (movies.index != target_idx)]

    if director_movies.empty:
        return display_director_name, []

    top_director_df = director_movies.sort_values("rating", ascending=False).head(n)
    
    director_recs = []
    for _, row in top_director_df.iterrows():
        director_recs.append({
            "id": str(row["id"]),
            "title": row["title"],
            "rating": row["rating"]
        })

    return display_director_name, director_recs

# ==========================================
# 5. STREAMLIT FRONTEND USER INTERFACE
# ==========================================
st.title("🎬 Cinematic DNA Engine")
st.markdown("<p style='color:#94a3b8; font-size:16px; margin-top:-10px;'>Level 06: Hybrid Lexical-Semantic Discovery Engine</p>", unsafe_allow_html=True)

# Predictive Autocomplete Dropdown Option List
movie_options = movies.sort_values(by="vote_count", ascending=False)["title"].tolist()

selected_movie_title = st.selectbox(
    "🔍 Search and Select a Movie (Type characters to filter options instantly):",
    options=movie_options,
    index=0
)

if selected_movie_title:
    target_idx = movies[movies["title"] == selected_movie_title].index[0]
    
    # ------------------------------------------
    # --- SECTION A: PRIMARY RECOMMENDATIONS UI GRID ---
    # ------------------------------------------
    st.subheader(f"🍿 Top 10 High Quality Recommendations")
    
    with st.spinner("Processing hybrid lexical-transformer text tensors..."):
        recommendations = recommend(target_idx)
    
    if recommendations:
        cols = st.columns(5)
        displayed_count = 0  # Dynamic layout track variable to lock grid boundaries
        
        for movie in recommendations:
            poster_url = fetch_poster_url_ultimate(movie["id"])
            
            # GUARDRAIL: Skip broken image paths seamlessly without fracturing the UI container grid
            if "placeholder" in poster_url:
                continue
                
            # Limit the display strictly to the top 10 valid cards
            if displayed_count >= 10:
                break
                
            col_idx = displayed_count % 5
            with cols[col_idx]:
                # Inject a custom HTML card to support sophisticated CSS padding and lift transitions
                st.markdown(f"""
                    <div class="movie-card">
                        <img class="movie-poster" src="{poster_url}">
                        <div class="movie-info">
                            <div class="movie-title">{movie['title']}</div>
                            <div class="movie-rating">⭐ {movie['rating']:.2f}</div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            displayed_count += 1
    else:
        st.warning("No adjacent movies passed baseline quality thresholds.")

    st.markdown("<br><hr style='border-color:#232936;'><br>", unsafe_allow_html=True)

    # ------------------------------------------
    # --- SECTION B: DIRECTOR'S SPOTLIGHT FIXED GRID ---
    # ------------------------------------------
    director_name, dir_recs = director_spotlight_by_idx(target_idx)
    
    if dir_recs:
        st.subheader(f"🎥 Director's Spotlight: More from **{director_name}**")
        
        # Pre-filter out any spotlight movies without completely working posters
        valid_dir_recs = []
        for d_movie in dir_recs:
            p_url = fetch_poster_url_ultimate(d_movie["id"])
            if "placeholder" not in p_url:
                d_movie["poster_url"] = p_url  
                valid_dir_recs.append(d_movie)
        
        if valid_dir_recs:
            # Force structural 5-column maximum constraint to restrict layout stretching
            dir_cols = st.columns(5)
            
            for i, d_movie in enumerate(valid_dir_recs):
                col_idx = i % 5
                with dir_cols[col_idx]:
                    st.markdown(f"""
                        <div class="movie-card">
                            <img class="movie-poster" src="{d_movie['poster_url']}">
                            <div class="movie-info">
                                <div class="movie-title">{d_movie['title']}</div>
                                <div class="movie-rating">⭐ {d_movie['rating']:.2f}</div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
        else:
            st.info(f"Films by **{director_name}** exist, but their visuals are currently unreachable.")
    else:
        st.subheader("🎥 Director's Spotlight")
        st.info(f"No other high-quality films by **{director_name}** found in our active directory catalog.")
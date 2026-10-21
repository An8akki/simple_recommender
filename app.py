"""
Streamlit Web Application for the Movie Recommendation System.
Provides a cinematic, responsive UI for:
1. Movie-to-Movie Content & Hybrid Recommendations
2. Natural Language Plot & Theme Search
3. Interactive Dataset & Model Analytics
4. Architecture & Pipeline Overview
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
from recommender import MovieRecommender

# Set Page Config
st.set_page_config(
    page_title="CineMatch AI | Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Cinematic CSS
st.markdown("""
<style>
    /* Dark Theme & Typography */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@500;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #030712 100%);
        padding: 2.2rem 2.5rem;
        border-radius: 18px;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.5);
    }
    
    .title-gradient {
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc, #f43f5e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.6rem;
        font-weight: 800;
        font-family: 'Space Grotesk', sans-serif;
        margin-bottom: 0.4rem;
    }
    
    .subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        font-weight: 400;
    }
    
    /* Movie Card Styling */
    .movie-card {
        background: linear-gradient(145deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 1.3rem;
        margin-bottom: 1.2rem;
        transition: transform 0.25s ease, box-shadow 0.25s ease;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    
    .movie-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 24px -8px rgba(56, 189, 248, 0.25);
        border-color: rgba(56, 189, 248, 0.4);
    }
    
    .card-banner {
        height: 120px;
        border-radius: 10px;
        background: linear-gradient(135deg, #312e81 0%, #1e1b4b 50%, #0284c7 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        position: relative;
        overflow: hidden;
        margin-bottom: 0.9rem;
    }
    
    .card-banner::after {
        content: "🎬";
        font-size: 3rem;
        opacity: 0.35;
    }
    
    .badge-rating {
        position: absolute;
        top: 8px;
        right: 8px;
        background: rgba(15, 23, 42, 0.85);
        backdrop-filter: blur(8px);
        color: #facc15;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.8rem;
        border: 1px solid rgba(250, 204, 21, 0.3);
    }
    
    .badge-year {
        position: absolute;
        bottom: 8px;
        left: 8px;
        background: rgba(15, 23, 42, 0.85);
        color: #e2e8f0;
        padding: 2px 7px;
        border-radius: 5px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    .card-title {
        color: #f8fafc;
        font-size: 1.15rem;
        font-weight: 700;
        margin-bottom: 0.4rem;
        line-height: 1.3;
    }
    
    .genre-pill {
        display: inline-block;
        background: rgba(56, 189, 248, 0.12);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.25);
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.72rem;
        font-weight: 600;
        margin-right: 4px;
        margin-bottom: 4px;
    }
    
    .similarity-pill {
        display: inline-block;
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 2px 9px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 700;
        margin-bottom: 0.6rem;
    }
    
    .card-overview {
        color: #94a3b8;
        font-size: 0.84rem;
        line-height: 1.45;
        margin: 0.6rem 0;
        display: -webkit-box;
        -webkit-line-clamp: 3;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }
    
    .card-meta {
        font-size: 0.78rem;
        color: #64748b;
        margin-top: 0.4rem;
    }
    
    .stat-box {
        background: #1e293b;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .stat-number {
        font-size: 1.8rem;
        font-weight: 800;
        color: #38bdf8;
    }
    .stat-label {
        font-size: 0.85rem;
        color: #94a3b8;
    }
</style>
""", unsafe_allow_html=True)

# Cache Recommender Instance
@st.cache_resource
def get_recommender():
    return MovieRecommender(model_dir="models")

try:
    rec = get_recommender()
except Exception as e:
    st.error(f"Error loading recommendation model: {e}")
    st.info("Please make sure you have run `data_preprocessing.py` and `train_model.py` first.")
    st.stop()

# Header
st.markdown("""
<div class="main-header">
    <div class="title-gradient">🎬 CineMatch AI</div>
    <div class="subtitle">Next-Gen Intelligent Movie Recommendation Engine & Semantic Plot Discovery</div>
</div>
""", unsafe_allow_html=True)

# Sidebar Controls
st.sidebar.title("⚙️ Engine Controls")
st.sidebar.markdown("---")

all_genres = sorted(list({g for genres in rec.df['genres_display'] for g in genres}))
genre_choice = st.sidebar.selectbox("Filter by Genre", ["All Genres"] + all_genres)
active_genre = None if genre_choice == "All Genres" else genre_choice

num_recs = st.sidebar.slider("Number of Recommendations", min_value=3, max_value=20, value=8, step=1)
min_rating_filter = st.sidebar.slider("Minimum Rating (IMDb)", min_value=0.0, max_value=9.0, value=0.0, step=0.5)

st.sidebar.markdown("---")
st.sidebar.subheader("⚖️ Hybrid Scoring Weights")
hybrid_weight = st.sidebar.slider(
    "Quality / Popularity Weight",
    min_value=0.0,
    max_value=0.6,
    value=0.15,
    step=0.05,
    help="0.0 = 100% Content Similarity. Higher values boost critically acclaimed and popular films."
)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Dataset**: TMDB 5,000 Movies & Credits\n\n⚡ **NLP Model**: TF-IDF (6,000 features, 1-2 ngrams) + Cosine Similarity")

# Navigation Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Movie Recommender",
    "🔍 Plot & Theme Search",
    "📊 Dataset & Model Insights",
    "🧠 System Architecture"
])

# ----------------- TAB 1: MOVIE RECOMMENDER -----------------
with tab1:
    st.markdown("### Find Movies Similar to Your Favorites")
    st.write("Pick a movie from the catalog to discover highly relevant, mathematically matched titles.")
    
    col_input, col_btn = st.columns([4, 1])
    
    with col_input:
        default_idx = rec.all_titles.index("The Dark Knight") if "The Dark Knight" in rec.all_titles else 0
        selected_title = st.selectbox(
            "Select or Type a Movie Title:",
            options=rec.all_titles,
            index=default_idx
        )
        
    with col_btn:
        st.write("")
        st.write("")
        recommend_clicked = st.button("🚀 Recommend", use_container_width=True, type="primary")

    if selected_title:
        res = rec.recommend(
            movie_title=selected_title,
            top_n=num_recs,
            genre_filter=active_genre,
            min_rating=min_rating_filter,
            hybrid_weight=hybrid_weight
        )
        
        if "error" in res and res["error"]:
            st.error(res["error"])
        else:
            q = res["query_movie"]
            
            # Selected Movie Details Box
            with st.container():
                st.markdown(f"""
                <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 12px; padding: 1.2rem; margin: 1rem 0 1.5rem 0;">
                    <div style="display: flex; justify-content: space-between; align-items: baseline; flex-wrap: wrap;">
                        <h3 style="color: #f8fafc; margin: 0;">{q['title']} <span style="color: #94a3b8; font-size: 1rem; font-weight: 400;">({q['release_year']})</span></h3>
                        <div style="background: rgba(250, 204, 21, 0.15); color: #facc15; border: 1px solid rgba(250, 204, 21, 0.3); padding: 4px 10px; border-radius: 8px; font-weight: 700;">
                            ⭐ {q['vote_average']} / 10 <span style="font-size: 0.8rem; color: #94a3b8;">({q['vote_count']:,} votes)</span>
                        </div>
                    </div>
                    <div style="margin: 0.5rem 0;">
                        {' '.join([f'<span class="genre-pill">{g}</span>' for g in q['genres']])}
                        <span style="color: #64748b; font-size: 0.85rem; margin-left: 8px;">⏳ {q['runtime']} mins</span>
                    </div>
                    <p style="color: #cbd5e1; font-size: 0.92rem; margin: 0.4rem 0;">{q['overview']}</p>
                    <div style="color: #94a3b8; font-size: 0.82rem; margin-top: 0.4rem;">
                        <strong>Director:</strong> {', '.join(q['director']) if q['director'] else 'Unknown'} &nbsp;|&nbsp; 
                        <strong>Cast:</strong> {', '.join(q['cast']) if q['cast'] else 'Unknown'}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            st.subheader(f"Recommended Movies ({len(res['recommendations'])} Results)")
            
            if not res['recommendations']:
                st.warning("No recommendations found matching your genre/rating filter. Try relaxing the filters in the sidebar!")
            else:
                cols = st.columns(4)
                for i, r in enumerate(res['recommendations']):
                    with cols[i % 4]:
                        match_pct = int(r.get('similarity_score', 0) * 100)
                        genre_badges = "".join([f'<span class="genre-pill">{g}</span>' for g in r['genres'][:3]])
                        
                        st.markdown(f"""
                        <div class="movie-card">
                            <div>
                                <div class="card-banner">
                                    <span class="badge-rating">⭐ {r['vote_average']}</span>
                                    <span class="badge-year">{r['release_year']}</span>
                                </div>
                                <div class="similarity-pill">🎯 {match_pct}% Match</div>
                                <div class="card-title">{r['title']}</div>
                                <div>{genre_badges}</div>
                                <div class="card-overview">{r['overview']}</div>
                            </div>
                            <div class="card-meta">
                                <div><strong>Director:</strong> {r['director'][0] if r['director'] else 'N/A'}</div>
                                <div><strong>Cast:</strong> {', '.join(r['cast'][:2]) if r['cast'] else 'N/A'}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

# ----------------- TAB 2: PLOT SEARCH -----------------
with tab2:
    st.markdown("### Search by Plot, Mood, or Storyline")
    st.write("Can't remember a title? Type what the movie is about, or describe the vibe you're craving.")
    
    preset_col1, preset_col2, preset_col3 = st.columns(3)
    preset_query = ""
    with preset_col1:
        if st.button("🚀 Interstellar Space Exploration", use_container_width=True):
            preset_query = "astronauts traveling through wormhole in space to save humanity from dying earth"
    with preset_col2:
        if st.button("🕵️ Mind-Bending Crime Mystery", use_container_width=True):
            preset_query = "detective investigating psychological dream mystery conspiracy thriller"
    with preset_col3:
        if st.button("🧙 Magical Fantasy Quest", use_container_width=True):
            preset_query = "magical kingdom epic fantasy war with dragons wizards and ancient relics"

    query_input = st.text_area(
        "Enter Movie Plot Description:",
        value=preset_query if preset_query else "a superhero who fights against dark forces and corruption in gotham city",
        height=90
    )
    
    search_clicked = st.button("🔎 Search by Plot", type="primary")
    
    if query_input:
        search_results = rec.search_by_plot(
            query_text=query_input,
            top_n=num_recs,
            genre_filter=active_genre,
            min_rating=min_rating_filter
        )
        
        st.subheader(f"Matches for your storyline ({len(search_results)} Results)")
        if not search_results:
            st.info("No matching movies found for this query. Try adding more keywords or relaxing filters.")
        else:
            cols = st.columns(4)
            for i, r in enumerate(search_results):
                with cols[i % 4]:
                    rel_pct = int(r.get('similarity_score', 0) * 100)
                    genre_badges = "".join([f'<span class="genre-pill">{g}</span>' for g in r['genres'][:3]])
                    
                    st.markdown(f"""
                    <div class="movie-card">
                        <div>
                            <div class="card-banner">
                                <span class="badge-rating">⭐ {r['vote_average']}</span>
                                <span class="badge-year">{r['release_year']}</span>
                            </div>
                            <div class="similarity-pill">🔍 Relevance {rel_pct}%</div>
                            <div class="card-title">{r['title']}</div>
                            <div>{genre_badges}</div>
                            <div class="card-overview">{r['overview']}</div>
                        </div>
                        <div class="card-meta">
                            <div><strong>Director:</strong> {r['director'][0] if r['director'] else 'N/A'}</div>
                            <div><strong>Cast:</strong> {', '.join(r['cast'][:2]) if r['cast'] else 'N/A'}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

# ----------------- TAB 3: DATASET & MODEL INSIGHTS -----------------
with tab3:
    st.markdown("### Dataset Distribution & Model Analytics")
    
    # Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown("""
        <div class="stat-box">
            <div class="stat-number">4,803</div>
            <div class="stat-label">Total Cleaned Movies</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown("""
        <div class="stat-box">
            <div class="stat-number">6,000</div>
            <div class="stat-label">TF-IDF Vector Features</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown("""
        <div class="stat-box">
            <div class="stat-number">20</div>
            <div class="stat-label">Distinct Genres</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown("""
        <div class="stat-box">
            <div class="stat-number">&lt; 1 ms</div>
            <div class="stat-label">Inference Latency</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.write("")
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Top Genres Distribution")
        genre_flat = [g for sublist in rec.df['genres_display'] for g in sublist]
        genre_counts = pd.Series(genre_flat).value_counts().head(12)
        st.bar_chart(genre_counts)
        
    with c2:
        st.subheader("Highest Rated Movies (Minimum 2,500 Votes)")
        top_rated = rec.df[rec.df['vote_count'] >= 2500].sort_values(by='vote_average', ascending=False)[
            ['title', 'release_year', 'vote_average', 'vote_count']
        ].head(10).reset_index(drop=True)
        st.dataframe(top_rated, use_container_width=True)

# ----------------- TAB 4: ARCHITECTURE -----------------
with tab4:
    st.markdown("### Recommendation System Architecture")
    st.markdown("""
    This recommendation system is built using a complete end-to-end Machine Learning and Natural Language Processing (NLP) pipeline:
    
    1. **Data Ingestion & Cleaning (`data_preprocessing.py`)**:
       - Merged `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` on movie ID with 100% fidelity.
       - Parsed JSON structures to extract genres, keywords, top 4 cast members, and directors.
       - Collapsed compound names (e.g., `"James Cameron"` → `"jamescameron"`, `"Science Fiction"` → `"sciencefiction"`) to prevent split token ambiguities.
       - Handled missing values in overview, runtime, and release date.
       - Computed IMDb weighted rating ($WR$) using Bayesian mean adjustment to counteract vote-count bias.
       - Applied Porter Stemming to reduce morphological variants to root words.
       
    2. **NLP Vectorization & Model Training (`train_model.py`)**:
       - Vectorized composite text using **TF-IDF (Term Frequency - Inverse Document Frequency)** with unigram and bigram representation ($6,000$ features).
       - Computed pairwise **Cosine Similarity**:
         $$\\text{Cosine Similarity}(\\mathbf{u}, \\mathbf{v}) = \\frac{\\mathbf{u} \\cdot \\mathbf{v}}{\\|\\mathbf{u}\\| \\|\\mathbf{v}\\|}$$
       - Precomputed top-50 nearest neighbors lookup for sub-millisecond retrieval.
       - Serialized compact model artifacts into `models/`.
       
    3. **Hybrid Inference Engine (`recommender.py`)**:
       - Exact, lowercase, and fuzzy string matching for resilient title lookup.
       - Hybrid scoring formula:
         $$\\text{Score} = (1 - \\alpha) \\times \\text{Sim}(\\text{movie}_a, \\text{movie}_b) + \\alpha \\times \\text{Norm}(\\text{Rating})$$
       - Free-text semantic plot search by vectorizing arbitrary user prompts against the trained TF-IDF vector space.
    """)

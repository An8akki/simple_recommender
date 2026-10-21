# 🎬 CineMatch AI — Movie Recommendation System

An end-to-end Machine Learning and Natural Language Processing (NLP) movie recommendation engine built on the TMDB 5,000 Movies & Credits dataset from `https://github.com/nishanthjay94-hub/SSN/tree/main/SSN`.

---

## 🚀 Key Features

1. **Content-Based Filtering**:
   - Analyzes high-dimensional semantic feature tags combining movie plot overviews, genres, keywords, top actors, and director names.
   - Applies morphological normalization with **Porter Stemming**.
   - Vectorizes features using **TF-IDF with unigrams and bigrams** (6,000 top n-gram features).
   - Computes pairwise **Cosine Similarity** for sub-millisecond similarity scoring.

2. **Bayesian Weighted Rating (Hybrid Scoring)**:
   - Integrates the IMDb weighted formula to counteract sample size bias in vote counts:
     $$WR = \left(\frac{v}{v+m}\right) R + \left(\frac{m}{v+m}\right) C$$
   - Users can dynamically balance pure content similarity vs. critical acclaim and popularity.

3. **Natural Language Plot / Theme Search**:
   - Enables users to describe arbitrary movie plots, moods, or tropes (e.g. *"detective trapped in a virtual reality nightmare"* or *"superhero multiverse time travel"*) and computes real-time vector similarity against the catalog.

4. **Interactive Cinematic Web App**:
   - Built with Streamlit featuring a modern dark-mode aesthetic, genre filter badges, rating stars, responsive movie cards, dataset analytics charts, and real-time query controls.

---

## 📁 Repository Structure

```
d:/movies/
├── data/
│   ├── tmdb_5000_movies.csv      # Raw TMDB movies dataset (5.7 MB)
│   ├── tmdb_5000_credits.csv     # Raw TMDB credits dataset (40 MB)
│   ├── cleaned_movies.csv        # Preprocessed & stemmed movies CSV (8.6 MB)
│   └── cleaned_movies.pkl        # Cleaned dataset pickle (8.4 MB)
├── models/
│   ├── movies_meta.pkl           # Processed movie metadata dataframe
│   ├── top_similarities.pkl      # Precomputed top-50 neighbor lookups
│   ├── tfidf_vectorizer.pkl      # Trained TF-IDF vectorizer
│   ├── tfidf_matrix.pkl          # Sparse TF-IDF matrix
│   └── similarity_matrix.npz     # Compressed cosine similarity matrix
├── data_preprocessing.py         # Data cleaning & NLP pipeline
├── train_model.py                # Model training & artifact export
├── recommender.py                # High-performance recommendation API
├── test_recommender.py           # Unit tests
├── app.py                        # Streamlit web application
└── README.md                     # Documentation
```

---

## 🛠️ Installation & Setup

Ensure Python 3.10+ is available. Install dependencies (if not already installed):

```bash
pip install pandas numpy scikit-learn nltk streamlit
```

---

## 🔄 Pipeline Execution

### Step 1: Clean and Preprocess Data
```bash
python data_preprocessing.py
```
- Extracts metadata from raw JSON columns: `genres`, `keywords`, `cast`, `crew`.
- Collapses names to eliminate token ambiguity (e.g., `"James Cameron"` → `"jamescameron"`).
- Handles missing values across runtime, release date, and plot summaries.
- Computes weighted IMDb score.
- Applies Porter Stemming and saves artifacts into `data/`.

### Step 2: Train Model & Compute Similarities
```bash
python train_model.py
```
- Fits TF-IDF on stemmed composite tags.
- Calculates Cosine Similarity matrix.
- Precomputes top-50 nearest neighbors for instant lookups.
- Saves model artifacts into `models/`.

### Step 3: Run Unit Tests
```bash
python -m unittest test_recommender.py
```

### Step 4: Launch Interactive Web Application
```bash
streamlit run app.py
```

---

## 💻 Python Usage Example

```python
from recommender import MovieRecommender

# Initialize recommender engine
rec = MovieRecommender(model_dir="models")

# 1. Movie-to-Movie Content Recommendation
results = rec.recommend("The Dark Knight", top_n=5, hybrid_weight=0.15)
print(f"Recommendations for '{results['query_movie']['title']}':")
for r in results['recommendations']:
    print(f"- {r['title']} ({r['release_year']}) | Match: {r['similarity_score']} | Rating: {r['vote_average']}")

# 2. Natural Language Plot Search
matches = rec.search_by_plot("space mission crew encounter mysterious alien signal", top_n=5)
for m in matches:
    print(f"- {m['title']} | Score: {m['similarity_score']} | Genres: {m['genres']}")
```

---

## 📊 Sample Recommendations Benchmark

### Query: *The Dark Knight*
1. **The Dark Knight Rises** (Score: 0.419)
2. **Batman Begins** (Score: 0.365)
3. **Batman Forever** (Score: 0.284)
4. **Batman Returns** (Score: 0.278)
5. **Batman v Superman: Dawn of Justice** (Score: 0.243)

### Query: *Interstellar*
1. **Apollo 13** (Score: 0.188)
2. **Silent Running** (Score: 0.174)
3. **Space Cowboys** (Score: 0.166)
4. **The Martian** (Score: 0.164)
5. **The Right Stuff** (Score: 0.162)

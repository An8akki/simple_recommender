"""
Model Training Pipeline for Movie Recommendation System.
Trains TF-IDF vectorizer, computes Cosine Similarity matrix,
generates top-K recommendation lookups, and persists model artifacts.
"""

import os
import pickle
import time
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def train_recommendation_models(
    data_path="data/cleaned_movies.pkl",
    model_dir="models",
    max_features=6000,
    top_k=50
):
    start_time = time.time()
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"Loading cleaned dataset from {data_path}...")
    df = pd.read_pickle(data_path)
    num_movies = len(df)
    print(f"Loaded {num_movies} movies.")
    
    print(f"Training TF-IDF Vectorizer (max_features={max_features}, ngram_range=(1,2))...")
    tfidf = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True
    )
    
    tfidf_matrix = tfidf.fit_transform(df['tags_stemmed'])
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    
    print("Computing Cosine Similarity matrix...")
    # Compute similarity in float32 to conserve memory
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix).astype(np.float32)
    print(f"Cosine similarity matrix shape: {cosine_sim.shape}, dtype: {cosine_sim.dtype}")
    
    print(f"Precomputing top-{top_k} nearest neighbors for ultra-fast inference...")
    top_similarities = {}
    
    for idx in range(num_movies):
        # Similarity vector for movie at idx
        sim_scores = cosine_sim[idx]
        # Get top indices (excluding self at idx)
        # argsort gives ascending order, so take [-top_k-1 : -1][::-1]
        ranked_indices = np.argsort(sim_scores)[::-1]
        # Filter out self
        ranked_indices = [i for i in ranked_indices if i != idx][:top_k]
        
        scores = [float(sim_scores[i]) for i in ranked_indices]
        top_similarities[idx] = {
            "indices": ranked_indices,
            "scores": scores
        }
    
    # Save artifacts
    print(f"Persisting model artifacts in '{model_dir}'...")
    
    # 1. Cleaned movie metadata dataframe
    movies_meta_path = os.path.join(model_dir, "movies_meta.pkl")
    with open(movies_meta_path, "wb") as f:
        pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    # 2. Top-K lookup dictionary (lightweight, instant inference)
    top_sim_path = os.path.join(model_dir, "top_similarities.pkl")
    with open(top_sim_path, "wb") as f:
        pickle.dump(top_similarities, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    # 3. TF-IDF vectorizer (for free-text plot search)
    tfidf_model_path = os.path.join(model_dir, "tfidf_vectorizer.pkl")
    with open(tfidf_model_path, "wb") as f:
        pickle.dump(tfidf, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    # 4. TF-IDF sparse matrix (for query similarity)
    tfidf_matrix_path = os.path.join(model_dir, "tfidf_matrix.pkl")
    with open(tfidf_matrix_path, "wb") as f:
        pickle.dump(tfidf_matrix, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    # 5. Full cosine similarity matrix (compressed npz)
    sim_matrix_path = os.path.join(model_dir, "similarity_matrix.npz")
    np.savez_compressed(sim_matrix_path, matrix=cosine_sim)
    
    elapsed = time.time() - start_time
    print(f"\n[DONE] Training complete in {elapsed:.2f} seconds!")
    print(f"Artifacts saved:")
    print(f"  - {movies_meta_path} ({os.path.getsize(movies_meta_path) / 1024:.1f} KB)")
    print(f"  - {top_sim_path} ({os.path.getsize(top_sim_path) / 1024:.1f} KB)")
    print(f"  - {tfidf_model_path} ({os.path.getsize(tfidf_model_path) / 1024:.1f} KB)")
    print(f"  - {tfidf_matrix_path} ({os.path.getsize(tfidf_matrix_path) / 1024:.1f} KB)")
    print(f"  - {sim_matrix_path} ({os.path.getsize(sim_matrix_path) / (1024*1024):.2f} MB)")
    
    # Run test verification
    print("\n--- Test Verification ---")
    test_titles = ["The Dark Knight", "Avatar", "Inception", "Interstellar"]
    for title in test_titles:
        matches = df[df['title'].str.lower() == title.lower()]
        if not matches.empty:
            test_idx = matches.index[0]
            recs = top_similarities[test_idx]["indices"][:5]
            rec_scores = top_similarities[test_idx]["scores"][:5]
            print(f"\nTop 5 Recommendations for '{df.loc[test_idx, 'title']}':")
            for rank, (r_idx, score) in enumerate(zip(recs, rec_scores), start=1):
                rec_title = df.loc[r_idx, 'title']
                rec_genres = ", ".join(df.loc[r_idx, 'genres_display'])
                print(f"  {rank}. {rec_title} (Score: {score:.3f}) [{rec_genres}]")

if __name__ == "__main__":
    train_recommendation_models()

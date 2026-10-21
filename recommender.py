"""
Movie Recommender Engine.
Provides high-performance recommendation querying:
- Content-based recommendations by movie title with fuzzy search
- Hybrid scoring (content similarity + IMDb weighted rating)
- Filtering by genre, release year, minimum rating
- Natural language plot / theme search
"""

import os
import pickle
import difflib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer

class MovieRecommender:
    def __init__(self, model_dir="models"):
        self.model_dir = model_dir
        self.stemmer = PorterStemmer()
        self._load_artifacts()
        
    def _load_artifacts(self):
        movies_path = os.path.join(self.model_dir, "movies_meta.pkl")
        top_sim_path = os.path.join(self.model_dir, "top_similarities.pkl")
        tfidf_path = os.path.join(self.model_dir, "tfidf_vectorizer.pkl")
        tfidf_mat_path = os.path.join(self.model_dir, "tfidf_matrix.pkl")
        
        if not all(os.path.exists(p) for p in [movies_path, top_sim_path, tfidf_path, tfidf_mat_path]):
            raise FileNotFoundError("Model artifacts not found. Please run train_model.py first.")
            
        with open(movies_path, "rb") as f:
            self.df = pickle.load(f)
            
        with open(top_sim_path, "rb") as f:
            self.top_similarities = pickle.load(f)
            
        with open(tfidf_path, "rb") as f:
            self.tfidf = pickle.load(f)
            
        with open(tfidf_mat_path, "rb") as f:
            self.tfidf_matrix = pickle.load(f)
            
        # Build normalized weighted rating for hybrid scoring
        wr = self.df['weighted_rating'].values
        min_wr, max_wr = wr.min(), wr.max()
        if max_wr > min_wr:
            self.norm_wr = (wr - min_wr) / (max_wr - min_wr)
        else:
            self.norm_wr = np.zeros_like(wr)
            
        # Title index mapping for fast exact and lowercase lookup
        self.title_to_idx = {title.lower().strip(): idx for idx, title in enumerate(self.df['title'])}
        self.all_titles = self.df['title'].tolist()

    def find_movie(self, title_query):
        """
        Finds the closest movie index and title matching the query.
        Uses exact match first, then case-insensitive, then difflib fuzzy match.
        """
        clean_query = str(title_query).lower().strip()
        
        # 1. Exact match
        if clean_query in self.title_to_idx:
            idx = self.title_to_idx[clean_query]
            return idx, self.df.loc[idx, 'title']
            
        # 2. Substring match
        for key, idx in self.title_to_idx.items():
            if clean_query == key or clean_query in key:
                return idx, self.df.loc[idx, 'title']
                
        # 3. Fuzzy match
        close_matches = difflib.get_close_matches(title_query, self.all_titles, n=1, cutoff=0.45)
        if close_matches:
            match_title = close_matches[0]
            idx = self.title_to_idx[match_title.lower().strip()]
            return idx, match_title
            
        return None, None

    def recommend(
        self,
        movie_title,
        top_n=10,
        genre_filter=None,
        min_year=None,
        max_year=None,
        min_rating=0.0,
        hybrid_weight=0.0
    ):
        """
        Returns top_n movie recommendations for a given movie title.
        
        Args:
            movie_title (str): Title or partial title of query movie.
            top_n (int): Number of recommendations to return.
            genre_filter (str): Filter by genre (e.g. 'Action', 'Sci-Fi').
            min_year (int): Minimum release year.
            max_year (int): Maximum release year.
            min_rating (float): Minimum vote average (0-10).
            hybrid_weight (float): 0.0 for pure content similarity, up to 1.0 for quality/popularity boost.
            
        Returns:
            dict: {
                'query_movie': dict of matched movie metadata,
                'recommendations': list of dicts with recommended movies
            }
        """
        idx, matched_title = self.find_movie(movie_title)
        if idx is None:
            return {
                'error': f"Movie '{movie_title}' not found in database. Try another title.",
                'query_movie': None,
                'recommendations': []
            }
            
        query_movie_data = self._format_movie_dict(idx)
        
        # Retrieve precomputed candidates
        candidate_info = self.top_similarities.get(idx, None)
        if not candidate_info:
            return {'query_movie': query_movie_data, 'recommendations': []}
            
        cand_indices = candidate_info["indices"]
        cand_sims = candidate_info["scores"]
        
        results = []
        for c_idx, sim in zip(cand_indices, cand_sims):
            row = self.df.iloc[c_idx]
            
            # Apply filters
            if min_rating > 0 and row['vote_average'] < min_rating:
                continue
            if min_year and row['release_year'] > 0 and row['release_year'] < min_year:
                continue
            if max_year and row['release_year'] > 0 and row['release_year'] > max_year:
                continue
            if genre_filter:
                genres = [g.lower() for g in row['genres_display']]
                if genre_filter.lower() not in genres:
                    continue
                    
            # Calculate composite hybrid score
            # hybrid_score = (1 - alpha) * similarity + alpha * normalized_rating
            norm_rating = float(self.norm_wr[c_idx])
            final_score = (1.0 - hybrid_weight) * sim + hybrid_weight * norm_rating
            
            rec_dict = self._format_movie_dict(c_idx)
            rec_dict['similarity_score'] = round(float(sim), 4)
            rec_dict['final_score'] = round(float(final_score), 4)
            results.append(rec_dict)
            
        # Re-sort if hybrid weight applied
        if hybrid_weight > 0:
            results.sort(key=lambda x: x['final_score'], reverse=True)
            
        return {
            'query_movie': query_movie_data,
            'recommendations': results[:top_n]
        }

    def search_by_plot(self, query_text, top_n=10, genre_filter=None, min_rating=0.0):
        """
        Recommends movies based on natural language plot or theme description.
        Vectorizes the query using trained TF-IDF and computes cosine similarity with all movies.
        """
        if not query_text or not query_text.strip():
            return []
            
        # Clean and stem query words
        words = query_text.lower().split()
        stemmed_query = " ".join([self.stemmer.stem(w) for w in words])
        
        query_vec = self.tfidf.transform([stemmed_query])
        sim_scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        ranked_indices = np.argsort(sim_scores)[::-1]
        
        results = []
        for idx in ranked_indices:
            score = float(sim_scores[idx])
            if score <= 0.01:
                break
            row = self.df.iloc[idx]
            
            if min_rating > 0 and row['vote_average'] < min_rating:
                continue
            if genre_filter:
                genres = [g.lower() for g in row['genres_display']]
                if genre_filter.lower() not in genres:
                    continue
                    
            rec_dict = self._format_movie_dict(idx)
            rec_dict['similarity_score'] = round(score, 4)
            results.append(rec_dict)
            if len(results) >= top_n:
                break
                
        return results

    def _format_movie_dict(self, idx):
        row = self.df.iloc[idx]
        return {
            'movie_id': int(row['movie_id']),
            'title': str(row['title']),
            'release_year': int(row['release_year']) if row['release_year'] > 0 else 'N/A',
            'runtime': int(row['runtime']) if row['runtime'] > 0 else 'N/A',
            'vote_average': float(row['vote_average']),
            'vote_count': int(row['vote_count']),
            'genres': list(row['genres_display']),
            'cast': list(row['cast_display']),
            'director': list(row['director_display']),
            'overview': str(row['overview']),
            'tagline': str(row['tagline']),
        }

if __name__ == "__main__":
    recommender = MovieRecommender()
    
    print("Testing Movie-to-Movie Recommendation:")
    res = recommender.recommend("interstellar", top_n=5)
    print(f"\nQuery: {res['query_movie']['title']} ({res['query_movie']['release_year']})")
    print(f"Genres: {res['query_movie']['genres']}")
    print("Top Recommendations:")
    for i, r in enumerate(res['recommendations'], 1):
        print(f" {i}. {r['title']} ({r['release_year']}) - Sim: {r['similarity_score']} - Rating: {r['vote_average']}/10")
        
    print("\nTesting Free-Text Plot Search:")
    query = "alien lifeform found in deep space threatening crew members"
    print(f"Query: '{query}'")
    search_res = recommender.search_by_plot(query, top_n=5)
    for i, r in enumerate(search_res, 1):
        print(f" {i}. {r['title']} ({r['release_year']}) - Score: {r['similarity_score']} - Genres: {r['genres']}")

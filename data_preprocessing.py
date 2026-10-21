"""
Data Preprocessing Pipeline for TMDB 5000 Movie Recommendation System.
Loads raw datasets, parses JSON metadata, handles missing values, cleans tokens,
applies stemming, computes weighted ratings, and prepares cleaned data artifacts.
"""

import os
import json
import ast
import pandas as pd
import numpy as np
from nltk.stem.porter import PorterStemmer

def load_raw_data(data_dir="data"):
    movies_path = os.path.join(data_dir, "tmdb_5000_movies.csv")
    credits_path = os.path.join(data_dir, "tmdb_5000_credits.csv")
    
    if not os.path.exists(movies_path) or not os.path.exists(credits_path):
        raise FileNotFoundError(f"Missing data files in {data_dir}. Expected tmdb_5000_movies.csv and tmdb_5000_credits.csv")
    
    print(f"Loading raw data from {movies_path} and {credits_path}...")
    movies = pd.read_csv(movies_path)
    credits = pd.read_csv(credits_path)
    
    print(f"Movies shape: {movies.shape}")
    print(f"Credits shape: {credits.shape}")
    return movies, credits

def safe_parse_json(val):
    if pd.isna(val):
        return []
    if isinstance(val, list):
        return val
    try:
        return json.loads(val)
    except Exception:
        try:
            return ast.literal_eval(val)
        except Exception:
            return []

def extract_genres(json_str):
    items = safe_parse_json(json_str)
    readable = [item['name'] for item in items if 'name' in item]
    tokens = [name.replace(" ", "").lower() for name in readable]
    return readable, tokens

def extract_keywords(json_str):
    items = safe_parse_json(json_str)
    readable = [item['name'] for item in items if 'name' in item]
    tokens = [name.replace(" ", "").lower() for name in readable]
    return readable, tokens

def extract_cast(json_str, top_k=4):
    items = safe_parse_json(json_str)
    top_items = items[:top_k]
    readable = [item['name'] for item in top_items if 'name' in item]
    tokens = [name.replace(" ", "").lower() for name in readable]
    return readable, tokens

def extract_director(json_str):
    items = safe_parse_json(json_str)
    readable = []
    for item in items:
        if item.get("job") == "Director" and "name" in item:
            readable.append(item["name"])
    tokens = [name.replace(" ", "").lower() for name in readable]
    return readable, tokens

def compute_weighted_rating(df, percentile=0.70):
    """
    IMDb Weighted Rating formula:
    WR = (v / (v + m)) * R + (m / (v + m)) * C
    where:
    v = vote_count
    m = minimum votes required to be listed (percentile threshold)
    R = vote_average
    C = mean vote across whole report
    """
    v = df['vote_count']
    R = df['vote_average']
    C = df['vote_average'].mean()
    m = df['vote_count'].quantile(percentile)
    
    wr = (v / (v + m)) * R + (m / (v + m)) * C
    return wr

def clean_data(data_dir="data", output_dir="data"):
    movies, credits = load_raw_data(data_dir)
    
    # Merge datasets on movie ID
    # Note: 'id' in movies corresponds to 'movie_id' in credits
    df = movies.merge(credits, left_on="id", right_on="movie_id", suffixes=("", "_credits"))
    print(f"Merged dataset shape: {df.shape}")
    
    # Select and standardize essential columns
    # Handle missing values
    df['overview'] = df['overview'].fillna("").astype(str)
    df['tagline'] = df['tagline'].fillna("").astype(str)
    df['release_date'] = df['release_date'].fillna("")
    df['runtime'] = df['runtime'].fillna(0).astype(int)
    df['vote_average'] = df['vote_average'].fillna(0.0)
    df['vote_count'] = df['vote_count'].fillna(0).astype(int)
    df['popularity'] = df['popularity'].fillna(0.0)
    
    # Extract release year
    df['release_year'] = df['release_date'].apply(
        lambda x: int(str(x).split("-")[0]) if str(x).split("-")[0].isdigit() else 0
    )
    
    print("Extracting metadata (genres, keywords, cast, crew)...")
    genre_data = df['genres'].apply(extract_genres)
    df['genres_display'] = genre_data.apply(lambda x: x[0])
    df['genres_tokens'] = genre_data.apply(lambda x: x[1])
    
    keyword_data = df['keywords'].apply(extract_keywords)
    df['keywords_display'] = keyword_data.apply(lambda x: x[0])
    df['keywords_tokens'] = keyword_data.apply(lambda x: x[1])
    
    cast_data = df['cast'].apply(lambda x: extract_cast(x, top_k=4))
    df['cast_display'] = cast_data.apply(lambda x: x[0])
    df['cast_tokens'] = cast_data.apply(lambda x: x[1])
    
    crew_data = df['crew'].apply(extract_director)
    df['director_display'] = crew_data.apply(lambda x: x[0])
    df['director_tokens'] = crew_data.apply(lambda x: x[1])
    
    # Compute IMDb weighted rating for hybrid recommendations
    df['weighted_rating'] = compute_weighted_rating(df)
    
    # Clean text overview
    def clean_overview(text):
        return text.lower().replace("\n", " ").strip()
    
    df['overview_clean'] = df['overview'].apply(clean_overview)
    
    # Build combined tags for recommendation model
    print("Building composite tags for NLP vectorization...")
    def build_tags(row):
        parts = [row['overview_clean']]
        if row['genres_tokens']:
            parts.append(" ".join(row['genres_tokens']))
        if row['keywords_tokens']:
            parts.append(" ".join(row['keywords_tokens']))
        if row['cast_tokens']:
            parts.append(" ".join(row['cast_tokens']))
        if row['director_tokens']:
            parts.append(" ".join(row['director_tokens']))
        return " ".join(parts)
    
    df['tags'] = df.apply(build_tags, axis=1)
    
    # Stemming
    print("Applying PorterStemmer to tags...")
    stemmer = PorterStemmer()
    def stem_text(text):
        words = text.split()
        return " ".join([stemmer.stem(w) for w in words])
    
    df['tags_stemmed'] = df['tags'].apply(stem_text)
    
    # Select clean output columns
    output_cols = [
        'id', 'title', 'release_year', 'runtime', 'vote_average', 'vote_count',
        'popularity', 'weighted_rating', 'overview', 'tagline',
        'genres_display', 'genres_tokens',
        'keywords_display', 'keywords_tokens',
        'cast_display', 'cast_tokens',
        'director_display', 'director_tokens',
        'tags', 'tags_stemmed'
    ]
    
    cleaned_df = df[output_cols].copy()
    cleaned_df.rename(columns={'id': 'movie_id'}, inplace=True)
    
    # Remove any duplicate movie IDs
    cleaned_df.drop_duplicates(subset=['movie_id'], inplace=True)
    cleaned_df.reset_index(drop=True, inplace=True)
    
    os.makedirs(output_dir, exist_ok=True)
    csv_out = os.path.join(output_dir, "cleaned_movies.csv")
    pkl_out = os.path.join(output_dir, "cleaned_movies.pkl")
    
    # For CSV, serialize list columns as json
    csv_df = cleaned_df.copy()
    for col in ['genres_display', 'genres_tokens', 'keywords_display', 'keywords_tokens',
                'cast_display', 'cast_tokens', 'director_display', 'director_tokens']:
        csv_df[col] = csv_df[col].apply(json.dumps)
        
    csv_df.to_csv(csv_out, index=False)
    cleaned_df.to_pickle(pkl_out)
    
    print(f"\n[DONE] Successfully cleaned dataset:")
    print(f"Total movies processed: {len(cleaned_df)}")
    print(f"Saved CSV: {csv_out} ({os.path.getsize(csv_out) / 1024:.1f} KB)")
    print(f"Saved Pickle: {pkl_out} ({os.path.getsize(pkl_out) / 1024:.1f} KB)")
    
    return cleaned_df

if __name__ == "__main__":
    clean_data()

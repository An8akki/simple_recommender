"""
Unit tests for the Movie Recommendation System.
"""

import unittest
import os
import pandas as pd
from recommender import MovieRecommender

class TestMovieRecommender(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recommender = MovieRecommender(model_dir="models")
        
    def test_artifacts_exist(self):
        self.assertTrue(os.path.exists("models/movies_meta.pkl"))
        self.assertTrue(os.path.exists("models/top_similarities.pkl"))
        self.assertTrue(os.path.exists("models/tfidf_vectorizer.pkl"))
        self.assertTrue(os.path.exists("models/tfidf_matrix.pkl"))
        self.assertGreater(len(self.recommender.df), 4000)
        
    def test_movie_lookup_exact_and_fuzzy(self):
        idx, title = self.recommender.find_movie("The Dark Knight")
        self.assertIsNotNone(idx)
        self.assertEqual(title, "The Dark Knight")
        
        # Test lowercase
        idx2, title2 = self.recommender.find_movie("inception")
        self.assertIsNotNone(idx2)
        self.assertEqual(title2, "Inception")
        
        # Test fuzzy match
        idx3, title3 = self.recommender.find_movie("dark knight rises")
        self.assertIsNotNone(idx3)
        self.assertIn("Dark Knight", title3)
        
    def test_recommendations_count_and_keys(self):
        res = self.recommender.recommend("Avatar", top_n=5)
        self.assertIn("query_movie", res)
        self.assertIn("recommendations", res)
        self.assertEqual(len(res["recommendations"]), 5)
        self.assertEqual(res["query_movie"]["title"], "Avatar")
        
        first_rec = res["recommendations"][0]
        self.assertIn("title", first_rec)
        self.assertIn("similarity_score", first_rec)
        self.assertIn("genres", first_rec)
        self.assertGreater(first_rec["similarity_score"], 0)
        
    def test_genre_filter(self):
        res = self.recommender.recommend("Interstellar", top_n=5, genre_filter="Drama")
        for rec in res["recommendations"]:
            genres_lower = [g.lower() for g in rec["genres"]]
            self.assertIn("drama", genres_lower)
            
    def test_search_by_plot(self):
        results = self.recommender.search_by_plot("time travel black hole physics", top_n=5)
        self.assertGreater(len(results), 0)
        titles = [r["title"] for r in results]
        self.assertTrue(any("Interstellar" in t or "Time" in t or "Space" in t for t in titles))
        
    def test_unknown_movie_graceful_handling(self):
        res = self.recommender.recommend("xyz123nonsensemovie9999", top_n=5)
        # Should either return error or empty recommendations without throwing an unhandled exception
        self.assertTrue("error" in res or len(res["recommendations"]) == 0)

if __name__ == "__main__":
    unittest.main()

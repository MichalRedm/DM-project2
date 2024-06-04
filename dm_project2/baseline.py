import pandas as pd
from preprocessing import MovieLensDataset


class BaselinePredictor:

    _dataset: MovieLensDataset

    def __init__(self) -> None:
        self._dataset = None
    
    def fit(self, dataset: MovieLensDataset):
        self._dataset = dataset
    
    def predict(self, user_id: int, movie_id: int, alpha: float = 0.5) -> float:
        ratings = self._dataset.get_ratings()
        movie_avg = ratings[ratings.index.get_level_values('movieId') == movie_id]['rating'].mean()
        if pd.isna(movie_avg):
            movie_avg = 3.5
        user_avg = ratings[ratings.index.get_level_values('userId') == user_id]['rating'].mean()
        return movie_avg * alpha + user_avg * (1 - alpha)

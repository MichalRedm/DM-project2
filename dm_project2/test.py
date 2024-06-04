import sys
import random
import pandas as pd
from dataset import MovieLensDataset
from preprocessing import MovieLensDatasetPreprocessor
from predict import Predictor
from sklearn.metrics import mean_squared_error
from typing import List, Tuple


def baseline_predictor(user_id, movie_id, dataset: MovieLensDataset, alpha: float = 0.5) -> float:
    ratings = dataset.get_ratings()
    movie_avg = ratings[ratings.index.get_level_values('movieId') == movie_id]['rating'].mean()
    if pd.isna(movie_avg):
        movie_avg = 3.5
    user_avg = ratings[ratings.index.get_level_values('userId') == user_id]['rating'].mean()
    return movie_avg * alpha + user_avg * (1 - alpha)


def main(args: List[str]) -> None:

    random.seed(42)

    dataset_name = args[0]
    sample_size = int(args[1])

    dataset = MovieLensDataset(dataset_name)

    sample: List[Tuple[int, int, float]] = []

    for (user_id, movie_id), (rating, _) in dataset.get_ratings().sample(sample_size).iterrows():
        dataset.delete_rating(user_id, movie_id)
        sample.append((user_id, movie_id, rating))

    preprocessor = MovieLensDatasetPreprocessor().fit_transform(dataset)
    predictor = Predictor()
    predictor.fit(preprocessor)

    y_true: List[float] = []
    y_pred_model: List[float] = []
    y_pred_base: List[float] = []
        
    for user_id, movie_id, rating in sample:
        prediction = predictor.predict(user_id, movie_id)
        baseline_prediction = baseline_predictor(user_id, movie_id, dataset)
        y_true.append(rating)
        y_pred_model.append(prediction)
        y_pred_base.append(baseline_prediction)

    mse_model = mean_squared_error(y_true, y_pred_model)
    mse_baseline = mean_squared_error(y_true, y_pred_base)

    var = dataset.get_ratings()['rating'].var()

    print(f'Model MSE:    {mse_model : .4f} (standarized: {mse_model / var : .4f})')
    print(f'Baseline MSE: {mse_baseline : .4f} (standarized: {mse_baseline / var : .4f})')


if __name__ == "__main__":
    main(sys.argv[1:])

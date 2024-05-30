import pandas as pd
from dataset import MovieLensDataset
from preprocessing import Preprocessing
from mlxtend.frequent_patterns import fpgrowth, association_rules
from sklearn.metrics import mean_squared_error


def baseline_predictor(user_id, movie_id, dataset_name = 'ml-latest-small') -> float:
    dataset = MovieLensDataset(dataset_name)
    dataset.delete_rating(user_id, movie_id)
    ratings = dataset.get_ratings()
    return ratings[ratings.index.get_level_values('userId') == user_id]['rating'].mean()


def predict(
        user_id: int,
        movie_id: int,
        dataset_name: str = 'ml-latest-small',
        treshold_itemsets: float = 0.01,
        treshold_rules: float = 0.01,
        weighted_mean_metric: str = 'confidence',
        alpha: float = 0.5,
        beta: float = 0.5
    ) -> float:

    assert weighted_mean_metric in ('antecedent support', 'consequent support', 'support', 'confidence', 'lift')
    
    dataset = MovieLensDataset(dataset_name)
    dataset.delete_rating(user_id, movie_id)
    preprocessing = Preprocessing(dataset)

    prepr = pd.get_dummies(dataset.get_ratings().join(preprocessing.movies_ohe()).drop('timestamp', axis=1).droplevel('movieId'), columns=['rating']).astype(bool)
    user_prepr = prepr[prepr.index == user_id]

    frequent_itemsets = apriori(user_prepr, min_support=treshold_itemsets, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="support", min_threshold=treshold_rules)
    rules = rules[rules['consequents'].apply(lambda x: len(x) == 1 and 'rating' in list(x)[0])]

    movie = dataset.get_movie_by_id(movie_id)
    genres = frozenset(movie.genres)

    relevant_rules = rules[rules['antecedents'] <= genres]
    relevant_rules.loc[:, 'consequents'] = relevant_rules['consequents'].apply(lambda x: float(list(x)[0].split('_')[1]))
    prediction = (relevant_rules['consequents'] * relevant_rules[weighted_mean_metric]).sum() / relevant_rules[weighted_mean_metric].sum()

    baseline_prediction = baseline_predictor(user_id, movie_id, dataset_name, alpha)

    if pd.isna(prediction):
        prediction = baseline_prediction

    return prediction * beta + baseline_prediction * (1 - beta)


def main() -> None:

    dataset = MovieLensDataset('ml-latest-small')

    y_true = []
    y_pred_model = []
    y_pred_base = []
        
    for (user_id, movie_id), (rating, _) in dataset.get_ratings().sample(100).iterrows():
        prediction = predict(user_id, movie_id, weighted_mean_metric='confidence')
        baseline_prediction = baseline_predictor(user_id, movie_id)
        y_true.append(rating)
        y_pred_model.append(prediction)
        y_pred_base.append(baseline_prediction)

    print('Model:', mean_squared_error(y_true, y_pred_model))
    print('Baseline:', mean_squared_error(y_true, y_pred_base))


if __name__ == "__main__":
    main()

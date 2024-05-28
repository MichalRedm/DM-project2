import pandas as pd
from sklearn.cluster import KMeans
from preprocessing import Preprocessing
from functools import cache
from typing import List


class Clustering:

    _preprocessing: Preprocessing

    def __init__(self, preprocessing: Preprocessing) -> None:
        self._preprocessing = preprocessing

    @staticmethod
    def get_cluster(data: pd.DataFrame, id: int, n_clusters: int = 8) -> List[int]:
        kmeans = KMeans(n_clusters)
        kmeans.fit(data)
        kmeans_df = pd.DataFrame({
            "cluster": kmeans.labels_
        }, index=data.index)
        cluster = kmeans_df.loc[id]['cluster']
        return kmeans_df[kmeans_df['cluster'] == cluster].index.tolist()
    
    def get_movie_cluster(self, movie_id: int, n_clusters: int = 8) -> List[int]:
        movies = self._preprocessing.preprocess_movies()
        return Clustering.get_cluster(movies, movie_id, n_clusters)
    
    def get_user_cluster(self, user_id: int, movie_id: int, user_n_clusters: int = 8, movie_n_clusters: int = 8) -> List[int]:
        movie_cluster = self.get_movie_cluster(movie_id, movie_n_clusters)
        users = self._preprocessing.preprocess_users(movie_cluster, user_id)
        return Clustering.get_cluster(users, user_id, user_n_clusters)
    
    # @cache
    # def cluster_movies(self, n_clusters: int = 15) -> pd.DataFrame:
    #     movies_sd = self._preprocessing.preprocess_movies()
    #     kmeans = KMeans(n_clusters)
    #     kmeans.fit(movies_sd)
    #     return pd.DataFrame({
    #         'movie_id': movies_sd.index,
    #         'label': kmeans.labels_
    #     }).set_index('movie_id')
    
    # def get_movie_cluster(self, movie_id: int) -> List[int]:
    #     movie_clusters = self.cluster_movies()
    #     label = movie_clusters.loc[movie_id]['label']
    #     return movie_clusters.index[movie_clusters['label'] == label].to_list()

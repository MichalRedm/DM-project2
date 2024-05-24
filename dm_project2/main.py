import sys
import logging
from typing import List
from dataset import MovieLensDataset, InvalidMovieException, InvalidUserException

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def main(args: List[str]) -> None:

    if len(args) != 2:
        print('Invalid number of arguments.')
        exit(1)
    
    user_id = int(args[0])
    movie_id = int(args[1])

    dataset = MovieLensDataset()

    try:
        movie = dataset.get_movie_by_id(movie_id)
    except InvalidMovieException:
        logger.error(f'There is no movie with movieId={movie_id}.')
        exit(1)
    
    logger.info(f'Estimating the rating for user with userId={user_id} for the movie with movieId={movie.id} ({movie.title}).')
    try:
        logger.info('Rating deleted from the dataset.' if dataset.delete_rating(user_id, movie_id)
                    else 'No such rating in the dataset - no necessity to remove it.')
    except InvalidUserException:
        logger.error(f'There is no user with user with userId={user_id}.')
        sys.exit(1)


if __name__ == '__main__':
    main(sys.argv[1:])

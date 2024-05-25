import sys
import logging
from typing import List
from dataset import MovieLensDataset, InvalidMovieException, InvalidUserException, InvalidDatasetException

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def predict(dataset_name: str, user_id: int, movie_id: int) -> float:

    dataset = MovieLensDataset(dataset_name)

    movie = dataset.get_movie_by_id(movie_id)
    dataset.delete_rating(user_id, movie_id)

    return 0.0


def main(args: List[str]) -> None:

    if len(args) != 3:
        print('Invalid number of arguments.')
        exit(1)
    
    dataset_name = args[0]
    user_id = int(args[1])
    movie_id = int(args[2])

    try:
        prediction = predict(dataset_name, user_id, movie_id)
    except InvalidDatasetException:
        logger.error(f'There is no dataset named {dataset_name}')
        exit(1)
    except InvalidUserException:
        logger.error(f'There is no user with userId={user_id}')
        exit(1)
    except InvalidMovieException:
        logger.error(f'There is no movie with movieId={movie_id}')
        exit(1)
    except Exception as e:
        logger.error(str(e))
        exit(1)
    else:
        logger.info(f'Predicted rating: {prediction}')


if __name__ == '__main__':
    main(sys.argv[1:])

# e524e3a976ec87d2027681ab9d909158
#
#
# https://api.themoviedb.org/3/movie/550?api_key=e524e3a976ec87d2027681ab9d909158
#
# https://www.kaggle.com/rounakbanik/the-movies-dataset#movies_metadata.csv


from surprise import SVD
from surprise import Dataset
from surprise.model_selection import cross_validate

# Load the movielens-100k dataset (download it if needed).
data = Dataset.load_builtin('movies_metadata.cvs')

# Use the famous SVD algorithm.
algo = SVD()

# Run 5-fold cross-validation and print results.
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

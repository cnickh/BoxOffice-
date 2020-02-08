# e524e3a976ec87d2027681ab9d909158
#
#
# https://api.themoviedb.org/3/movie/550?api_key=e524e3a976ec87d2027681ab9d909158
#
# https://www.kaggle.com/rounakbanik/the-movies-dataset#movies_metadata.csv


import pandas as pd
import matplotlib as plt
import numpy as np


df = pd.read_csv("movies_metadata.csv")

df.info()

df['budget'] = pd.to_numeric(df['budget'])

df.info()

X = pd.DataFrame(df.loc['revenue'])

Y = pd.DataFrame(df - df.loc['revenue'])

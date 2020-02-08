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
#df['imdb_id'] = pd.to_numeric(df['imdb_id'])
df.drop(columns=['homepage','title','poster_path','popularity','genres','original_language','original_title','spoken_languages','production_countries','video','tagline','overview','adult','status','imdb_id','release_date','production_companies'], inplace=True)
df['belongs_to_collection'].fillna(0, inplace=True)
df['belongs_to_collection'] = df['belongs_to_collection'].apply(lambda x: x if x==0 else 1)
df.dropna(inplace=True)
df['budget'] = pd.to_numeric(df['budget'])
df['id'] = pd.to_numeric(df['id'])
df['belongs_to_collection'] = pd.to_numeric(df['belongs_to_collection'])
# df = df[df.revenue > 0]
#df = df[df.budget > 0]
# df['budget:revenue'] = df['budget'] / df['revenue']
# df.dropna(inplace=True)
print(len(df))

def LinearRegression(X,Y):
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=5)

    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression(copy_X=True, fit_intercept=False, n_jobs=None, normalize=True)
    regressor.fit(X_train, Y_train)

    print(regressor.intercept_)

    v = pd.DataFrame(regressor.coef_,index=['Co-efficient']).transpose()
    w = pd.DataFrame(X.columns, columns=['Attribute'])

    coeff_df = pd.concat([w,v],axis=1, join='inner')

    Y_pred = regressor.predict(X_test)
    Y_pred = pd.DataFrame(Y_pred, columns=['Predicted'])

    from sklearn import metrics
    print("Mean Absolute Error: {}" .format(metrics.mean_absolute_error(Y_test, Y_pred)))
    return coeff_df
# print("Mean Squared Error: {}" .format(metrics.mean_squared_error(Y_test, Y_pred)))
# print("Root Mean Squared Error: {}" .format(np.sqrt(metrics.mean_squared_error(Y_test, Y_pred))))
Y = pd.DataFrame(df['revenue'])
X = pd.DataFrame(df[['budget','runtime','vote_average','vote_count','belongs_to_collection']])
result = LinearRegression(X,Y)

print("Revenue = {}*budget {}*runtime {}*vote_average +{}*vote_count +{}*belongs_to_collection" .format(result.at[0,'Co-efficient'],result.at[1,'Co-efficient'],result.at[2,'Co-efficient'],result.at[3,'Co-efficient'],result.at[4,'Co-efficient']))

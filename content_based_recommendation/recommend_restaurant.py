import pandas as pd
import unicodedata

from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from sklearn.preprocessing import MultiLabelBinarizer


#normalzing kagaznāgār - to. - kagaznagar

def norm(word):
  return unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8')

def recommend_restaurants(input_restaurant, sim_df, top_n=3):

    if input_restaurant not in sim_df.columns:
        return f"Restaurant '{input_restaurant}' not found in the data."

    sim_scores = sim_df[input_restaurant].sort_values(ascending=False)
    top_similar = sim_scores[
                  1:top_n + 1]  # Skips the first row (index 0) since it's the restaurant compared with itself.
    return top_similar

def recommend_restaurants_euclidean(restaurant_name, distance_matrix, top_n=3):
    if restaurant_name not in distance_matrix.index:
        return f"Restaurant '{restaurant_name}' not found in data."

    distances = distance_matrix.loc[restaurant_name].sort_values()

    # Exclude the restaurant itself (distance = 0)
    recommendations = distances[1:top_n+1]
    return recommendations

def recommend_restaurant():
    df = pd.read_csv("restaurant.csv")
    df['City'] = df['City'].apply(str.lower)
    # print(df.info())
    df.columns = ['restaurant', 'city', 'cuisine']
    lat_lon = pd.read_csv("lat_lon.csv")
    lat_lon['city'] = lat_lon['city'].apply(str.lower)
    # print(lat_lon.head())
    # print(lat_lon["city"].unique())
    lat_lon['city'] = lat_lon['city'].apply(norm)
    df = pd.merge(df, lat_lon, on='city', how='left')
    # print(df.info())
    df.dropna(inplace=True)
    # print(df.shape)
    # print(df.head())
    df["cuisine_list"] = df["cuisine"].apply(lambda x: [c.strip() for c in x.split(',')])
    print(df["cuisine_list"])
    mlb = MultiLabelBinarizer()
    cuisine_encoded = pd.DataFrame(mlb.fit_transform(df['cuisine_list']), columns=mlb.classes_, index=df.index)

    """
    
    df['cuisine_list'] contains a list of cuisines for each restaurant (e.g., ['North Indian', 'Chinese']).

    MultiLabelBinarizer() is used to convert multi-label categorical data into a binary matrix (i.e., one-hot encoded format).

    fit_transform() learns all unique cuisines and transforms the list into columns of 1s and 0s.

    Each column represents a cuisine. A value of 1 means that restaurant serves that cuisine, 0 means it does n’t.

    The result is stored in cuisine_encoded, a new DataFrame with the same index as the original df.
    
    """
    df_final = pd.concat([df.drop(columns=['cuisine_list']), cuisine_encoded], axis=1)
    df_final.drop(["cuisine", "city"], axis=1, inplace=True)
    df_final.set_index("restaurant", inplace=True)
    # print(df_final.head())
    cosine_sim_matrix = cosine_similarity(df_final)
    cosine_sim_df = pd.DataFrame(cosine_sim_matrix, index=df_final.index, columns=df_final.index)
    # print(cosine_sim_df.head())
    input_restaurant = "Galaxy Surat"
    recommendations = recommend_restaurants(input_restaurant, cosine_sim_df)
    print(f"Top 3 similar restaurants to '{input_restaurant}':\n")
    print(recommendations)

    # Euclidean distance matrix
    euclidean_dist = pairwise_distances(df_final, metric='euclidean')
    euclidean_df = pd.DataFrame(euclidean_dist, index=df_final.index, columns=df_final.index)
    euclidean_df.head()
    recommendations = recommend_restaurants_euclidean("Galaxy Surat", euclidean_df, top_n=3)
    print("Top 3 similar restaurants using Euclidean Distance:\n", recommendations)


if __name__=='__main__':
    recommend_restaurant()
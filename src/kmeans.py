import pandas as pd

from sklearn.cluster import KMeans

def kmeans(df, feature_keys):
    train_df = df.reset_index(drop=False)
    features = train_df[feature_keys]
    kmeans = KMeans(n_clusters=2).fit(features)
    return kmeans

def evaluate_kmeans_model(model, features, classes):
    kmeans_predicitons = model.predict(features)
    predictions_df = pd.DataFrame()
    predictions_df['age_group'] = classes
    predictions_df['cluster'] = kmeans_predicitons
    accuracy = (1-(predictions_df.groupby('age_group')[['cluster']].agg('mean').sum().values[0]/2))
    return accuracy



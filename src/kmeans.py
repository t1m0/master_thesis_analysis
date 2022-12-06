from sklearn.cluster import KMeans

def kmeans(df, feature_keys):
    train_df = df.reset_index(drop=False)
    features = train_df[feature_keys]
    kmeans = KMeans(n_clusters=2).fit(features)
    return kmeans
from sklearn.neighbors import KNeighborsClassifier

def k_neighbors(df, feature_keys, neighbors=3):
    train_df = df.reset_index(drop=False)
    features = train_df[feature_keys]
    target = train_df['age_group']
    model = KNeighborsClassifier(n_neighbors=neighbors)
    model = model.fit(features,target)
    return model
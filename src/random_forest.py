from sklearn.ensemble import RandomForestClassifier

def random_forest(df, feature_keys):
    train_df = df.reset_index(drop=False)
    features = train_df[feature_keys]
    target = train_df['age_group']
    clf = RandomForestClassifier()
    clf = clf.fit(features,target)
    return clf

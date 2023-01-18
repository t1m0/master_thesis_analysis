from sklearn.ensemble import RandomForestClassifier

def random_forest(df, feature_keys):
    train_df = df.reset_index(drop=False)
    features = train_df[feature_keys]
    target = train_df['age_group']
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf = clf.fit(features,target)
    return clf

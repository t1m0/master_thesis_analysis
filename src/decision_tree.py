from sklearn.tree import DecisionTreeClassifier

def decision_tree(df, feature_keys):
    train_df = df.reset_index(drop=False)
    features = train_df[feature_keys]
    target = train_df['age_group']
    clf = DecisionTreeClassifier()
    clf = clf.fit(features,target)
    return clf

from sklearn.linear_model import SGDClassifier

def stochastic_gradient_descent(df, feature_keys, neighbors=3):
    train_df = df.reset_index(drop=False)
    features = train_df[feature_keys]
    target = train_df['age_group']
    model = SGDClassifier(loss="hinge", penalty="l2", max_iter=5)
    model = model.fit(features,target)
    return model
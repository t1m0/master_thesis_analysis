from sklearn.svm import SVC

def support_vector_machine(df, feature_keys):
    train_df = df.reset_index(drop=False)
    features = train_df[feature_keys]
    target = train_df['age_group']
    model = SVC()
    model = model.fit(features,target)
    return model

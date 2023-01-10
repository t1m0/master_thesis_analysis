from sklearn.neural_network import MLPClassifier

def multi_layer_perceptron(df, feature_keys):
    features = df[feature_keys]
    target = df['age_group']
    model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1, max_iter=400)
    model = model.fit(features,target)
    return model
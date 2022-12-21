from sklearn import metrics
def evaluate_model(model, features, labels, prediction_correction = lambda x : x):
    predictions = model.predict(features)
    predictions = prediction_correction(predictions)
    return metrics.accuracy_score(labels, predictions)
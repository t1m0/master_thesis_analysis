import pandas as pd

from sklearn import metrics
from sklearn.model_selection import train_test_split

from src.kmeans import kmeans
from src.decision_tree import decision_tree

def evaluate_model(model, features, labels, prediction_correction = lambda x : x):
    predictions = model.predict(features)
    predictions = prediction_correction(predictions)
    return metrics.accuracy_score(labels, predictions)

def run_feature_algorithms(df, feature_keys):
    df_copy = df.copy()
    df_copy = df_copy.reset_index(drop=False)
    train_df, test_df = train_test_split(df_copy, test_size=0.15)

    results = {}

    print("Training KMEANS")
    kmeans_model = kmeans(train_df, feature_keys)
    kmeans_predicitons = kmeans_model.predict(test_df[feature_keys])
    predictions_df = pd.DataFrame()
    predictions_df['age_group'] = test_df['age_group']
    predictions_df['cluster'] = kmeans_predicitons
    accuracy = (1-(predictions_df.groupby('age_group')[['cluster']].agg('sem').sum()/2))
    results['kmeans'] = accuracy
    print(f"KMEANS accuracy: {accuracy}")

    print("Training Decision Tree")
    model = decision_tree(train_df, feature_keys)
    accuracy = evaluate_model(model, test_df[feature_keys],test_df['age_group'])
    results['decision_tree'] = accuracy
    print(f"Decision Tree accuracy: {accuracy}")

    return results
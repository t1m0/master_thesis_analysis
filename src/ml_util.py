
import numpy as np

from sklearn import metrics
from sklearn.model_selection import ShuffleSplit
from sklearn.dummy import DummyClassifier

from src.kmeans import kmeans, evaluate_kmeans_model
from src.decision_tree import decision_tree
from src.k_neighbors import k_neighbors
from src.support_vector_machine import support_vector_machine
from src.random_forest import random_forest

def evaluate_model(model, features, labels, prediction_correction=lambda x: x):
    predictions = model.predict(features)
    predictions = prediction_correction(predictions)
    accuracy = metrics.accuracy_score(labels, predictions)
    accuracy = accuracy * 100
    return round(accuracy, 2)

def dummy_classifier(df, feature_keys, neighbors=3):
    train_df = df.reset_index(drop=False)
    features = train_df[feature_keys]
    target = train_df['age_group']
    model = DummyClassifier(strategy="uniform")
    model.fit(features, target)
    return model

def run_feature_algorithms(df, feature_keys, cross_validations=10):
    df_copy = df.copy()
    df_copy = df_copy.reset_index(drop=False)
    
    instance_count = len(df)
    shuffle_split = ShuffleSplit(n_splits=cross_validations, test_size=0.20, random_state=0)
    results = {'dummy': [], 'kmeans': [], 'decision_tree': [], 'k_neighbors': [], 'random_forest': [], 'support_vector_machine': []}
    for train_index, test_index in shuffle_split.split(range(instance_count)):
        train_df = df_copy.iloc[train_index]
        test_df = df_copy.iloc[test_index]

        test_features = test_df[feature_keys]
        test_labels = test_df['age_group']

        # Dummy
        dummy_model = dummy_classifier(train_df, feature_keys)
        accuracy = evaluate_model(dummy_model, test_features, test_labels)
        results['dummy'].append(accuracy)

        # KMEANS
        kmeans_model = kmeans(train_df, feature_keys)
        accuracy = evaluate_kmeans_model(
            kmeans_model, test_features, test_labels)
        results['kmeans'].append(accuracy)

        # Decision Tree
        dt_model = decision_tree(train_df, feature_keys)
        accuracy = evaluate_model(dt_model, test_features, test_labels)
        results['decision_tree'].append(accuracy)

        # Random Forest
        rf_model = random_forest(train_df, feature_keys)
        accuracy = evaluate_model(rf_model, test_features, test_labels)
        results['random_forest'].append(accuracy)

        # K Neighbors
        kn_model = k_neighbors(train_df, feature_keys)
        accuracy = evaluate_model(kn_model, test_features, test_labels)
        results['k_neighbors'].append(accuracy)

        # Support Vector Machine
        svc_model = support_vector_machine(train_df, feature_keys)
        accuracy = evaluate_model(svc_model, test_features, test_labels)
        results['support_vector_machine'].append(accuracy)

    for key in results.keys():
        array = results[key]
        print(f"{key} single accuracy: {array}")
        mean_accuracy = np.mean(array)
        mean_accuracy = round(mean_accuracy,2)
        print(f"{key} mean accuracy: {mean_accuracy}")
        results[key] = mean_accuracy

    return results


import numpy as np

from sklearn import metrics
from sklearn.model_selection import ShuffleSplit

from src.kmeans import kmeans, evaluate_kmeans_model
from src.decision_tree import decision_tree
from src.k_neighbors import k_neighbors
from src.multi_layer_perceptron import multi_layer_perceptron

def evaluate_model(model, features, labels, prediction_correction = lambda x : x):
    predictions = model.predict(features)
    predictions = prediction_correction(predictions)
    return metrics.accuracy_score(labels, predictions)

def run_feature_algorithms(df, feature_keys, cross_validations=10):
    df_copy = df.copy()
    df_copy = df_copy.reset_index(drop=False)
    instance_count = len(df)
    shuffle_split = ShuffleSplit(n_splits=cross_validations, test_size=0.20, random_state=0)
    results = {'kmeans':[], 'decision_tree':[], 'k_neighbors':[], 'multi_layer_perceptron':[]}
    for train_index, test_index in shuffle_split.split(range(instance_count)):
        train_df = df_copy.iloc[train_index]
        test_df = df_copy.iloc[test_index]

    
        #KMEANS
        kmeans_model = kmeans(train_df, feature_keys)
        accuracy = evaluate_kmeans_model(kmeans_model, test_df[feature_keys],test_df['age_group'])
        results['kmeans'].append(accuracy)

        #Decision Tree
        dt_model = decision_tree(train_df, feature_keys)
        accuracy = evaluate_model(dt_model, test_df[feature_keys],test_df['age_group'])
        results['decision_tree'].append( accuracy)

        #K Neighbors
        kn_model = k_neighbors(train_df, feature_keys)
        accuracy = evaluate_model(kn_model, test_df[feature_keys],test_df['age_group'])
        results['k_neighbors'].append( accuracy)

        # Multi Layer Perceptron
        kn_model = multi_layer_perceptron(train_df, feature_keys)
        accuracy = evaluate_model(kn_model, test_df[feature_keys],test_df['age_group'])
        results['multi_layer_perceptron'].append( accuracy)

    for key in results.keys():
        array = results[key]
        print(f"{key} single accuracy: {array}")
        mean_accuracy = np.mean(array)
        print(f"{key} mean accuracy: {mean_accuracy}")
        results[key] = mean_accuracy

    return results
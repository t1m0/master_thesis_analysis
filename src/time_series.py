import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.losses import BinaryCrossentropy

from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from tslearn.svm import TimeSeriesSVC


def _preprocess_data(df, sequence_column, class_column, feature_columns):
    sequences_features = []
    sequences_length = []
    sequences_labels = []
    for uuid in df[sequence_column].unique():
        current_df = df[df[sequence_column] == uuid]
        label = current_df[class_column].unique()[0]
        sequences_labels.append(label)
        sequence = current_df[feature_columns].values
        sequences_length.append(len(sequence))
        sequences_features.append(sequence)
    return {
        'sequences': sequences_features,
        'length': sequences_length,
        'labels': sequences_labels
    }


def align_sequences_to_same_length(sequences, target_length):
    new_sequences = []
    for current_sequence in sequences:
        sequence_length = len(current_sequence)
        last_value = current_sequence[-1]
        padding_count = target_length - sequence_length
        column_count = len(current_sequence[0])
        if padding_count > 0:
            to_concat = np.repeat(last_value, padding_count).reshape(column_count, padding_count).transpose()
            new_sequence = np.concatenate([current_sequence, to_concat])
        else:
            new_sequence = current_sequence[:target_length]
        new_sequences.append(new_sequence)

    return np.stack(new_sequences)


def compile_sequences(df, sequence_column='uuid', class_column='age_group', feature_columns=['x', 'y', 'z', 'mag']):
    preprocessed_data = _preprocess_data(df, sequence_column, class_column, feature_columns)
    target_length = np.int_(pd.Series(preprocessed_data['length']).quantile(0.9))
    aligned_sequences = align_sequences_to_same_length(preprocessed_data['sequences'], target_length)
    return {
        'sequences': aligned_sequences,
        'labels': preprocessed_data['labels']
    }

def long_short_term_memory(sequences, labels):
    #https://www.analyticsvidhya.com/blog/2019/01/introduction-time-series-classification/
    model = Sequential()
    model.add(LSTM(256, input_shape=(len(sequences[0]), len(sequences[0][0]))))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss=BinaryCrossentropy(from_logits=False), optimizer='adam', metrics=['accuracy'])
    model.fit(np.array(sequences), np.array(labels), epochs=200, batch_size=128, verbose=0)

    return model

def knn_time_series(sequences, labels):
    model = KNeighborsTimeSeriesClassifier(n_neighbors=2)
    model.fit(sequences, labels)
    return model

def svc_time_series(sequences, labels):
    model = TimeSeriesSVC(C=1.0, kernel="gak")
    model.fit(sequences, labels)
    return model

def _median_filter_on_session(df, window, columns):
    df_copy = df.copy()
    for column in columns:
        df_copy[column] = df_copy[column].rolling(center=True, window=window).median()
    df_copy = df_copy.dropna()
    return df_copy

def median_filter(df, session_identifier='uuid', window=3, columns=['x','y','z','mag']):
    new_df = pd.DataFrame(data=None, columns=df.columns)
    for current_session_identifier in df[session_identifier].unique():
        current_session_df = df[df[session_identifier] == current_session_identifier]
        filtered_df = _median_filter_on_session(current_session_df,window, columns)
        new_df = pd.concat([new_df, filtered_df])
    return new_df
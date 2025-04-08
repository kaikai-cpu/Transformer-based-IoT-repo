# preprocess_cicids.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

def load_cicids_data():
    # Files
    files = [
        'data/Wednesday-workingHours.pcap_ISCX.csv',
        'data/Friday-WorkingHours-Morning.pcap_ISCX.csv',
        'data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'
    ]

    # Load and concatenate all files
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    # Strip column names to avoid leading/trailing space issues
    df.columns = df.columns.str.strip()

    # Drop unnamed columns (like 'Unnamed: 0')
    df.drop(columns=[col for col in df.columns if 'Unnamed' in col], inplace=True)

    # Replace infinite values and drop NaNs
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # Confirm label column exists and normalize its values
    if 'Label' not in df.columns:
        raise ValueError("Expected 'Label' column not found in the dataset.")

    # Label as binary: 'BENIGN' = 0, anything else = 1
    df['Label'] = df['Label'].apply(lambda x: 'normal' if x == 'BENIGN' else 'attack')

    # Encode label (normal = 0, attack = 1)
    label_encoder = LabelEncoder()
    df['Label'] = label_encoder.fit_transform(df['Label'])

    # Separate features and label
    X = df.drop('Label', axis=1)
    y = df['Label']

    # Normalize features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    return X_train, y_train, X_test, y_test

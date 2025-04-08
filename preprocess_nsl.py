import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def load_nsl_kdd():
    # File paths
    train_path = 'data/KDDTrain+.txt'
    test_path = 'data/KDDTest+.txt'

    # Column names from NSL-KDD
    col_names = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
        'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
        'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
        'num_root', 'num_file_creations', 'num_shells', 'num_access_files',
        'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
        'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
        'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
        'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
        'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
        'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
        'dst_host_srv_rerror_rate', 'label', 'difficulty'
    ]

    # Load data
    df_train = pd.read_csv(train_path, names=col_names)
    df_test = pd.read_csv(test_path, names=col_names)

    # Remove 'difficulty'
    df_train.drop('difficulty', axis=1, inplace=True)
    df_test.drop('difficulty', axis=1, inplace=True)

    # Encode categorical features
    cat_cols = ['protocol_type', 'service', 'flag']
    encoders = {col: LabelEncoder() for col in cat_cols}

    for col in cat_cols:
        df_train[col] = encoders[col].fit_transform(df_train[col])
        df_test[col] = encoders[col].transform(df_test[col])  # use same encoder

    # Binary label: normal vs. attack
    df_train['label'] = df_train['label'].apply(lambda x: 'normal' if x == 'normal' else 'attack')
    df_test['label'] = df_test['label'].apply(lambda x: 'normal' if x == 'normal' else 'attack')

    label_encoder = LabelEncoder()
    df_train['label'] = label_encoder.fit_transform(df_train['label'])
    df_test['label'] = label_encoder.transform(df_test['label'])

    # Separate features and labels
    X_train = df_train.drop('label', axis=1)
    y_train = df_train['label']
    X_test = df_test.drop('label', axis=1)
    y_test = df_test['label']

    # Normalize features
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test

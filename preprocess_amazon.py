import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_amazon_data(instance_name, data_dir="data"):
    """
    Load Amazon dataset instance (amazon1, amazon2, amazon3).
    Assumes folder structure: data/<instance_name>/train_amazonX.sample, test_amazonX.sample
    """
    train_path = os.path.join(data_dir, instance_name, f"train_{instance_name}.sample")
    test_path = os.path.join(data_dir, instance_name, f"test_{instance_name}.sample")

    def parse_file(filepath):
        features, labels = [], []
        with open(filepath, "r") as f:
            for line in f:
                parts = line.strip().split()
                # parts layout: user_id(1), resource_id(1), user_meta(8), resource_meta(?), label(1)
                user_meta = list(map(int, parts[2:10]))
                label = int(parts[-1])
                resource_meta = list(map(int, parts[10:-1]))
                feat = user_meta + resource_meta
                features.append(feat)
                labels.append(label)
        return np.array(features), np.array(labels)

    X_train, y_train = parse_file(train_path)
    X_test, y_test = parse_file(test_path)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test

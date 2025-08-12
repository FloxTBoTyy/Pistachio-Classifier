import pandas as pd
from sklearn.model_selection import train_test_split

def split_dataset_4_ways(path_csv):
    df = pd.read_csv(path_csv)
    X = df.drop('CLASS', axis=1)
    y = df['CLASS']

    X_temp, X_extra_virgen, y_temp, y_extra_virgen = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_temp2, X_virgen, y_temp2, y_virgen = train_test_split(
        X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X_temp2, y_temp2, test_size=0.243, random_state=42, stratify=y_temp2
    )

    return X_train, X_test, X_virgen, X_extra_virgen, y_train, y_test, y_virgen, y_extra_virgen

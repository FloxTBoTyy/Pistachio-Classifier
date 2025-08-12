import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from src.preprocessing.data_preprocessing import split_dataset_4_ways
from .model_evaluation import evaluate_model_complete

def train_with_complete_evaluation(path_csv):
    X_train, X_test, X_virgen, X_extra_virgen, y_train, y_test, y_virgen, y_extra_virgen = split_dataset_4_ways(path_csv)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_virgen_scaled = scaler.transform(X_virgen)
    X_extra_virgen_scaled = scaler.transform(X_extra_virgen)

    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)

    sets = [
        ("Train", X_train_scaled, y_train),
        ("Test", X_test_scaled, y_test),
        ("Virgen", X_virgen_scaled, y_virgen),
        ("Extra Virgen", X_extra_virgen_scaled, y_extra_virgen),
    ]

    results = []
    for name, X_set, y_set in sets:
        y_pred = model.predict(X_set)
        y_proba = model.predict_proba(X_set)
        results.append((name,) + evaluate_model_complete(model, X_set, y_set, y_pred, y_proba, name))

    print("\n=== RESUMEN FINAL ===")
    print(f"{'Set':<15} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'AUC-ROC':<10}")
    for res in results:
        print(f"{res[0]:<15} {res[1]:<10.4f} {res[2]:<10.4f} {res[3]:<10.4f} {res[4]:<10.4f} {res[5]:<10.4f}")

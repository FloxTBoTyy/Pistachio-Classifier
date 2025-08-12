import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

def evaluate_model_complete(model, X, y, y_pred, y_pred_proba, set_name):
    print(f"\n=== {set_name.upper()} ===")
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, pos_label='Kirmizi_Pistachio')
    recall = recall_score(y, y_pred, pos_label='Kirmizi_Pistachio')
    f1 = f1_score(y, y_pred, pos_label='Kirmizi_Pistachio')

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    auc_roc = roc_auc_score(y_encoded, y_pred_proba[:, 1])

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")
    print(f"AUC-ROC: {auc_roc:.4f}")

    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Kirmizi', 'Siirt'],
                yticklabels=['Kirmizi', 'Siirt'])
    plt.title(f'Matriz de Confusión - {set_name}')
    plt.ylabel('Real')
    plt.xlabel('Predicción')
    plt.show()

    return accuracy, precision, recall, f1, auc_roc

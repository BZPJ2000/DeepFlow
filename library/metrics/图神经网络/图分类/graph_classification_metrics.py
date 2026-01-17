from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')

def roc_auc(y_true, y_pred):
    return roc_auc_score(y_true, y_pred)

# Example usage
if __name__ == "__main__":
    y_true = [0, 1, 0, 1]
    y_pred = [0.1, 0.9, 0.2, 0.8]

    print(f"Accuracy: {accuracy(y_true, y_pred):.4f}")
    print(f"F1 Score: {f1(y_true, y_pred):.4f}")
    print(f"ROC-AUC: {roc_auc(y_true, y_pred):.4f}")
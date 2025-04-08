import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_curve
)
import matplotlib.pyplot as plt
import numpy as np

from preprocess_nsl import load_nsl_kdd
from preprocess_cicids import load_cicids_data
from transformer_model import TransformerClassifier

def train_and_evaluate(dataset_name, load_data_fn):
    print(f"\n=== Training on {dataset_name.upper()} Dataset ===")
    results_dir = f"results/{dataset_name}"
    os.makedirs(results_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    X_train, y_train, X_test, y_test = load_data_fn()

    # Tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values if hasattr(y_train, "values") else y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values if hasattr(y_test, "values") else y_test, dtype=torch.long)

    # DataLoaders
    batch_size = 64
    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=batch_size)

    # Model
    input_dim = X_train.shape[1]
    model = TransformerClassifier(input_dim=input_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training
    epochs = 10
    train_losses = []
    train_accuracies = []

    print("Training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device).unsqueeze(1), y_batch.to(device)

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)

        avg_loss = total_loss / len(train_loader)
        acc = correct / total
        train_losses.append(avg_loss)
        train_accuracies.append(acc)
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}")

    # Evaluation
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device).unsqueeze(1)
            outputs = model(X_batch)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y_batch.numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)

    print(f"\nEvaluation on {dataset_name.upper()}:")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"AUC      : {auc:.4f}")

    # Save metrics
    with open(os.path.join(results_dir, "metrics.txt"), "w") as f:
        f.write(f"Accuracy : {accuracy:.4f}\n")
        f.write(f"F1 Score : {f1:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall   : {recall:.4f}\n")
        f.write(f"AUC      : {auc:.4f}\n")

    # Classification Report
    with open(os.path.join(results_dir, "classification_report.txt"), "w") as f:
        f.write(classification_report(all_labels, all_preds, target_names=["Normal", "Attack"]))

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Attack"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - {dataset_name.upper()}")
    plt.savefig(os.path.join(results_dir, "confusion_matrix.png"))
    plt.close()

    # Loss Curve
    plt.plot(range(1, epochs + 1), train_losses, label="Training Loss", marker="o")
    plt.title(f"Loss Curve - {dataset_name.upper()}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid()
    plt.savefig(os.path.join(results_dir, "loss_curve.png"))
    plt.close()

    # Accuracy Curve
    plt.plot(range(1, epochs + 1), train_accuracies, label="Training Accuracy", marker="o", color="green")
    plt.title(f"Accuracy Curve - {dataset_name.upper()}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.savefig(os.path.join(results_dir, "accuracy_curve.png"))
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {dataset_name.upper()}")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(results_dir, "roc_curve.png"))
    plt.close()

if __name__ == "__main__":
    train_and_evaluate("nsl_kdd", load_nsl_kdd)
    train_and_evaluate("cicids", load_cicids_data)


import argparse
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from modified_resnet50 import make_resnet50_with_attention


def k_fold_train_model(model, dataset, criterion, optimizer, k, early_stopping_rounds, num_epochs, device):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    fold_results = []  # To store results for each fold
    best_model = None
    best_val_acc = 0.0

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"Fold {fold + 1}/{k}")

        # Split dataset into training and validation sets
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)

        # Reinitialize model for each fold to avoid data leakage
        fold_model = model.to(device)

        # Track best validation accuracy for early stopping
        best_acc_for_fold = 0.0
        epochs_without_improvement = 0.0

        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            fold_model.train()

            running_loss = 0.0
            running_corrects = 0
            total = 0

            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = fold_model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                total += labels.size(0)

            epoch_loss = running_loss / total
            epoch_acc = running_corrects.double() / total

            print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            val_acc = evaluate_model(fold_model, val_loader, device)
            print(f"Validation Accuracy for Epoch {epoch + 1}: {val_acc:.4f}")

            # Early stopping logic
            if val_acc > best_acc_for_fold:
                best_acc_for_fold = val_acc
                epochs_without_improvement = 0  # Reset the counter if improvement
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= early_stopping_rounds:
                    print(f"Early stopping after epoch {epoch + 1}")
                    break

            fold_results.append(best_acc_for_fold)
            if best_acc_for_fold > best_val_acc:
                best_val_acc = best_acc_for_fold
                best_model = fold_model

    # Compute the average validation accuracy across all folds
    avg_val_acc = sum(fold_results) / k
    print(f"Average Validation Accuracy: {avg_val_acc:.4f}")

    return best_model


def evaluate_model(model, dataloader, device):
    model.eval()
    preds_all = []
    labels_all = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            preds_all.extend(preds.cpu().numpy())
            labels_all.extend(labels.cpu().numpy())
    accuracy = accuracy_score(labels_all, preds_all)
    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Hyperparameters and algorithm parameters are described here
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--nfold", type=int, default=5)
    parser.add_argument("--early_stopping_rounds", type=int, default=3)

    # SageMaker specific arguments
    parser.add_argument("--train_data_dir", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--validation_data_dir", type=str, default=os.environ.get("SM_CHANNEL_VALIDATION"))
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--output_data_dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR"))

    args, _ = parser.parse_known_args()

    # Pytorch arguments
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]) # Replace with mean + std
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    }

    # Change ImageFolder if manual labels; requires dummy / labeled subfolders
    image_datasets = {
        'train': datasets.ImageFolder(args.train_data_dir, data_transforms['train']),
        'val': datasets.ImageFolder(args.validation_data_dir, data_transforms['val'])
    }

    train_data = DataLoader(image_datasets['train'], batch_size=args.batch_size, shuffle=True)
    validation_data = DataLoader(image_datasets['val'], batch_size=args.batch_size)

    # Declare arguments for cross-validation training
    num_classes = len(image_datasets['train'].classes)
    model = make_resnet50_with_attention(num_classes)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    nfold = args.nfold
    early_stopping_rounds = args.early_stopping_rounds

    model = k_fold_train_model(model, train_data, criterion, optimizer, nfold, early_stopping_rounds, args.epochs, device)

    train_acc = evaluate_model(model, train_data, device)
    val_acc = evaluate_model(model, validation_data, device)
    print(f"Final Validation Accuracy: {val_acc:.4f}")

    # Save metrics
    metrics = {
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
    }

    # Save the evaluation metrics to the location specified by output_data_dir
    metrics_location = args.output_data_dir + "/metrics.json"
    with open(metrics_location, "w") as f:
        json.dump(metrics, f)

    # Save the trained model to the location specified by model_dir
    model_location = os.path.join(args.model_dir, "resnet50_eeg.pth")
    torch.save(model.state_dict(), model_location)
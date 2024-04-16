import torch
import argparse
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_curve, roc_auc_score, roc_curve

from model import Attention, VarMIL
from dataloader import Bags, CustomDataset
from utils import initialize, update, summarize, best, initialize_fold, load_model, early_stop

parser = argparse.ArgumentParser(description='MSI/MSS MIL Classifier')
parser.add_argument('--epochs', type=int, default=25, help='Number Of Epochs (default: 50)')
parser.add_argument('--seed', type=int, default=9, help='Seed (default: 9)')
parser.add_argument('--bag_size', type=int, default=25, help='Bag Size (default: 25)')
parser.add_argument('--data_size', type=float, default=1, help='Data Size (default: 1)')
parser.add_argument('--k_fold', type=int, default=5, help='K-Fold Cross-Validation (default: 5)')
parser.add_argument('--model', type=str, default='var', help='Choose Model: att or var (default: var)')
parser.add_argument('--job', type=int, default=1, help='Job Number (default: 1)')
parser.add_argument('--loss', type=str, default='svm',help='Choose Loss: nll or svm (default: svm)')
parser.add_argument('--stop', type=int, default=10,help='Choose Early Stopping criterion: (default: 10)')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

torch.manual_seed(args.seed)
if device.type == 'cuda':
    torch.cuda.manual_seed(args.seed)

train_dataset = Bags(train=True, bag_size=args.bag_size, seed=args.seed)
train_bags, train_labels = train_dataset.bags_list, train_dataset.labels_list

test_dataset = Bags(train=False, bag_size=args.bag_size, seed=args.seed)
valid_bags, valid_labels = test_dataset.bags_list, test_dataset.labels_list

train_dataset.print_label_counts()
test_dataset.print_label_counts()


def train(epoch, model, train_loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_true_labels = []
    all_predicted_labels = []
    all_probs = []

    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()

        loss, _ = model.calculate_objective(data, labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        _, predicted_labels = model.calculate_classification_error(data, labels)
        total_correct += (predicted_labels == labels).sum().item()
        total_samples += labels.size(0)

        all_true_labels.extend(labels.cpu().numpy())
        all_predicted_labels.extend(predicted_labels.cpu().numpy())

        Y_prob, Y_hat, _ = model(data)
        prob = Y_prob[:, 1].cpu().detach().numpy() if Y_prob.ndim > 1 else Y_prob[1].cpu().item()
        all_probs.extend(prob)

    train_accuracy = 100 * total_correct / total_samples
    average_loss = total_loss / len(train_loader)

    auc_score = roc_auc_score(all_true_labels, all_probs)
    f1_score_val = f1_score(all_true_labels, all_predicted_labels)

    print(f'\nEpoch: {epoch}')
    print(f'Training Set, Loss: {average_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%')
    print(confusion_matrix(all_true_labels, all_predicted_labels))
    print(f'Training Set, AUC: {auc_score:.2f}')
    print(f'Training Set, F1 Score: {f1_score_val:.2f}')


def validation(model, valid_loader, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_true_labels = []
    all_predicted_labels = []
    all_probs = []

    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(valid_loader):
            data, labels = data.to(device), labels.to(device)

            loss, _ = model.calculate_objective(data, labels)
            total_loss += loss.item()

            _, predicted_labels = model.calculate_classification_error(data, labels)
            total_correct += (predicted_labels == labels).sum().item()
            total_samples += labels.size(0)

            all_true_labels.extend(labels.cpu().numpy())
            all_predicted_labels.extend(predicted_labels.cpu().numpy())

            Y_prob, Y_hat, _ = model(data)
            prob = Y_prob[:, 1].cpu().numpy() if Y_prob.ndim > 1 else Y_prob[1].cpu().item()
            all_probs.extend(prob)

    test_accuracy = 100 * total_correct / total_samples
    average_loss = total_loss / len(valid_loader)

    auc_score = roc_auc_score(all_true_labels, all_probs)
    f1_score_val = f1_score(all_true_labels, all_predicted_labels)

    print(f'\nValidation Set, Loss: {average_loss:.4f} Validation Accuracy: {test_accuracy:.2f}%')
    print(confusion_matrix(all_true_labels, all_predicted_labels))
    print(f'Validation Set, AUC: {auc_score:.2f}')
    print(f'Validation Set, F1 Score: {f1_score_val:.2f}')
    return auc_score, f1_score_val, test_accuracy, average_loss


def final_validation(model, valid_loader, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_true_labels = []
    all_predicted_labels = []
    all_probs = []

    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(valid_loader):
            data, labels = data.to(device), labels.to(device)

            loss, _ = model.calculate_objective(data, labels)
            total_loss += loss.item()

            _, predicted_labels = model.calculate_classification_error(data, labels)
            total_correct += (predicted_labels == labels).sum().item()
            total_samples += labels.size(0)

            all_true_labels.extend(labels.cpu().numpy())
            all_predicted_labels.extend(predicted_labels.cpu().numpy())

            Y_prob, Y_hat, _ = model(data)
            prob = Y_prob[:, 1].cpu().numpy() if Y_prob.ndim > 1 else Y_prob[1].cpu().item()
            all_probs.extend(prob)

    test_accuracy = 100 * total_correct / total_samples
    average_loss = total_loss / len(valid_loader)

    auc_score = roc_auc_score(all_true_labels, all_probs)
    f1_score_val = f1_score(all_true_labels, all_predicted_labels)
    fpr, tpr, _ = roc_curve(all_true_labels, all_probs)

    print(f'\nFinal Validation Set, Loss: {average_loss:.4f} Validation Accuracy: {test_accuracy:.2f}%')
    print(confusion_matrix(all_true_labels, all_predicted_labels))
    print(f'Final Validation Set, AUC: {auc_score:.2f}')
    print(f'Final Validation Set, F1 Score: {f1_score_val:.2f}')
    return auc_score, f1_score_val, test_accuracy, fpr, tpr, all_probs, all_true_labels, average_loss


skf = StratifiedKFold(n_splits=args.k_fold, shuffle=True, random_state=args.seed)
score_storage = initialize()

for fold_number, (train_index, test_index) in enumerate(skf.split(train_bags, train_labels), start=1):
    print(f"Fold {fold_number}:")

    initialize_fold(score_storage)

    model = Attention(loss=args.loss, input_dim=train_bags.shape[-1]) if args.model == 'att' else VarMIL(loss=args.loss, input_dim=train_bags.shape[-1])
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=10e-5)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=7, factor=0.1, verbose=True)

    train_loader = DataLoader(CustomDataset(train_bags[train_index], train_labels[train_index], args.data_size, args.seed, train=True), batch_size=2, shuffle=True, num_workers=0)
    valid_loader = DataLoader(CustomDataset(train_bags[test_index], train_labels[test_index], args.data_size, args.seed, train=False), batch_size=2, shuffle=True, num_workers=0)

    print("Train class counts:", train_loader.dataset.get_class_counts())
    print("Validation class counts:", valid_loader.dataset.get_class_counts())

    best_metrics = {'loss': float('inf'), 'epoch': None, 'vauc': 0.0, 'vf1': 0.0, 'counter': 0}
    for epoch in range(1, args.epochs + 1):
        train(epoch, model, train_loader, optimizer, device)
        vauc, vf1, test_accuracy, vloss = validation(model, valid_loader, device)

        best(vloss, vauc, vf1, model, optimizer, args, epoch, fold_number, best_metrics)

        scheduler.step(vloss)
        if early_stop(best_metrics['counter'], args.stop, epoch, vauc, vf1, vloss):
            print(f"Early stopping triggered at epoch {epoch} with validation AUC {vauc:.2f}, validation F1 {vf1:.2f}, and validation loss {vloss:.2f}")
            break

    model, optimizer = load_model(model, optimizer, args, fold_number)

    model.eval()
    final_test_loader = DataLoader(CustomDataset(valid_bags, valid_labels, train=False), batch_size=2, shuffle=False, num_workers=0)
    aucs, f1, test_accuracy, fpr, tpr, probs, labels_list, f_loss = final_validation(model, final_test_loader, device)
    precision, recall, _ = precision_recall_curve(labels_list, probs)

    score_storage["valid_auc"][-1] = aucs
    score_storage["valid_f1"][-1] = f1
    
    update(score_storage, test_accuracy, fpr, tpr, aucs, f1, precision, recall)

    print(score_storage["valid_auc"])
    print(score_storage["valid_f1"])

summarize(score_storage, args.model, args.job)
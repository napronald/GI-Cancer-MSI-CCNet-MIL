import os
import pickle
import numpy as np
import torch
from sklearn.metrics import auc


def smooth_svm_loss(outputs, targets, delta=1.0):
    hinge_loss = torch.clamp(1 - outputs * targets, min=0)
    smooth_loss = torch.where(hinge_loss > delta, hinge_loss - 0.5 * delta, 0.5 * (hinge_loss ** 2) / delta)
    return smooth_loss.mean()


def initialize_fold(score_storage):
    score_storage["valid_auc"].append(0)  
    score_storage["valid_f1"].append(0)


def initialize():
    return {
        "accuracy_scores": [],
        "f1_scores": [],
        "tpr_scores": [],
        "auc_scores": [],
        "precision_scores": [],
        "recall_scores": [],
        "valid_f1": [],
        "valid_auc": [],
        "mean_fpr": np.linspace(0, 1, 50),
        "mean_recall": np.linspace(0, 1, 50)
    }


def update(score_storage, test_accuracy, fpr, tpr, auc_score, f1, precision, recall):
    score_storage["accuracy_scores"].append(test_accuracy)
    score_storage["f1_scores"].append(f1)
    score_storage["tpr_scores"].append(np.interp(score_storage["mean_fpr"], fpr, tpr))
    score_storage["auc_scores"].append(auc_score)
    score_storage["precision_scores"].append(np.interp(score_storage["mean_recall"], precision, recall))


def summarize(score_storage, model_name, job_number):
    mean_tpr = np.mean(score_storage["tpr_scores"], axis=0)
    mean_tpr[-1] = 1
    mean_auc = auc(score_storage["mean_fpr"], mean_tpr)
    std_auc = np.std(score_storage["auc_scores"])

    std_tpr = np.std(score_storage["tpr_scores"], axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

    average_accuracy = np.mean(score_storage["accuracy_scores"])
    std_f1 = np.std(score_storage["f1_scores"])
    average_f1 = np.mean(score_storage["f1_scores"])

    print(f'Average Testing Accuracy: {average_accuracy:.2f}%')

    valid_std_f1 = np.std(score_storage["valid_f1"])
    valid_meanf1 = np.mean(score_storage["valid_f1"])
    valid_meanauc = np.mean(score_storage["valid_auc"])
    valid_std_auc = np.std(score_storage["valid_auc"])

    print(f'Mean F1 Score: {valid_meanf1:.2f} ± {valid_std_f1:.2f}')
    print(f"Mean AUROC {valid_meanauc:.2f} ± {valid_std_auc:.2f}")

    roc_data = {
        'mean_fpr': score_storage["mean_fpr"],
        'mean_tpr': mean_tpr,
        'std_tpr': std_tpr,
        'tprs_upper': tprs_upper,
        'tprs_lower': tprs_lower,
        'accuracy_scores': score_storage["accuracy_scores"],
        'f1_scores': score_storage["f1_scores"],
        'precision_scores': score_storage["precision_scores"],
        'mean_auc': mean_auc,
        'std_auc': std_auc,
        'std_f1': std_f1,
        'average_f1': average_f1
    }

    path = f'{model_name}_roc_data{job_number}.pkl'
    with open(path, 'wb') as file:
        pickle.dump(roc_data, file)

        
def best(vloss, vauc, vf1, model, optimizer, args, epoch, fold_number, best_metrics):
    if vloss < best_metrics['loss']:
        best_metrics['loss'] = vloss
        best_metrics['epoch'] = epoch
        best_metrics['vauc'] = vauc
        best_metrics['vf1'] = vf1
        best_metrics['counter'] = 0  
        print(f'Epoch {epoch}: New best validation loss: {vloss:.4f}')
        save(model, optimizer, args, epoch, fold_number)
    else:
        best_metrics['counter'] += 1


def early_stop(counter, patience, epoch, vauc, vf1, vloss):
    if counter >= patience:
        print(f"Early stopping triggered at epoch {epoch} with validation AUC {vauc}, validation F1 {vf1}, and validation loss {vloss}")
        return True
    return False


def save(model, optimizer, args, epoch, fold_number):
    model_directory = os.path.join(os.path.dirname(__file__), 'model')
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)

    filename = f'{args.model}_job{args.job}_fold{fold_number}_best.tar'
    file_path = os.path.join(model_directory, filename)
    state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'fold': fold_number, 'epoch': epoch}
    torch.save(state, file_path)
    print(f"Model saved to {file_path}.")
    return file_path


def load_model(model, optimizer, args, fold_number):
    model_directory = os.path.join(os.path.dirname(__file__), 'model')
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)

    filename = f'{args.model}_job{args.job}_fold{fold_number}_best.tar'
    path = os.path.join(model_directory, filename)
    
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['net'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer
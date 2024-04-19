import os
import torch
import argparse
import numpy as np

from modules import resnet, network, contrastive_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--num_workers', type=int, default=28, help='num workers')
parser.add_argument('--resnet', type=str, default='ResNet18', help='resnet')
parser.add_argument('--epoch', type=int, default=100, help='epoch')
parser.add_argument('--num_class', type=int, default=2, help='num_class')
parser.add_argument('--feature_dim', type=int, default=128, help='feature_dim')
parser.add_argument('--model_path', type=str, default='model/path', help='model_path') 
parser.add_argument('--seed', type=str, default=9, help='seed') 
parser.add_argument('--dataset-mode', type=str, default='train', choices=['train', 'test'], help='Dataset mode: train or test')
args = parser.parse_args()

def extraction(loader, model, device, path):
    model.eval()
    feature_vectors, labels_vector, wsi_vector = [], [], []
    for (x, y, w) in loader:
        x = x.to(device)
        with torch.no_grad():
            ci = model.extract(x)
            feature_vectors.extend(ci.cpu().numpy())
            labels_vector.extend(y.numpy())
            wsi_vector.extend(w)

    np.save(f"{path}_feature.npy", np.array(feature_vectors))
    np.save(f"{path}_label.npy", np.array(labels_vector))
    np.save(f"{path}_wsi.npy", np.array(wsi_vector))
    return np.array(feature_vectors), np.array(labels_vector), np.array(wsi_vector)


torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.manual_seed(args.seed) if device.type == 'cuda' else None

dataset_file = "train_dataset.csv" if args.dataset_mode == 'train' else "test_dataset.csv"
data_loader = torch.utils.data.DataLoader(
    contrastive_dataset.CustomImageDataset(dataset_file),
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
)

model = network.Network(resnet.get_resnet(args.resnet), args.feature_dim, args.num_class)
model_fp = os.path.join(args.model_path, f"checkpoint_{args.epoch}.tar")
checkpoint = torch.load(model_fp, map_location=device)
model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['net'].items()})
model.to(device)  

X, Y, WSI = extraction(data_loader, model, device, args.dataset_mode)
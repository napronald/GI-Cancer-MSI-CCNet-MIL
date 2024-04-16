import numpy as np
import random
from torch.utils.data import Dataset
from collections import Counter
from sklearn.utils import shuffle


class Bags(Dataset):
    def __init__(self, train=True, bag_size=25, seed=9):
        self.train = train
        self.bag_size = bag_size
        self.seed = seed
        self.bags_list, self.labels_list = self._create_bags()
        self.labels_list = np.array(self.labels_list, dtype=np.int64)

    def _load_and_normalize_data(self, data_type):
        feature_np = np.load(f"{data_type}_feature.npy")
        feature_np /= np.linalg.norm(feature_np, axis=1, keepdims=True)
        label_np = np.load(f"{data_type}_label.npy")
        wsi_np = np.load(f"{data_type}_wsi.npy")
        return feature_np, label_np, wsi_np

    def _create_bags(self):
        data_type = "train" if self.train else "test"
        feature_np, label_np, wsi_np = self._load_and_normalize_data(data_type)
        return self._make_bags(feature_np, label_np, wsi_np)

    def _make_bags(self, feature_np, label_np, wsi_np):
        wsi_to_indices = {wsi: np.where(wsi_np == wsi)[0] for wsi in np.unique(wsi_np)}

        random.seed(self.seed)
        shuffled_wsi_to_indices = {wsi: random.sample(list(indices), len(indices)) for wsi, indices in wsi_to_indices.items()}
        bags, labels = [], []

        for wsi, indices in shuffled_wsi_to_indices.items():
            for i in range(0, len(indices), self.bag_size):
                current_indices = indices[i:i + self.bag_size]
                if len(current_indices) < self.bag_size:
                    continue

                bags.append(feature_np[current_indices])
                labels.append(label_np[current_indices[0]])

        return np.array(bags), np.array(labels, dtype=np.int64)

    def print_label_counts(self):
        label_counts = Counter(self.labels_list)
        dataset_type = "Train" if self.train else "Test"
        print(f"{dataset_type} Label Counts:", label_counts)

    def __len__(self):
        return len(self.labels_list)

    def __getitem__(self, index):
        return self.bags_list[index], [self.labels_list[index]]
    

class CustomDataset(Dataset):
    def __init__(self, bags_list, labels_list, data_size=1, seed=9, train=False):
        if train:
            np.random.seed(seed)
            num_samples = int(len(bags_list) * data_size)
            class_labels = np.unique(labels_list)
            num_samples_per_class = num_samples // len(class_labels)

            combined_bags, combined_labels = [], []
            for class_label in class_labels:
                class_indices = np.where(labels_list == class_label)[0]
                sampled_indices = class_indices if len(class_indices) < num_samples_per_class else np.random.choice(class_indices, num_samples_per_class, replace=False)
                combined_bags.append(bags_list[sampled_indices])
                combined_labels.append(labels_list[sampled_indices])

            bags_list = np.concatenate(combined_bags)
            labels_list = np.concatenate(combined_labels)
            bags_list, labels_list = shuffle(bags_list, labels_list, random_state=seed)

        self.bags_list = bags_list
        self.labels_list = labels_list

    def get_class_counts(self):
        return Counter(self.labels_list)

    def __len__(self):
        return len(self.bags_list)

    def __getitem__(self, index):
        return self.bags_list[index], self.labels_list[index]
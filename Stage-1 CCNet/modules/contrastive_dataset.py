import torchvision
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


class ContrastiveDataset(Dataset):
    def __init__(self, annotations_file, size=224, s=1.0):
        self.img_labels = pd.read_csv(annotations_file)

        self.train_transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply(
                [transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)],
                p=0.8
            ),
            torchvision.transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_labels.iloc[idx, 0]
        label = self.img_labels.iloc[idx, 1]

        image = Image.open(img_path)

        view1 = self.train_transforms(image)
        view2 = self.train_transforms(image)
        
        return view1, view2, label
    

class CustomImageDataset:
    def __init__(self, annotations_file):
        self.img_labels = pd.read_csv(annotations_file)
        self.transform_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_labels.iloc[idx, 0]
        label = self.img_labels.iloc[idx, 1]
        wsi = self.img_labels.iloc[idx, 2]

        image = Image.open(img_path)

        image = self.transform_tensor(image)
        
        return image, label, wsi
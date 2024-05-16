import os
from torch.utils.data import Dataset
from PIL import Image

class FlSeaDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = self._load_data()

    def _load_data(self):
        data = []
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith('.tif'):
                    if 'depth' in root:
                        depth_path = os.path.join(root, file)
                        image_path = os.path.join(root.replace('depth', 'imgs'), file.replace('_SeaErra_abs_depth.tif', '.tiff'))
                        data.append((image_path, depth_path))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, depth_path = self.data[idx]
        image = Image.open(image_path)
        depth = Image.open(depth_path)
        if self.transform:
            image = self.transform(image)
            depth = self.transform(depth)
        return image, depth

class EiffelTowerDataset(Dataset):
    def __init__(self, data_dir, label_dir, transforms=None):
        self.data_dir = data_dir
        self.transforms = transforms
        self.image_files = []
        self.mask_files = []
        
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.jpg') or file.endswith('.png'):
                    image_file = os.path.join(root, file)
                    self.image_files.append(image_file)
                    
        for root, _, files in os.walk(label_dir):
            for file in files:
                if file.endswith('.jpg') or file.endswith('.png'):
                    mask_file = os.path.join(root, file)
                    self.mask_files.append(mask_file)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_file = self.image_files[index]
        mask_file = self.mask_files[index]
        
        image = Image.open(image_file).convert('RGB')
        mask = Image.open(mask_file).convert('L')   # Convert target image to grayscale
        
        if self.transforms:
            image = self.transforms(image)
            mask = self.transforms(mask)
        
        return image, mask
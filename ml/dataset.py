from pathlib import Path
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
CLASS_LABELS = dict(zip(CLASS_NAMES, ['Actinic keratoses and intraepithelial carcinoma', 'Basal cell carcinoma', 'Benign keratosis-like lesions', 'Dermatofibroma', 'Melanoma', 'Melanocytic nevi', 'Vascular lesions']))
IMAGE_SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

def build_transforms(training=False):
    ops = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=InterpolationMode.BILINEAR)]
    if training:
        ops += [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), transforms.RandomRotation(20), transforms.ColorJitter(brightness=.1, contrast=.1, saturation=.1, hue=.02)]
    return transforms.Compose(ops + [transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

def load_metadata(root):
    root = Path(root)
    df = pd.read_csv(root / 'HAM10000_metadata.csv')
    required = ['image_id', 'lesion_id', 'dx']
    if not set(required).issubset(df.columns) or df[required].isna().any().any():
        raise ValueError('Metadata requires non-null image_id, lesion_id and dx')
    if df.image_id.duplicated().any(): raise ValueError('Duplicate image_id')
    if set(df.dx) - set(CLASS_NAMES): raise ValueError('Unknown diagnosis code')
    paths = {}
    for name in ['HAM10000_images_part_1', 'HAM10000_images_part_2']:
        directory = root / name
        if not directory.is_dir(): raise FileNotFoundError(directory)
        for p in directory.iterdir():
            if p.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                if p.stem in paths: raise ValueError(f'Duplicate image {p.stem}')
                paths[p.stem] = p
    df['image_path'] = df.image_id.map(paths)
    if df.image_path.isna().any(): raise FileNotFoundError(f"Missing images: {df.loc[df.image_path.isna(), 'image_id'].head().tolist()}")
    df['label'] = df.dx.map({x: i for i,x in enumerate(CLASS_NAMES)}).astype(int)
    return df.reset_index(drop=True)

class HAM10000Dataset(Dataset):
    def __init__(self, frame, training=False):
        self.frame = frame.reset_index(drop=True)
        self.transform = build_transforms(training)
    def __len__(self): return len(self.frame)
    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(row.image_path) as image:
            x = self.transform(image.convert('RGB'))
        return x, int(row.label)

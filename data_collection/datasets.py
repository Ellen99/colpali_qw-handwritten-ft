import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class IAMDataset(Dataset):
    def __init__(self, lines_txt_path, base_image_path, transform=None):
        self.image_paths = []
        self.transcriptions = []
        self.transform = transform

        with open(lines_txt_path, 'r') as file:
            for line in file:
                if line.startswith('#'):
                    continue
                parts = line.strip().split(' ')
                if len(parts) < 9:
                    continue
                img_id = parts[0]
                transcription = ' '.join(parts[8:]).replace('|', ' ')

                img_id_parts = img_id.strip().split('-')
                subdir = img_id_parts[0]
                lvl2_dir = subdir + '-' +  img_id_parts[1]
                img_path = os.path.join(base_image_path, subdir, lvl2_dir, f"{img_id}.png")
                if os.path.exists(img_path):
                    self.image_paths.append(img_path)
                    self.transcriptions.append(transcription)
                else:
                    print(f"Image not found: {img_path}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('L')  # Convert to grayscale
        if self.transform:
            img = self.transform(img)
        text = self.transcriptions[idx]
        return img, text


class RIMESDataset(Dataset):
    def __init__(self, lines_txt_path, base_image_path, transform=None):
        self.image_paths = []
        self.transcriptions = []
        self.transform = transform

        with open(lines_txt_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(',', 1)
                if len(parts) != 2:
                    continue
                image_filename, transcription = parts
                img_path = os.path.join(base_image_path, image_filename)
                if os.path.exists(img_path):
                    self.image_paths.append(img_path)
                    self.transcriptions.append(transcription)
                else:
                    print(f"Image not found: {img_path}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('L')
        if self.transform:
            img = self.transform(img)
        text = self.transcriptions[idx]
        return img, text

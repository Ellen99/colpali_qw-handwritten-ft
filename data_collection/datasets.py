import os
import json
import csv
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


def load_queries(queries_jsonl):
    """Return list[dict] from lam_queries.jsonl."""
    with open(queries_jsonl, encoding="utf-8") as f:
        return [json.loads(line) for line in f]

class LAMQueryDataset(Dataset):
    """
    PyTorch-style dataset that iterates over query records and,
    optionally, loads the corresponding page image.

    Args
    ----
    queries_jsonl : str
        Path to lam_queries.jsonl
    load_images   : bool
        If True, __getitem__ returns a PIL.Image as first element.
        If False, it returns None instead (faster when encoding text only).
    transform     : callable or None
        Optional torchvision transform applied to each image.
    """

    def __init__(self, queries_jsonl, load_images=True, transform=None):

        self.records = load_queries(queries_jsonl)
        self.load_images = load_images
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]

        img = None
        if self.load_images:
            img = Image.open(rec["image_path"]).convert("RGB")
            if self.transform:
                img = self.transform(img)

        return {
            "image": img,                 # PIL.Image or None
            "query_id": rec["id"],
            "query": rec["query"],
            "answer": rec["answer"],
            "doc_id": rec["doc_id"],
            "image_path": rec["image_path"],
        }

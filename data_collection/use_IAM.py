from torchvision import transforms
from data_collection.scripts.datasets import IAMDataset

# Define any transformations needed

transform = transforms.Compose([
    transforms.RandomRotation(degrees=5),
    transforms.RandomAffine(degrees=0, translate=(0.02, 0.02), scale=(0.95, 1.05), shear=5),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


dataset = IAMDataset(lines_txt_path='data/ascii/lines.txt',
                     base_image_path='data/lines',
                     transform=transform)

# Access data samples
for img, text in dataset:
    # process the images and texts here
    pass
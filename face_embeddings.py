# save_embeddings_mobile.py

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from train import MobileFaceNet  # ✅ Use MobileFaceNet model
from tqdm import tqdm
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load trained MobileFaceNet model
model = MobileFaceNet(embedding_size=512)
model.load_state_dict(torch.load('mobilefacenet_face_recognition2.pth'))
model = model.to(device)
model.eval()

# Dataset config
dataset_root = r'C:\Users\Lenovo\Desktop\project1\mobile_facenet_base_models\Dataset\train'
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

dataset = datasets.ImageFolder(root=dataset_root, transform=transform)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

# Generate embeddings
embeddings_db = {}

for img, label in tqdm(loader, desc="Creating embeddings"):
    img = img.to(device)
    with torch.no_grad():
        embedding = model(img)
        embedding = F.normalize(embedding, p=2, dim=1)

    person_name = dataset.classes[label.item()]
    if person_name not in embeddings_db:
        embeddings_db[person_name] = []

    embeddings_db[person_name].append(embedding.cpu())

# Save the embeddings dictionary
torch.save(embeddings_db, 'mobilefacenet_embeddings2.pth')
print("✅ Embeddings saved to mobilefacenet_embeddings2.pth")

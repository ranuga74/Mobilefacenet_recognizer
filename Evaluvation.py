import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from train import MobileFaceNet
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load trained model
model = MobileFaceNet(embedding_size=512)
model.load_state_dict(torch.load(
    'mobilefacenet_face_recognition2.pth', map_location=device))
model.to(device)
model.eval()

# Load saved embeddings
embeddings_db = torch.load('mobilefacenet_embeddings2.pth')
for k in embeddings_db:
    embeddings_db[k] = [F.normalize(e, dim=1) for e in embeddings_db[k]]

# Setup transforms and dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])
dataset = datasets.ImageFolder(
    r'C:\Users\Lenovo\Desktop\project1\mobile_facenet_base_models\Dataset\eval', transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Output image folder
os.makedirs("results_images", exist_ok=True)

true_labels = []
predicted_labels = []
label_names = dataset.classes

font = ImageFont.load_default()

# Inference and visualization
for idx, (img_tensor, label_tensor) in enumerate(tqdm(dataloader, desc="Evaluating")):
    img_tensor = img_tensor.to(device)
    label = label_tensor.item()
    true_name = label_names[label]

    with torch.no_grad():
        embedding = F.normalize(model(img_tensor), p=2, dim=1)

    # Compare with all embeddings in DB
    best_sim = -1
    predicted_name = "Unknown"

    for person, embeddings in embeddings_db.items():
        for e in embeddings:
            sim = F.cosine_similarity(embedding.cpu(), e).item()
            if sim > best_sim:
                best_sim = sim
                predicted_name = person

    true_labels.append(true_name)
    predicted_labels.append(predicted_name)

    # Save result image
    img = transforms.ToPILImage()(img_tensor[0].cpu()).convert("RGB")
    draw = ImageDraw.Draw(img)
    draw.text((5, 5), f"True: {true_name}", fill="green", font=font)
    draw.text((5, 25), f"Pred: {predicted_name}", fill="blue", font=font)
    img.save(
        f"results_images/img_{idx+1}_{true_name}_pred_{predicted_name}.jpg")

# Accuracy
correct = sum(t == p for t, p in zip(true_labels, predicted_labels))
accuracy = 100 * correct / len(true_labels)
print(f"\n🎯 Final Accuracy: {accuracy:.2f}%")

# Confusion matrix with improved clarity
cm_labels = sorted(set(true_labels + predicted_labels))
cm = confusion_matrix(true_labels, predicted_labels, labels=cm_labels)

# Save confusion matrix image with enhanced readability
plt.figure(figsize=(12, 10))  # Increase figure size for better readability
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cm_labels)
disp.plot(cmap='Blues', values_format='.0f', xticks_rotation=45, colorbar=True)
plt.title(f'Confusion Matrix (Accuracy: {accuracy:.2f}%)', pad=20)
plt.xlabel('Predicted Label', labelpad=15)
plt.ylabel('True Label', labelpad=15)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)  # Increase DPI for clarity
plt.close()

# Save accuracy as image
plt.figure(figsize=(4, 2))
plt.text(0.5, 0.5, f"Accuracy: {accuracy:.2f}%",
         fontsize=16, ha='center', va='center')
plt.axis('off')
plt.savefig("accuracy.png", dpi=300)
plt.close()

# Save combined summary image
fig, ax = plt.subplots(2, 1, figsize=(8, 12))
ax[0].text(0.5, 0.5, f"Accuracy: {accuracy:.2f}%",
           fontsize=18, ha='center', va='center')
ax[0].axis('off')
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cm_labels).plot(
    ax=ax[1], cmap='Blues', values_format='.0f', xticks_rotation=45, colorbar=True)
plt.tight_layout()
plt.savefig("evaluation_summary.png", dpi=300)
plt.close()

print("✅ Evaluation complete. Results saved as:")
print("- accuracy.png")
print("- confusion_matrix.png")
print("- evaluation_summary.png")
print("- results_images/ (per-image results)")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os

# MobileFaceNet Architecture


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=1, stride=1, padding=0, groups=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.prelu = nn.PReLU(out_channels)

    def forward(self, x):
        return self.prelu(self.bn(self.conv(x)))


class DepthWiseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(DepthWiseBlock, self).__init__()
        self.use_shortcut = (stride == 1 and in_channels == out_channels)

        self.conv = nn.Sequential(
            ConvBlock(in_channels, in_channels, kernel=3,
                      stride=stride, padding=1, groups=in_channels),
            ConvBlock(in_channels, out_channels, kernel=1, stride=1)
        )

        if not self.use_shortcut:
            self.shortcut = ConvBlock(
                in_channels, out_channels, kernel=1, stride=stride)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class MobileFaceNet(nn.Module):
    def __init__(self, embedding_size=512):
        super(MobileFaceNet, self).__init__()

        self.backbone = nn.Sequential(
            ConvBlock(3, 64, kernel=3, stride=2, padding=1),
            ConvBlock(64, 64, kernel=3, stride=1, padding=1, groups=64),

            DepthWiseBlock(64, 64, stride=2),
            DepthWiseBlock(64, 64, stride=1),
            DepthWiseBlock(64, 64, stride=1),
            DepthWiseBlock(64, 64, stride=1),

            DepthWiseBlock(64, 128, stride=2),
            DepthWiseBlock(128, 128, stride=1),
            DepthWiseBlock(128, 128, stride=1),
            DepthWiseBlock(128, 128, stride=1),
            DepthWiseBlock(128, 128, stride=1),
            DepthWiseBlock(128, 128, stride=1),

            ConvBlock(128, 512, kernel=1, stride=1)
        )

        self.output_layer = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Dropout(0.4),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, embedding_size),
            nn.BatchNorm1d(embedding_size)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.output_layer(x)
        return x


# ArcFace Loss
class ArcFaceLoss(nn.Module):
    def __init__(self, num_classes, embedding_size=512, margin=0.5, scale=64):
        super(ArcFaceLoss, self).__init__()
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.margin = margin
        self.scale = scale

        self.weight = nn.Parameter(torch.Tensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels):
        embeddings = F.normalize(embeddings, dim=1)
        weights = F.normalize(self.weight, dim=1)

        cosine = torch.matmul(embeddings, weights.t())
        theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))
        target_logits = torch.cos(theta + self.margin)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        output = self.scale * (one_hot * target_logits +
                               (1.0 - one_hot) * cosine)
        return F.cross_entropy(output, labels)


# Main training function
def main():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    dataset = datasets.ImageFolder(
        root=r'C:\Users\Lenovo\Desktop\project1\mobile_facenet_base_models\Dataset\train', transform=transform)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=16, shuffle=True, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MobileFaceNet(embedding_size=512).to(device)
    criterion = ArcFaceLoss(num_classes=len(
        dataset.classes), embedding_size=512).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 45

    loss_history = []
    accuracy_history = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            embeddings = model(images)
            loss = criterion(embeddings, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            with torch.no_grad():
                embeddings_norm = F.normalize(embeddings, dim=1)
                weights_norm = F.normalize(criterion.weight, dim=1)
                logits = torch.matmul(
                    embeddings_norm, weights_norm.t()) * criterion.scale
                _, predicted = torch.max(logits, dim=1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        avg_loss = total_loss / len(dataloader)
        accuracy = 100 * correct / total
        loss_history.append(avg_loss)
        accuracy_history.append(accuracy)

        print(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    # Plot training loss and accuracy
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss_history, 'r', label='Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy_history, 'b', label='Accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_plot2.png')
    print("Training plot saved as training_plot2.png")

    torch.save(model.state_dict(), 'mobilefacenet_face_recognition2.pth')
    print("Model saved as mobilefacenet_face_recognition2.pth")


# Safe entry point
if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.freeze_support()
    main()

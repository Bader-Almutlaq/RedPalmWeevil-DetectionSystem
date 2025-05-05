import torch
import os
import torchvision.models as models
from torchvision import datasets, transforms
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# === Settings ===
data_dir = "data"  # Path to your data folder
model_path = "saved_full_models/mobilenetv3_rpw.pth"  # Path to your MobileNetV3 model
batch_size = 32

# Check if MPS is available (Metal Performance Shaders on Mac)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# === MobileNetV3-Specific Input Size and Normalization ===
input_size = 224  # MobileNetV3 expects 224x224 input size
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# Define transform to apply to the data
transform = transforms.Compose(
    [
        transforms.Resize((input_size, input_size)),  # Resize images to 224x224
        transforms.ToTensor(),  # Convert to tensor
        normalize,  # Normalize with pre-trained ImageNet stats
    ]
)

# Load the MobileNetV3 model
model = torch.load(model_path, map_location=device, weights_only=False)
model.eval()

# === Dataset & DataLoader ===
dataset = datasets.ImageFolder(root=data_dir, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)


# === AUC Evaluation Function ===
def evaluate_auc(model, dataloader):
    model.eval()
    y_probs, y_true = [], []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Get model outputs (logits or probabilities)
            outputs = model(inputs)

            # If output has 2 classes, get probabilities for class 1 (positive class)
            if outputs.shape[1] == 2:
                probs = torch.softmax(outputs, dim=1)[
                    :, 1
                ]  # Get probabilities for class 1
            else:
                probs = torch.sigmoid(
                    outputs
                )  # In case of single-class output (sigmoid)

            y_probs.extend(probs.cpu().numpy().tolist())
            y_true.extend(labels.cpu().numpy().tolist())

    auc = roc_auc_score(y_true, y_probs)
    return auc, y_true, y_probs


# === Evaluate AUC ===
auc, y_true, y_probs = evaluate_auc(model, dataloader)
print(f"MobileNetV3 AUC: {auc:.4f}")

# === Plot ROC Curve ===
fpr, tpr, _ = roc_curve(y_true, y_probs)
plt.plot(fpr, tpr, label=f"MobileNetV3 (AUC = {auc:.2f})")
plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for MobileNetV3")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

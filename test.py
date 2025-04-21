import torch
from torchvision import transforms

MODEL_CONFIGS = {
    "mobilenet": {
        "classifier_index": 3,
        "input_size": 224,
    },
    "efficientnet_b0": {
        "classifier_index": 1,
        "input_size": 224,
    },
    "efficientnet_b4": {
        "classifier_index": 1,
        "input_size": 380,
    },
}

transform = transforms.Compose(
    [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
model = torch.load(
    "saved_models/mobilenetv3.2_rpw.pth", map_location="cpu", weights_only=False
)
print(type(model))

import torch
model = torch.load("saved_models/mobilenetv3.2_rpw.pth" ,map_location="cpu", weights_only=False)
print(type(model))

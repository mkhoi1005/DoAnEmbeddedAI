import torch

model = torch.load("best.pt", map_location="cpu", weights_only=False)
if hasattr(model, 'nc'):
    print("Number of classes:", model.nc)
if hasattr(model, 'names'):
    print("Class names:", model.names)
else:
    if isinstance(model, dict):
        if 'model' in model:
            model = model['model']
        if hasattr(model, 'nc'):
            print("Number of classes:", model.nc)
        if hasattr(model, 'names'):
            print("Class names:", model.names)
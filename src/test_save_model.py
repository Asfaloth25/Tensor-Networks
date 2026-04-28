import torch
import os

from src.model import BinaryTTN

if __name__ == "__main__":
    FILE_PATH = './saved_models/test_save_model.pt'
    model = BinaryTTN((8,4), pixel_embedding_dim=2, bond_dim=8)

    model_clone = BinaryTTN((8, 4), 2, 8)# .to(device)
    for l1, l2 in zip(model._layers, model_clone._layers):
        l2.weights.data.copy_(l1.weights)

    theta = torch.rand((8, 4))

    img = torch.stack([
        torch.cos(theta * torch.pi/2),
        torch.sin(theta * torch.pi/2)
    ]).unsqueeze(dim=0)

    model.rightmost_canonicalize()
    model.save(FILE_PATH)
    model_loaded = BinaryTTN.from_file(FILE_PATH)

    print('Original model:', model(img, normalize_output=False))
    print('Loaded model:', model_loaded(img, normalize_output=False))

    os.remove(FILE_PATH)
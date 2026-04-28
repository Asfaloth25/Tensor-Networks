import torch
from src.model import BinaryTTN



if __name__ == "__main__":
    # seed = 69420
    # torch.random.manual_seed(seed)

    model = BinaryTTN((8,4), pixel_embedding_dim=2, bond_dim=8)

    model_clone = BinaryTTN((8, 4), 2, 8)# .to(device)
    for l1, l2 in zip(model._layers, model_clone._layers):
        l2.weights.data.copy_(l1.weights)

    theta = torch.rand((8, 4))

    img = torch.stack([
        torch.cos(theta * torch.pi/2),
        torch.sin(theta * torch.pi/2)
    ]).unsqueeze(dim=0)

    print('Each pixel has modulus 1:', torch.allclose(img[:, 0, ...]**2 + img[:, 1, ...]**2, torch.ones_like(theta).unsqueeze(dim=0)))

    print('Pre canon:', model(img, normalize_output=False))

    model.canonicalize_network()
    print('Canon 1:', model(img, normalize_output=False))

    print('Clone, pre canon:', model_clone(img, normalize_output=False))

    per_layer_differences = [(l1.weights - l2.weights).abs().mean().item() for l1, l2 in zip(model._layers, model_clone._layers)]

    print('Difference per layer:', per_layer_differences)

    model_clone.rightmost_canonicalize()
    per_layer_differences = [(l1.weights - l2.weights).abs().mean().item() for l1, l2 in zip(model._layers, model_clone._layers)]
    print('Difference per layer:', per_layer_differences)
    print('Clone, post canon:', model_clone(img, normalize_output=False))
    
    for node in model_clone.sweep():
        print(node)

    breakpoint()
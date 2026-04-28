import torch

from src.model import BinaryTTN


if __name__ == "__main__":
    # seed = 69420
    # torch.random.manual_seed(seed)


    model = BinaryTTN((8,4), pixel_embedding_dim=2, bond_dim=8)

    theta = torch.rand((8, 4))

    img = torch.stack([
        torch.cos(theta * torch.pi/2),
        torch.sin(theta * torch.pi/2)
    ]).unsqueeze(dim=0)

    print('Each pixel has modulus 1:', torch.allclose(img[:, 0, ...]**2 + img[:, 1, ...]**2, torch.ones_like(theta).unsqueeze(dim=0)))

    print(model(img, normalize_output=False))

    model.canonicalize_network()
    print(model(img, normalize_output=False))
import torch
from torch import nn
from torchvision.transforms import v2
from tqdm import tqdm

from src.model import BinaryTTN
from src.mnist import PadAndEmbed


device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

@torch.no_grad()
def sample_img(model, n_values:int=8, temperature:float=1):
    img_size, pixel_embedding_dim = model.input_shape, model.pixel_embedding_dim

    final_img = torch.zeros(img_size, device=device)
    img_batch = torch.ones((n_values, pixel_embedding_dim, *img_size), device=device)
    

    mask = torch.tensor([1.0, 0.0], device=device)
    img_batch *= mask.view(1, -1, 1, 1)


    possible_pixel_values = torch.linspace(0, 1, steps=n_values, device=device).unsqueeze(0)

    embedded_possible_values = torch.cat(
        [
            torch.cos(possible_pixel_values * torch.pi/2),
            torch.sin(possible_pixel_values * torch.pi/2)
        ]
    ).T

    for h in range(img_size[0]):
        for w in range(img_size[1]):
            img_batch[:, :, h, w] = embedded_possible_values

            outputs = model(img_batch, return_log_probability=True).squeeze()

            outputs -= outputs.max()
            probs = torch.softmax(outputs / temperature, dim=0)
            sampled_idx = torch.multinomial(probs, 1).item()
            # sampled_idx = torch.argmax(outputs.squeeze())

            final_img[h, w] = possible_pixel_values[0, sampled_idx]
            img_batch[..., h, w] = embedded_possible_values[sampled_idx]    
    
    return final_img


if __name__ == "__main__":
    FILE_PATH = './saved_models/bond_16_new/epoch_30.pt'
    N_VALUES = 8

    model = BinaryTTN.from_file(FILE_PATH).to(device)
    model.train()

    img = sample_img(model, n_values=N_VALUES, temperature=0.1)
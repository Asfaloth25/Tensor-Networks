import torch
from torchvision.transforms import v2


from src.model import BinaryTTN
from src.mnist import get_dataset, PadAndEmbed



if __name__ == "__main__":
    ds_train = get_dataset(
        train=True, 
        transform=v2.Compose(
            [
                v2.ToImage(), 
                v2.ToDtype(torch.float32, scale=True),
                PadAndEmbed()
            ]
        )
    )

    torch.utils.data.DataLoader(
        ds_train,
        batch_size=64, 
        shuffle=True, 
        drop_last=True
    )

    model = BinaryTTN((32, 32), 2, 16)
import torchvision
from torchvision.transforms import v2
import torch

def get_dataset(train:bool=True, transform=None)->torchvision.datasets.MNIST:
    dataset = torchvision.datasets.MNIST(
        f'./data/MNIST/{"train" if train else "test"}',
        train,
        transform,
        download=True
    )

    return dataset

class PadAndEmbed:
    def __call__(self, img:torch.Tensor):
        '''
        Given an image (`Tensor`) of shape `[1, n, n]` pads it to the lowest power of 2 shape that fits it.

        Returns a `Tensor` with of shape `[2, 2^m, 2^m] | m : int`. Now, each pixel (previously a float) is 2-dimensional (`[1, 0] = black, [0, 1] = white`).
        This is essential for TTNs to model the image space correctly.
        '''
        embedded_img = torch.cat([torch.cos(img * torch.pi/2), torch.sin(img * torch.pi/2)])
        embed_dim, orig_h, orig_w = embedded_img.shape

        img_size = 2 ** torch.log2(torch.Tensor([orig_h, orig_w])).ceil().int()

        padded_img = torch.zeros((embed_dim, *img_size))
        padded_img[0, ...] = 1
        padded_img[..., :orig_h, :orig_w] = embedded_img

        return padded_img


if __name__ == "__main__":
    ds = get_dataset(
        train=True, 
        transform=v2.Compose(
            [
                v2.ToImage(), 
                v2.ToDtype(torch.float32, scale=True),
                PadAndEmbed()
            ]
        )
    )[0]


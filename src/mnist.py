import torchvision
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
        orig_h, orig_w = embedded_img.shape[-2:]

        img_size = 2 ** torch.log2(torch.Tensor(list(img.shape[-2:]))).ceil().int()

        padded_img = torch.zeros((*embedded_img.shape[:-2], *img_size))
        padded_img[..., :orig_h, :orig_w] = embedded_img

        return padded_img


if __name__ == "__main__":
    ds = get_dataset(True, torchvision.transforms.ToTensor())
    breakpoint()
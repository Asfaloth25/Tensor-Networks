import torch
from torch import nn
from torchvision.transforms import v2
import math

from src.mnist import get_dataset, PadAndEmbed
from src.qr import qr_factorize_tens

torch.set_default_dtype(torch.float64)
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

class BinaryTTNLayer(torch.nn.Module):
    def __init__(self, bond_dim:int=16, input_shape:tuple[int, int]=(32, 32), in_dim:int=2):
        '''
        Tree Tensor Network layer. It consists of `(input_shape[0]*input_shape[1])//2` tensors, each of shape `[bond_dim, in_dim, in_dim]`.

        Args:
            bond_dim (int): The dimension of the output vectors from this layer.

            input_shape (tuple[int, int]): Number of pixels (tensors) in the previous layer. 
                Since the TTN consists of interweaved horizontal-vertical layers, the orientation
                will be determined from `input_shape` automatically.
                For vertical (`orientation=0`) layers, the output will be of shape `[input_shape[0]//2, input_shape[1], bond_dim]`.
                For horizontal (`orientation=1`) layers, the output will be of shape `[input_shape[0], input_shape[1]//2, bond_dim]`.

            in_dim (int): The dimension of the input vectors to this layer. Inputs should be of size `[input_shape[0], input_shape[1], in_dim]`.
        '''

        super().__init__()

        self.orientation = input_shape[0] < input_shape[1] # 0 vertical, 1 horizontal

        self.grid_shape = [input_shape[0] // (1 + (1-self.orientation)), input_shape[1] // (1 + self.orientation)]

        self.weights = torch.nn.Parameter(
            torch.rand((*self.grid_shape, bond_dim, in_dim, in_dim))
        )

        self.init_isometric_()

    @torch.no_grad()
    def init_isometric_(self):
        weights = torch.rand_like(self.weights)
        h, w, bond_dim, in_dim, _ = weights.shape

        weights_reshaped = weights.permute((0, 1, 3, 4, 2)).reshape((h, w, in_dim**2, bond_dim))
        Q, R = qr_factorize_tens(weights_reshaped)

        self.weights.copy_(
            Q.reshape((h, w, in_dim, in_dim, bond_dim)).permute((0, 1, 4, 2, 3))
        )


    def forward(self, x:torch.Tensor): 
        batch_size, _, _, in_dim = x.shape

        h, w = self.grid_shape
        if self.orientation:
            x_reshaped = x.reshape((batch_size, h, w, 2, in_dim))
            left, right = x_reshaped[:, :, :, 0, :], x_reshaped[:, :, :, 1, :]
        else:
            x_reshaped = x.reshape((batch_size, h, 2, w, in_dim))
            left, right = x_reshaped[:, :, 0, :, :], x_reshaped[:, :, 1, :, :]

        output = torch.einsum('x y b i j, n x y i, n x y j -> n x y b', self.weights, left, right)
        # - n: batch
        # - x: x index
        # - y: y index
        # - b: bond dimension
        # - i: in_dim
        # - j: in_dim

        print(f'[Layer - {("V", "H")[self.orientation]}: {self.grid_shape}] min: {output.min():.3f}, max: {output.max():.3f}')
        return output


class BinaryTTN(torch.nn.Module):
    def __init__(self, input_shape:tuple[int, int]=(32, 32), pixel_embedding_dim:int=2, bond_dim:int=16):
        '''
        Tree Tensor Network module.

        Args:
            input_shape (tuple[int, int]): Image height and width, in pixels and after padding. 
                For example, the MNIST dataset has `28x28` images, which are then padded to the
                nearest power of 2 (`32x32`). This class supports rectangular images, as long as
                their height and width are both powers of 2.

            pixel_embedding_dim (int): Dimension of each pixel embedding. 2 in the case of MNIST.

            bond_dim (int): Bond dimension of the TTN.
        '''
        super().__init__()

        for i, n_pixels in enumerate(input_shape):
            assert (not math.log2(n_pixels) % 1), f'The {("height", "width")[i]} of the input should be a power of 2. [input_shape = {input_shape}]'
        
        self._input_shape = input_shape

        self._bond_dim = bond_dim

        self._pixel_embedding_dim = pixel_embedding_dim

        self._n_layers = round(math.log2(input_shape[0]*input_shape[1]))

        layers = [BinaryTTNLayer(bond_dim, input_shape, in_dim=pixel_embedding_dim)]
        for i in range(self._n_layers -2):
            layers.append(
                BinaryTTNLayer(bond_dim, input_shape=layers[-1].grid_shape, in_dim=bond_dim)
            )
        layers.append(BinaryTTNLayer(bond_dim=1, input_shape=(1,2), in_dim=bond_dim))

        self._layers = nn.Sequential(*layers)

    def forward(self, x:torch.Tensor):   
        x = x.permute(0, 2, 3, 1) # [w, h, pixel_dim]

        return self._layers(x)
    
    @torch.no_grad()
    def canonicalize_network(self, normalize_root:bool=False):
        layer = self._layers[0]
        h, w, bond_dim, in_dim, _ = layer.weights.shape
        weights_reshaped = layer.weights.permute((0, 1, 3, 4, 2)).reshape((h, w, in_dim**2, bond_dim))
        Q, R = qr_factorize_tens(weights_reshaped)

        layer.weights.copy_(
            Q.reshape((h, w, in_dim, in_dim, bond_dim)).permute((0, 1, 4, 2, 3))
        )

        for layer in self._layers[1:-1]:
            h, w, bond_dim, in_dim, _ = layer.weights.shape
            orientation = layer.orientation

            if orientation: # horizontal
                R_reshaped = R.reshape((h, w, 2, in_dim, in_dim))
                left, right = R_reshaped[:, :, 0, :, :], R_reshaped[:, :, 1, :, :]
            else: # vertical
                R_reshaped = R.reshape((h, 2, w, in_dim, in_dim))
                left, right = R_reshaped[:, 0, :, :, :], R_reshaped[:, 1, :, :, :]

            new_weights = torch.einsum('x y i k, x y b k j -> x y b i j', left, layer.weights)
            new_weights = torch.einsum('x y j k, x y b i k -> x y b i j', right, new_weights)
            # - x: x index
            # - y: y index
            # - b: bond dimension
            # - k: contracted index
            # - i: in_dim
            # - j: in_dim

            weights_reshaped = new_weights.permute((0, 1, 3, 4, 2)).reshape((h, w, in_dim**2, bond_dim))
            Q, R = qr_factorize_tens(weights_reshaped)

            layer.weights.copy_(
                Q.reshape((h, w, in_dim, in_dim, bond_dim)).permute((0, 1, 4, 2, 3))
            )
        
        layer = self._layers[-1]
        h, w, bond_dim, in_dim, _ = layer.weights.shape
        orientation = layer.orientation

        if orientation: # horizontal
            R_reshaped = R.reshape((h, w, 2, in_dim, in_dim))
            left, right = R_reshaped[:, :, 0, :, :], R_reshaped[:, :, 1, :, :]
        else: # vertical
            R_reshaped = R.reshape((h, 2, w, in_dim, in_dim))
            left, right = R_reshaped[:, 0, :, :, :], R_reshaped[:, 1, :, :, :]

        new_weights = torch.einsum('x y i k, x y b k j -> x y b i j', left, layer.weights)
        new_weights = torch.einsum('x y j k, x y b i k -> x y b i j', right, new_weights)
        layer.weights.copy_(new_weights)


if __name__ == '__main__':
    
    ds = get_dataset(
        train=True, 
        transform=v2.Compose(
            [
                v2.ToImage(), 
                v2.ToDtype(torch.float32, scale=True),
                PadAndEmbed()
            ]
        )
    )

    TTN = BinaryTTN((32,32), 2, 16)

    from pprint import pprint

    print('Original network:')
    pprint({i: layer.weights.norm().item() for i, layer in enumerate(TTN._layers)})

    img = ds[0][0].unsqueeze(0)

    print(TTN(img))

    TTN.canonicalize_network()
    
    print('\n\nCanonicalized network:')
    pprint({i: layer.weights.norm().item() for i, layer in enumerate(TTN._layers)})
    print(TTN(img))
import torch
from torch import nn
from torchvision.transforms import v2
import math

from typing import Generator

from src.mnist import get_dataset, PadAndEmbed
from src.qr import qr_factorize_tens, directional_node_qr, absorb_r_node

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
                For vertical (`orientation=0`) layers, the output will be of shape `[batch, input_shape[0]//2, input_shape[1], bond_dim]`.
                For horizontal (`orientation=1`) layers, the output will be of shape `[batch, input_shape[0], input_shape[1]//2, bond_dim]`.

            in_dim (int): The dimension of the input vectors to this layer. Inputs should be of size `[batch, input_shape[0], input_shape[1], in_dim]`.
        '''

        super().__init__()

        self.orientation = input_shape[0] < input_shape[1] # 0 vertical, 1 horizontal

        self.grid_shape = [input_shape[0] // (2-self.orientation), input_shape[1] // (1 + self.orientation)]

        self.weights = torch.nn.Parameter(
            torch.rand((*self.grid_shape, bond_dim, in_dim, in_dim))
        )

        self.init_isometric_()

    def __getitem__(self, key):
        return self.weights[key]
    
    def __setitem__(self, key, value):
        self.weights[key] = value

    @torch.no_grad()
    def init_isometric_(self):
        weights = torch.rand_like(self.weights)
        h, w, bond_dim, in_dim, _ = weights.shape

        weights_reshaped = weights.permute((0, 1, 3, 4, 2)).reshape((h, w, in_dim**2, bond_dim))
        Q, R = qr_factorize_tens(weights_reshaped)

        self.weights.copy_(
            Q.reshape((h, w, in_dim, in_dim, bond_dim)).permute((0, 1, 4, 2, 3))
        )


    def forward(self, x:torch.Tensor, prev_log_norm:torch.Tensor, eps:float=1e-9): 
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

        # print(f'[Layer - {("V", "H")[self.orientation]}: {self.grid_shape}] min: {output.min():.3f}, max: {output.max():.3f}')
        norm = output.norm(dim=-1).clamp_min(eps)
        log_norm = torch.log(norm).sum(dim=(1,2))

        return output/norm.unsqueeze(-1), log_norm + prev_log_norm

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
        
        self.input_shape = input_shape

        self._bond_dim = bond_dim

        self.pixel_embedding_dim = pixel_embedding_dim

        self._n_layers = round(math.log2(input_shape[0]*input_shape[1]))

        layers = [BinaryTTNLayer(bond_dim, input_shape, in_dim=pixel_embedding_dim)]
        for i in range(self._n_layers -2):
            layers.append(
                BinaryTTNLayer(bond_dim, input_shape=layers[-1].grid_shape, in_dim=bond_dim)
            )
        layers.append(BinaryTTNLayer(bond_dim=1, input_shape=(1,2), in_dim=bond_dim))

        self._layers = nn.Sequential(*layers)

        self.n_nodes = sum([torch.prod(torch.tensor(i.weights.shape[:2])) for i in self]).item()

        self._updated_sweep = [
            torch.zeros((layer.weights.shape[:2])) for layer in self._layers
        ]
        self._center=None


    @staticmethod
    def from_file(file_path:str):
        state = torch.load(file_path)
        model = BinaryTTN(**state['init_params'])
        model._center = state['center']
        with torch.no_grad():
            for i, layer in enumerate(model._layers):
                layer.weights.copy_(state['layer_weights'][f'_layers.{i}.weights'])
        return model

    def save(self, file_path:str):
        state = {
            'init_params': {
                'input_shape': self.input_shape,
                'bond_dim': self._bond_dim,
                'pixel_embedding_dim': self.pixel_embedding_dim
            },
            'center': self._center,
            'layer_weights': self.state_dict()
        }
        torch.save(state, file_path)

    def __getitem__(self, key:tuple[int, int, int])->torch.Tensor:
        '''
        Returns the node (tensor) at depth (layer) `key[0]` and grid coordinates `key[1:]`
        '''
        
        if isinstance(key, int) or isinstance(key, slice):
            return self._layers[key]
        
        if isinstance(key[0], int):
            assert key[0] <= self._n_layers - 1, f"Cannot retrieve tensor at position {key}, Layers = (0, ..., {self._n_layers-1}) < {key[0]}."
        
        return self[key[0]][*key[1:], ...]
    
    def __setitem__(self, key, value):
        self[key[0]][*key[1:], ...] = value

    def _get_adjacent(self, node_pos:tuple[int, int, int]=(0, 0, 0)):
        '''
        Notation: (depth (max at root), h, w)
        '''
        
        adj = {}
        if node_pos[0] < self._n_layers-1:
            orientation = self[node_pos[0]+1].orientation
            adj['up'] = (node_pos[0]+1, node_pos[1]//(2-orientation), node_pos[2]//(1+orientation))

        if node_pos[0]:
            orientation = self[node_pos[0]].orientation
            adj['left'] =  (node_pos[0]-1, node_pos[1]*(2-orientation), node_pos[2]*(1+orientation))
            adj['right'] = (node_pos[0]-1, node_pos[1]*(2-orientation) + 1 - orientation, node_pos[2]*(1+orientation) + orientation)

        return adj

    def _is_right_child(self, node_pos:tuple[torch.Tensor, torch.Tensor, torch.Tensor])->bool:
        if node_pos[0] >= self._n_layers-1:
            return
        orientation = self[node_pos[0]+1].orientation
        return node_pos[1+orientation]%2

    @torch.no_grad
    def sweep(self)->Generator[tuple[int, int, int], None, None]:
        '''
        Raises AssertionError if the center is not the rightmost tensor in the network (first layer, bottom-right corner).

        Returns only the positions of the tensors that should be updated, sequentially.
        '''
        start_center = (0, *(torch.tensor(self[0, ...].shape[:2])-1).tolist())
        assert self._center and self._center == start_center, f'Network must be rightmost-canonicalized: center must be {start_center}, currently {self._center}.'
        for sweep_left in (1, 0):
            [layer.zero_() for layer in self._updated_sweep]

            adj = self._get_adjacent(self._center)
            candidates = {
                i: adj[i] for i in adj 
                if not self._updated_sweep[adj[i][0]][adj[i][1:]]
            }
            while candidates:
                is_rchild = self._is_right_child(self._center)

                if len(candidates) == 1:
                    yield self._center
                    # SGD update here (trainer)
                    assert not self._updated_sweep[self._center[0]][self._center[1:]], f'Node {self._center} updated twice during sweep.'
                    self._updated_sweep[self._center[0]][self._center[1:]] = 1

                    direction, new_center = next(iter(candidates.items()))

                elif len(candidates) == 2:
                    keys = tuple(candidates)
                    if 'up' in keys:
                        # direction = keys[(1-is_rchild) ^ sweep_left] # (wrong) Literal interpretation of Algorithm 1.: "rightmost" = most to the right on the graph drawing.
                        direction = keys[1] # 'up' will always be idx 0. We want to explore downward first given the option.
                    else:
                        direction = ('left', 'right')[sweep_left]
                    new_center = candidates[direction]

                else:
                    direction = ('left', 'right')[sweep_left]
                    new_center = candidates[direction]

                Q, R = directional_node_qr(self[self._center], direction)

                self[self._center] = Q
                self._center = new_center
                rel_direction = ('left', 'right')[is_rchild] if direction=='up' else 'up'
                self[self._center] = absorb_r_node(self[self._center], R, rel_direction)

                adj = self._get_adjacent(self._center)
                candidates = {
                    i: adj[i] for i in adj 
                    if not self._updated_sweep[adj[i][0]][adj[i][1:]]
                }

            desired_end_center = (start_center, (0,0,0))[sweep_left]
            assert self._center == desired_end_center, f'Ending node of {("right", "left")[sweep_left]} sweep must be {("left", "right")[sweep_left]}most one, {desired_end_center}. Got {self._center} instead.'
            yield self._center
            self._updated_sweep[self._center[0]][self._center[1:]] = 1

            assert all(torch.all(layer) for layer in self._updated_sweep), f'Not all tensors updated within {("right", "left")[sweep_left]} sweep.'



    def forward(self, x:torch.Tensor, return_log_probability:bool=False, normalize_output:bool=True):   
        x = x.permute(0, 2, 3, 1) # [batch, w, h, pixel_dim]

        batch_size = x.shape[0]

        log_norm = torch.zeros(batch_size, device=x.device)

        i = 0
        for layer in self._layers:
            print('layer', i)
            x, log_norm = layer(x, log_norm)
            breakpoint()
            i = i+1

        Z = (self[self._center] ** 2).sum() if normalize_output else 1
        log_norm = log_norm.view(-1, 1, 1, 1)
        if return_log_probability:
            return log_norm * 2 - torch.log(Z)
        
        return (x * log_norm)**2 / Z

    @torch.no_grad()
    def rightmost_canonicalize(self, normalize_root:bool=False):
        self.canonicalize_network(normalize_root)
        node = (self._n_layers-1, 0, 0)
        adj = self._get_adjacent(node)
        while 'right' in adj:
            Q, R = directional_node_qr(self[node], direction='right')
            self[node] = Q
            node = adj['right']
            self[node] = absorb_r_node(self[node], R, direction='up')
            adj = self._get_adjacent(node)
        self._center = node

            
    @torch.no_grad()
    def canonicalize_network(self, normalize_root:bool=False):
        layer = self[0]
        h, w, bond_dim, in_dim, _ = layer.weights.shape
        weights_reshaped = layer.weights.permute((0, 1, 3, 4, 2)).reshape((h, w, in_dim**2, bond_dim))
        Q, R = qr_factorize_tens(weights_reshaped)

        layer.weights.copy_(
            Q.reshape((h, w, in_dim, in_dim, bond_dim)).permute((0, 1, 4, 2, 3))
        )

        for layer in self[1:-1]:
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
        
        layer = self[-1]
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
        if normalize_root:
            new_weights /= new_weights.norm()
        layer.weights.copy_(new_weights)

        self._center = (self._n_layers-1,0,0)

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

    TTN = BinaryTTN((32,32), 2, 4)

    from pprint import pprint

    print('Original network:')
    pprint({i: layer.weights.norm().item() for i, layer in enumerate(TTN._layers)})

    img = ds[0][0].unsqueeze(0)

    print(TTN(img))
    print('Log output:', TTN(img, return_log_probability=True))

    TTN.canonicalize_network()
    
    print('\n\nCanonicalized network:')
    pprint({i: layer.weights.norm().item() for i, layer in enumerate(TTN._layers)})
    print(TTN(img))
    print('Log output:', TTN(img, return_log_probability=True))

    node = (TTN._n_layers-1, 0, 0)
    adj = TTN._get_adjacent(node)
    print(adj)
    while "left" in adj:
        node = adj["right"]
        try:
            tens = TTN[node]
        except:
            breakpoint()
        adj = TTN._get_adjacent(node)
        print(adj)
    print(node)

    TTN.sweep()
    breakpoint()
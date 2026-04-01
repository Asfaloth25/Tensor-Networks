from src.model import *

class LogTTNLayer(BinaryTTNLayer):
    def forward(self, x, prev_log_norm:torch.Tensor, eps:float=1e-9)->tuple[torch.Tensor, torch.Tensor]:

        output = super().forward(x)
        norm = output.norm(dim=-1).clamp_min(eps)
        log_norm = torch.log(norm).sum(dim=(1,2))

        return output/norm.unsqueeze(-1), log_norm + prev_log_norm
    

class LogTTN(BinaryTTN):
    def __init__(self, input_shape:tuple[int, int]=(32, 32), pixel_embedding_dim:int=2, bond_dim:int=16):
        nn.Module.__init__(self)

        for i, n_pixels in enumerate(input_shape):
            assert (not math.log2(n_pixels) % 1), f'The {("height", "width")[i]} of the input should be a power of 2. [input_shape = {input_shape}]'
        
        self._input_shape = input_shape

        self._bond_dim = bond_dim

        self._pixel_embedding_dim = pixel_embedding_dim

        self._n_layers = round(math.log2(input_shape[0]*input_shape[1]))

        layers = [LogTTNLayer(bond_dim, input_shape, in_dim=pixel_embedding_dim)]
        for i in range(self._n_layers -2):
            layers.append(
                LogTTNLayer(bond_dim, input_shape=layers[-1].grid_shape, in_dim=bond_dim)
            )
        layers.append(LogTTNLayer(bond_dim=1, input_shape=(1,2), in_dim=bond_dim))

        self._layers = nn.Sequential(*layers)

    def forward(self, x:torch.Tensor):
        x = x.permute(0,2,3,1)
        batch_size = x.shape[0]

        log_norm = torch.zeros(batch_size, device=x.device)

        for layer in self._layers:
            x, log_norm = layer(x, log_norm)

        return x * log_norm.exp()

    
if __name__ == "__main__":

    model = BinaryTTN((8,4), pixel_embedding_dim=2, bond_dim=8)
    log_model = LogTTN((8,4), pixel_embedding_dim=2, bond_dim=8)

    for l1, l2 in zip(model._layers, log_model._layers):
        l2.weights.data.copy_(l1.weights)

    theta = torch.rand((8, 4))

    img = torch.stack([
        torch.cos(theta * torch.pi/2),
        torch.sin(theta * torch.pi/2)
    ]).unsqueeze(dim=0)

    print('Each pixel has modulus 1:', torch.allclose(img[:, 0, ...]**2 + img[:, 1, ...]**2, torch.ones_like(theta).unsqueeze(dim=0)))

    print('Binary TTN, pre canon:')
    print(model(img))

    model.canonicalize_network()
    print('Binary TTN, post canon:')
    print(model(img))


    print('Log Binary TTN, pre canon:')
    print(log_model(img))

    log_model.canonicalize_network()
    print('Log Binary TTN, post canon:')
    print(log_model(img))
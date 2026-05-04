import torch
from torch import nn
from torchvision.transforms import v2
from tqdm import tqdm

from src.model import BinaryTTN
from src.mnist import get_dataset, PadAndEmbed


device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"


class Loss(nn.Module):
    def __init__(self, epsilon:float=1e-12):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x:torch.Tensor):
        return -x.mean()

def cycle_dataloader(loader: torch.utils.data.DataLoader):
    while 1:
        for batch in loader:
            yield batch

if __name__ == "__main__":

    N_EPOCHS = 101
    LEARNING_RATE = 0.01
    BOND_DIM = 32
    PRINT_EVERY = 100 # steps
    SAVE_EVERY = 5 # epochs

    MODEL_NAME = input('Please name the model:\n> ')

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

    loader_train = torch.utils.data.DataLoader(
        ds_train,
        batch_size=32, 
        shuffle=True, 
        drop_last=True
    )

    get_data = cycle_dataloader(loader_train)

    model = BinaryTTN((32, 32), 2, BOND_DIM).to(device)
    model.rightmost_canonicalize(normalize_root=True)
    loss = Loss(epsilon=1e-12)

    # old_model = BinaryTTN((32, 32), 2, BOND_DIM).to(device)
    # for l1, l2 in zip(model._layers, old_model._layers):
    #     l2.weights.data.copy_(l1.weights)
    # old_model._center = model._center


    from pprint import pprint
    import os

    os.makedirs(f'./saved_models/{MODEL_NAME}/', exist_ok=False)
    model.save(f'./saved_models/{MODEL_NAME}/untrained.pt')
    for epoch in range(N_EPOCHS):
        print('\n', '-='*15+'{', f'Epoch {epoch}', '}'+'=-'*15)

        losses = {}

        for i, center in enumerate(tqdm(model.sweep(), ncols=70, total=2*model.n_nodes)):
            inputs, labels = next(get_data)
            inputs = inputs.to(device)

            outputs = model(inputs, return_log_probability=True)
            l = loss(outputs)
            if not i%PRINT_EVERY:
                losses[i] = l.item()
            l.backward()

            grad = model[center[0]].weights.grad[center[1:]]
            with torch.no_grad():
                model[center] -= grad * LEARNING_RATE

        print('- Losses:')
        pprint(losses, width=70)

        if not(epoch % SAVE_EVERY):
            model.save(f'./saved_models/{MODEL_NAME}/epoch_{epoch}.pt')

    model.save(f'./saved_models/{MODEL_NAME}/epoch_{epoch}.pt')

    iterator = loader_train.__iter__()
    imgs, labels = iterator.__next__()
    imgs = imgs.to(device)

    batch_size = imgs.shape[0]

    theta = torch.rand((batch_size, 32, 32))
    random_imgs = torch.stack([torch.cos(theta * torch.pi/2), torch.sin(theta * torch.pi/2)]).permute(1, 0, 2, 3).to(device)
    
    log_probs_real = model(imgs, return_log_probability=True)
    log_probs_fake = model(random_imgs, return_log_probability=True)

    # log_probs_old = old_model(imgs, return_log_probability=True)

    breakpoint()
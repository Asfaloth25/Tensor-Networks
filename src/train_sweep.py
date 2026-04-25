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


if __name__ == "__main__":

    N_EPOCHS = 20
    LEARNING_RATE = 0.01
    MOMENTUM = 0.0
    PRINT_EVERY = 100

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
        batch_size=64, 
        shuffle=True, 
        drop_last=True
    )

    model = BinaryTTN((32, 32), 2, 4).to(device)
    loss = Loss(epsilon=1e-12)

    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=LEARNING_RATE, 
        momentum=MOMENTUM
    )

    old_model = BinaryTTN((32, 32), 2, 4).to(device)
    for l1, l2 in zip(model._layers, old_model._layers):
        l2.weights.data.copy_(l1.weights)


    from pprint import pprint

    for epoch in range(N_EPOCHS):
        print('\n', '-='*15+'{', f'Epoch {epoch}', '}'+'=-'*15)

        losses = {}
        # import time
        # for i in tqdm(range(1000), ncols=70):
        #     time.sleep(0.0025)

        for i, data in enumerate(tqdm(loader_train, ncols=70)):
            inputs, labels = data
            inputs = inputs.to(device)
            optimizer.zero_grad()

            outputs = model(inputs, return_log_probability=True) * 2
            l = loss(outputs)

            if not i%PRINT_EVERY:
                losses[i] = l.item()

            # print('Breakpoint 1')
            # breakpoint()
            l.backward()
            # print('Breakpoint 2')
            # breakpoint()
            optimizer.step()
            # print('Breakpoint 3')
            # breakpoint()
            # model.canonicalize_network(normalize_root=True)

        print('- Losses:')
        pprint(losses, width=70)

    iterator = loader_train.__iter__()
    imgs, labels = iterator.__next__()
    imgs = imgs.to(device)

    batch_size = imgs.shape[0]

    theta = torch.rand((batch_size, 32, 32))
    random_imgs = torch.stack([torch.cos(theta * torch.pi/2), torch.sin(theta * torch.pi/2)]).permute(1, 0, 2, 3).to(device)
    
    log_probs_real = model(imgs, return_log_probability=True)
    log_probs_fake = model(random_imgs, return_log_probability=True)

    log_probs_old = old_model(imgs, return_log_probability=True)

    breakpoint()
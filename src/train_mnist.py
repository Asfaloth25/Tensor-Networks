import torch
from torch import nn
from torchvision.transforms import v2
from tqdm import tqdm

from src.model import BinaryTTN
from src.mnist import get_dataset, PadAndEmbed


class Loss(nn.Module):
    def __init__(self, epsilon:float=1e-12):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x:torch.Tensor):
        return -x.mean()


if __name__ == "__main__":

    N_EPOCHS = 5
    LEARNING_RATE = 0.1
    MOMENTUM = 0.9
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

    model = BinaryTTN((32, 32), 2, 16)
    loss = Loss(epsilon=1e-12)

    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=LEARNING_RATE, 
        momentum=MOMENTUM
    )

    from pprint import pprint

    old_weights = model._layers[-1].weights.clone()

    for epoch in range(N_EPOCHS):
        print('\n', '-='*15+'{', f'Epoch {epoch}', '}'+'=-'*15)

        losses = {}
        # import time
        # for i in tqdm(range(1000), ncols=70):
        #     time.sleep(0.0025)

        for i, data in enumerate(tqdm(loader_train, ncols=70)):
            inputs, labels = data
            
            optimizer.zero_grad()

            outputs = model(inputs, return_log_probability=True) * 2
            l = loss(outputs)

            if not i%PRINT_EVERY:
                losses[i] = l.item()

            l.backward()
            optimizer.step()


        print('- Losses:')
        pprint(losses, width=70)

    breakpoint()

import torch

from src.model import BinaryTTNLayer


if __name__ == "__main__":

    tensor_h = torch.arange(8).view((2, 4))
    tensor_v = torch.arange(8).view((4,2))

    orientation_h = tensor_h.shape[0] < tensor_h.shape[1]
    orientation_v = tensor_v.shape[0] < tensor_v.shape[1]

    print('horizontal', orientation_h, tensor_h, 'vertical', orientation_v, tensor_v, sep='\n')

    grid_shape_h = [tensor_h.shape[0] // (1 + (1-orientation_h)), tensor_h.shape[1] // (1 + orientation_h)]
    grid_shape_v = [tensor_v.shape[0] // (1 + (1-orientation_v)), tensor_v.shape[1] // (1 + orientation_v)]

    print('grid shapes', grid_shape_h, grid_shape_v)

    for grid_shape, x, orientation in zip((grid_shape_h, grid_shape_v), (tensor_h, tensor_v), ('horizontal', 'vertical')):
        print('orientation:', orientation)
        x_reshaped = x.reshape((*grid_shape, 2))
        left, right = x_reshaped[..., 0], x_reshaped[..., 1]
        print(left, right, sep='\n')

    x_reshaped_2 = x.reshape((grid_shape[0], 2, grid_shape[1]))
    left_2, right_2 = x_reshaped_2[:, 0, :], x_reshaped_2[:, 1, :]
    print('Corrected vertical split:', left_2, right_2, sep='\n')
import torch
import numpy as np
from src.preprocessing.data_loading import torch_with_padding


def masked_log_softmax(vector, mask, dim=1):
    """
    Implementation of the masked log softmax.
    :param vector: The input we want to mask, a tensor
    :param mask: a tensor
    :param dim: an int, giving the dimension to mask
    :return: a tensor
    """
    if mask is not None:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        vector = vector + (mask + 1e-45).log()
    return torch.nn.functional.log_softmax(vector, dim=dim)


def to_torch(x, padding, room_size, y=None):
    """
    Convert a set of rooms in a numpy format to a torch tensor.
    If there's padding to do : call torch_with_padding function.
    :param x: the input dataset (numpy format)
    :param padding: an int that is the amount of padding to add
    :param room_size: the size of the room with the padding included
    :param y: (optional) the label dataset
    :return: a torch tensor that contains the input dataset (with padding),
     and if y is given, a torch tensor that contains the label
    """
    with torch.no_grad():
        if padding != 0:
            if y is not None:
                torch_X, torch_Y = torch_with_padding(x, padding, room_size, y)
            else:
                torch_X = torch_with_padding(x, padding, room_size, y)
        else:
            torch_X = torch.Tensor(x).view(-1, room_size, room_size, 1)
            if y is not None:
                torch_Y = torch.Tensor(y).view(-1, room_size, room_size, 1)

        if y is not None:
            return torch_X, torch_Y
        return torch_X


def to_numpy(x, room_size, sizes, p, y=None):
    """
    Convert a set of rooms in a torch tensor format to numpy.
    If there's padding : remove the padding to have the initial size.
    :param x: the input dataset (tensor format)
    :param room_size: the size of the room with the padding included
    :param sizes: an array with the original size for each room
    :param p: the padding size
    :param y: (optional) the label dataset
    :return: a numpy array that contains the input dataset (with padding),
    and if y is given, a numpy array that contains the label
    """
    with torch.no_grad():

        x = x.detach().cpu().numpy()
        if y is not None:
            y = y.detach().cpu().numpy()

        if p == 0:
            if y is not None:
                return x, y
            return x

        if p != 0:
            x_numpy, y_numpy = [], []
            for idx_room, room in enumerate(x):

                if sizes[idx_room][0] != room_size:
                    if sizes[idx_room][1] != room_size:  # Remove padding on both directions
                        x_numpy.append(room[p:-p, p:-p])
                        if y is not None:
                            y_numpy.append(y[p:-p, p:-p])

                    else:  # Remove padding only on the rows
                        x_numpy.append(room[p:-p])
                        if y is not None:
                            y_numpy.append(y[p:-p])

                else:
                    if sizes[idx_room][1] != room_size:  # Remove padding only on the columns
                        x_numpy.append(room[:, p:-p])
                        if y is not None:
                            y_numpy.append(y[:, p:-p])
                    else:
                        x_numpy.append(room)
                        if y is not None:
                            y_numpy.append(y)

            if y is not None:
                return np.asarray(x_numpy), np.asarray(y_numpy)
            return np.asarray(x_numpy)

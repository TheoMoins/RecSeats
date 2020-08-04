import torch
import torch.nn as nn
import torch.nn.functional as F
from src.deep.deep_utils import masked_log_softmax, to_torch, to_numpy
import numpy as np

PATH_BEST_MODEL = "./save/best_models/best_model_"


class ContentPart(nn.Module):
    """
    Model for the Content-based part.
    Loads the already-trained model before each forward.
    """

    def __init__(self, room_size, padding, cnn_model):
        super(ContentPart, self).__init__()
        self.room_size = room_size
        self.cnn_model = cnn_model
        self.padding = padding

    def forward(self, x):
        self.cnn_model.load_state_dict(torch.load(PATH_BEST_MODEL + self.cnn_model.name))
        x = self.cnn_model(x).exp()
        return x


class UserPart(nn.Module):
    """
    Model for the User-specific part.
    It suppose that we have an already trained user-model.
    The forward function receive the data in the same format as the Content 
    part (torch tensor), but the real sizes without padding for all the rooms
    have also to be provided.
    """

    def __init__(self, room_size, padding, user_model):
        super(UserPart, self).__init__()
        self.room_size = room_size
        self.user_model = user_model
        self.padding = padding

    def forward(self, x, sizes):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            x = to_numpy(x, self.room_size, sizes, self.padding)
            x = self.user_model.pipeline.compute_feature(x)
            x = self.user_model.predict_heatmap(x, sizes=sizes,
                                                size_with_padding=self.room_size,
                                                scaled_by="Sum")
            x = torch.Tensor(x).view(-1, self.room_size * self.room_size)
            x = x.to(device)
            return x


class Hybrid(nn.Module):
    def __init__(self, room_size, padding, init_value, content_model, user_model):
        super(Hybrid, self).__init__()

        self.name = "Hybrid"
        self.room_size = room_size
        self.padding = padding
        self.content_model = content_model
        self.user_model = user_model

        self.alpha = torch.nn.Parameter(torch.Tensor([init_value]))
        self.alpha.requires_grad = True

    def forward(self, x):
        with torch.no_grad():
            inputs, sizes = x
            # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            # mask = to_torch(inputs.cpu(),
            #                 self.padding,
            #                 self.room_size).view(-1, self.room_size * self.room_size)
            # mask[mask == -1] = 0
            # mask = mask.to(device)

            x1 = self.content_model(inputs)
            x2 = self.user_model(inputs, sizes.cpu())

        a = torch.clamp(self.alpha, 0, 1)
        x = F.sigmoid(a * x1 + (1-a) * x2)
        x = x.log()
        return x


def load_data_hybrid(X, Y, padding, room_size, params):
    """
    Compute the Dataloader with the datasets X and Y for Hybrid models.
    :param X: the inputs
    :param Y: the corresponding labels in a numpy format
    :param padding: an int that is the amount of padding to add
    :param room_size: the size of the room with the padding included
    :param params: a JSON file that contains the hyperparameters of the Hybrid
    model. Especially here, a batch size has to be provide.
    :return: a dataloader
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Collect the original size of data before convert to tensor and add padding
    sizes = [[len(X[i]), len(X[i][0])] for i in range(len(X))]
    sizes = torch.Tensor(sizes)
    torch_X, torch_Y = to_torch(x=X,
                                padding=padding,
                                room_size=room_size,
                                y=Y)

    torch_X, torch_Y = torch_X.to(device), torch_Y.to(device)

    # Computing the label that is the number of the class, from the spatial label:
    x_mat = torch_X.view(-1, room_size, room_size, 1)
    y_mat = torch_Y.view(-1, room_size * room_size)
    y_mat = torch.max(y_mat, 1)[1]

    mat = torch.utils.data.TensorDataset(x_mat, sizes, y_mat)
    loader_mat = torch.utils.data.DataLoader(mat, batch_size=params.batch_size, shuffle=True)

    return loader_mat

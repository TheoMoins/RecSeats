import torch
import torch.nn as nn
from src.preprocessing.data_loading import load_data_matrix
from math import ceil
from src.deep.deep_utils import masked_log_softmax
from src.room_transformation import keep_left_seat_torch, keep_right_seat_torch


class CNN(nn.Module):
    def __init__(self, room_size, params):

        super(CNN, self).__init__()

        self.name = "CNN"
        self.room_size = room_size
        self.nb_channels = params.nb_channels
        self.nb_conv_layers = params.nb_conv_layers

        self.output_size = self.room_size * self.room_size

        self.conv_layers = nn.ModuleList([nn.Sequential(nn.Conv2d(in_channels=1, out_channels=self.nb_channels,
                                                                  kernel_size=3, stride=1, padding=1),
                                                        # nn.BatchNorm2d(self.nb_channels),
                                                        nn.ReLU()),
                                          torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1)])

        size_in_ch = self.nb_channels
        for i in range(1, self.nb_conv_layers):
            self.conv_layers.append(nn.Sequential(nn.Conv2d(in_channels=size_in_ch, out_channels=2 * size_in_ch,
                                                            kernel_size=5, stride=1, padding=2),
                                                  # nn.BatchNorm2d(2 * size_in_ch),
                                                  nn.ReLU(),
                                                  torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1)))
            size_in_ch = 2 * size_in_ch

        if self.nb_conv_layers == 1:
            self.input_size_fc_layer = self.nb_channels * ceil(self.room_size / 2) ** 2
        else:
            self.input_size_fc_layer = size_in_ch * ceil(self.room_size / 2 ** self.nb_conv_layers) ** 2

        self.fc1 = nn.Sequential(
            nn.Linear(self.input_size_fc_layer, self.output_size),
            # nn.ReLU()
        )

    def forward(self, inputs):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        mask = inputs.view(-1, self.room_size * self.room_size)
        mask[mask == -1] = 0
        mask = mask.to(device)

        x = inputs.transpose(1, 3)
        for lay in self.conv_layers:
            x = lay(x)

        x = x.view(-1, self.input_size_fc_layer)
        x = self.fc1(x)
        x = masked_log_softmax(x, mask)
        return x


def load_data_loader(path, file_params, model_params, shuffle,
                     augmentation=False, verbose=True):
    """
    Compute the Dataloader with the datasets X and Y for CNN models.
    :param path: the path of the data
    :param file_params: a JSON file that contains the hyperparameters of the data.
    :param model_params: a JSON file that contains the hyperparameters of the deep
    model.
    :param shuffle: a boolean to indicate if data are shuffled in the dataloader
    :param augmentation: if true, augment data by doing the right seat transformation
    :param verbose: a boolean activating verbose mode
    :return: a dataloader
    """
    room_size = file_params.room_size
    padding = file_params.padding
    is_wso = file_params.is_wso
    batch_size = model_params.batch_size
    is_couple = file_params.is_couple
    to_tensor = model_params.to_tensor

    x_mat, y_mat = load_data_matrix(path=path,
                                    room_size=room_size,
                                    padding=padding,
                                    verbose=verbose,
                                    to_tensor=to_tensor,
                                    is_wso=is_wso)

    x_mat = x_mat.view(-1, room_size, room_size)
    y_mat = y_mat.view(-1, room_size, room_size)

    if is_couple:
        x_mat_1 = keep_left_seat_torch(x_mat)
        y_mat_1 = keep_left_seat_torch(y_mat)
        if augmentation:
            x_mat_2 = keep_right_seat_torch(x_mat)
            y_mat_2 = keep_right_seat_torch(y_mat)

            x_mat = torch.cat((x_mat_1, x_mat_2), 0)
            y_mat = torch.cat((y_mat_1, y_mat_2), 0)
        else:
            x_mat = x_mat_1
            y_mat = y_mat_1

    x_mat = x_mat.view(-1, room_size, room_size, 1)
    y_mat = y_mat.view(-1, room_size * room_size)

    if verbose:
        print("Shape of X:", x_mat.shape)
        print("Shape of Y:", y_mat.shape)
        print("")
    # y is now the label (ie the seat number) and not the whole size with one 1:
    y_mat = torch.max(y_mat, 1)[1]

    mat = torch.utils.data.TensorDataset(x_mat, y_mat)
    loader_mat = torch.utils.data.DataLoader(mat, batch_size=batch_size, shuffle=shuffle)

    return loader_mat

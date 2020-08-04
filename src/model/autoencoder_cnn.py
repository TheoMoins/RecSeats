import torch
import torch.nn as nn
import torch.nn.functional as F
from src.preprocessing.data_loading import load_data_matrix
from src.deep.deep_utils import masked_log_softmax


class AutoencoderCNN(nn.Module):
    def __init__(self, room_size, params):
        super(AutoencoderCNN, self).__init__()

        self.name = "Autoencoder_CNN"
        self.room_size = room_size
        self.nb_channels = params.nb_channels

        self.output_size = self.room_size * self.room_size

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=self.nb_channels,
                                             kernel_size=3, stride=1, padding=1),
                                   nn.ReLU(),
                                   nn.AvgPool2d(3)
                                   )

        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=self.nb_channels, out_channels=4 * self.nb_channels,
                                             kernel_size=3, stride=1, padding=1),
                                   nn.ReLU(),
                                   nn.AvgPool2d(4)
                                   )

        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=4 * self.nb_channels, out_channels=8 * self.nb_channels,
                                             kernel_size=5, stride=1, padding=2),
                                   nn.ReLU(),
                                   nn.AvgPool2d(4)
                                   )

        self.fc1 = nn.Sequential(nn.Linear(8 * self.nb_channels, 8 * self.nb_channels),
                                 # nn.ReLU()
                                 )
        # self.fc2 = nn.Sequential(nn.ReLU(),
        #                          nn.Linear(8 * self.nb_channels, 8 * self.nb_channels)
        #                          )

        self.deconv1 = nn.Sequential(nn.Conv2d(in_channels=8 * self.nb_channels,
                                               out_channels=4 * self.nb_channels,
                                               kernel_size=5, stride=1, padding=2),
                                     nn.ReLU()
                                     )

        self.deconv2 = nn.Sequential(nn.Conv2d(in_channels=4 * self.nb_channels,
                                               out_channels=self.nb_channels,
                                               kernel_size=5, stride=1, padding=2),
                                     nn.ReLU()
                                     )

        self.deconv3 = nn.Sequential(nn.Conv2d(in_channels=self.nb_channels,
                                               out_channels=1,
                                               kernel_size=3, stride=1, padding=1),
                                     nn.ReLU()
                                     )

        # self.fc_final = nn.Linear(self.room_size * self.room_size, self.room_size * self.room_size)

    def forward(self, inputs):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        mask = inputs.view(-1, self.room_size * self.room_size)
        mask[mask == -1] = 0
        mask = mask.to(device)

        x = inputs.transpose(1, 3)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x.view(-1, 8 * self.nb_channels)
        x = self.fc1(x)
        # x = self.fc2(x)
        x = x.view(-1, 8 * self.nb_channels, 1, 1)

        # Scale factor for room of size 57x57 : 3/4/4 -> 4/4.75/3
        x = self.deconv1(F.interpolate(x, scale_factor=4))
        x = self.deconv2(F.interpolate(x, scale_factor=4.75))
        x = self.deconv3(F.interpolate(x, scale_factor=3))

        x = x.view(-1, self.room_size * self.room_size)
        x = masked_log_softmax(x, mask)
        return x


def load_data_autoencoder(path, file_params, model_params, shuffle, verbose=True):
    """
    Compute the Dataloader with the datasets X and Y for CDNN models.
    :param path: the path of the data
    :param file_params: a JSON file that contains the hyperparameters of the data.
    :param model_params: a JSON file that contains the hyperparameters of the deep
    model.
    :param shuffle: a boolean to indicate if data are shuffled in the dataloader
    :param verbose: a boolean activating verbose mode
    :return: a dataloader
    """
    room_size = file_params.room_size
    padding = file_params.padding
    is_wso = file_params.is_wso
    batch_size = model_params.batch_size
    to_tensor = model_params.to_tensor

    x_mat, y_mat = load_data_matrix(path=path,
                                    room_size=room_size,
                                    padding=padding,
                                    verbose=verbose,
                                    to_tensor=to_tensor,
                                    is_wso=is_wso)

    x_mat = x_mat.view(-1, room_size, room_size, 1)
    y_mat = x_mat.view(-1, room_size, room_size, 1) + y_mat.view(-1, room_size, room_size, 1)

    if verbose:
        print("Shape of X:", x_mat.shape)
        print("Shape of Y:", y_mat.shape)
        print("")

    mat = torch.utils.data.TensorDataset(x_mat, y_mat)
    loader_mat = torch.utils.data.DataLoader(mat, batch_size=batch_size, shuffle=shuffle)

    return loader_mat

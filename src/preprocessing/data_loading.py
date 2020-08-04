import torch
import torch.nn as nn
import numpy as np

import csv


def torch_with_padding(inputs, padding, room_size, outputs=None):
    """
    Add padding to numpy arrays.
    The padding is applied in order to keep the image at the center :
    (same padding on the left and on the right if there is, and same for up and down)
    :param inputs: a numpy arrays containing the inputs
    :param padding: the padding values
    :param room_size: the futur size after padding
    :param outputs: (optional), a numpy array on which padding can also be applied
    :return: a tensor
    """
    """

    """
    p = padding
    rs = room_size
    pad1 = nn.ZeroPad2d((p, p, p, p))
    pad2 = nn.ZeroPad2d((p, p, 0, 0))
    pad3 = nn.ZeroPad2d((0, 0, p, p))
    old_room_size = rs - 2 * p
    for idx_room, room in enumerate(inputs):
        if len(room) == old_room_size:
            if len(room[0]) == old_room_size:  # Padding on both directions
                inputs[idx_room] = pad1(torch.Tensor(room)).view(rs, rs, 1)
                if outputs is not None:
                    outputs[idx_room] = pad1(torch.Tensor(outputs[idx_room])).view(-1)

            else:  # Padding only on the rows
                inputs[idx_room] = pad3(torch.Tensor(room)).view(rs, rs, 1)
                if outputs is not None:
                    outputs[idx_room] = pad3(torch.Tensor(outputs[idx_room])).view(-1)

        else:
            if len(room[0]) == old_room_size:  # Padding only on the columns
                inputs[idx_room] = pad2(torch.Tensor(room)).view(rs, rs, 1)
                if outputs is not None:
                    outputs[idx_room] = pad2(torch.Tensor(outputs[idx_room])).view(-1)

            else:  # No padding
                inputs[idx_room] = torch.Tensor(room).view(rs, rs, 1)
                if outputs is not None:
                    outputs[idx_room] = torch.Tensor(outputs[idx_room]).view(-1)

    if outputs is not None:
        return torch.stack(list(inputs)), torch.stack(list(outputs))
    return torch.stack(list(inputs))


def init_room():
    """
    Use for concert hall data, to initialise an empty room
    :return: a 2D array
    """
    room = np.zeros((57, 57))
    seats_per_row = [24, 26, 30, 30, 32, 37, 40, 45, 46, 47, 48, 51, 46, 51, 54,
                     56, 54, 57, 56, 53, 56, 55, 54, 55, 56, 53, 54, 55, 52, 53, 52]
    for i in range(3, 31):
        ind_first_seat = (58 - seats_per_row[i]) // 2
        ind_last_seat = (56 + seats_per_row[i]) // 2
        for j in range(57):
            if ind_first_seat <= j <= ind_last_seat:
                room[i][j] = -1
    return room


def add_padding_wso(room):
    """
    Apply the same padding as the one of the concert hall for a room
    of the same size
    :param room: a numpy array
    :return: a numpy array
    """
    i_room = init_room()
    for i in range(len(room)):
        for j in range(len(room[0])):
            if i_room[i][j] == 0:
                room[i][j] = 0
    return room


def load_data_matrix(path, room_size, padding,
                     to_tensor, verbose=True,
                     nb_choice=None, nb_clients=None,
                     is_wso=False):
    """
    Returns two tensors, that contains the inputs and the outputs.
    The matrix contains 1 for every available seat, and 0 otherwise.
    :param path: A string that contains the name of the file
    :param room_size: The size of one side the room (the biggest possible if various size)
    :param padding: The padding needed if various size
    :param to_tensor: integer that indicate if the output is a pytorch tensor or not
    (otherwise it's a numpy array)
        if 0 : numpy array
        if 1 : pytorch tensor, with a dimension for each customer
        if 2 : pytorch tensor, with no distinction between customers
    :param verbose: a boolean, indicating verbose mode
    :param nb_choice:  Number of choice kept per clients.
    If None: the maximal number is kept, but in a way that all
    the clients has the same history size.
    :param nb_clients: Number of clients kept
    :param is_wso: indicating if we treat concert_hall_data
    :return:
    """

    #    delimiter = ";" if is_wso else ","
    with open(path, "r") as csvfile:
        file_size = sum(1 for _ in csvfile)
    # Opening file:
    with open(path, "r") as csvfile:
        reader = csv.reader(csvfile, delimiter=",", quotechar='"')

        # Initialization of arrays:
        inputs = [[]]
        outputs = [[]]

        #  Taking the header and the index of useful columns:
        header = next(reader)

        ind_ncols = header.index('Cond_NCol')
        ind_nrows = header.index('Cond_NRows')
        ind_row_available = header.index('RowNumber_Avail')
        ind_col_available = header.index('ColNumber_Avail')
        ind_chosen = header.index('Chosen')

        previous, nb_inputs = (-1, -1, -1), 0

        nb_id = 0 if to_tensor != 1 else 1

        for i_row, row in enumerate(reader):
            if verbose:
                print("\r{:.2%}".format(i_row / file_size), end="")
            if previous != (float(row[0]), float(row[1]), float(row[2])):
                nb_inputs += 1

                if is_wso:
                    new_input = init_room()
                    new_output = [[0 for _ in range(57)] for _ in range(57)]
                else:
                    new_input = [[-1 for _ in range(int(row[ind_ncols]))] for _ in range(int(row[ind_nrows]))]
                    new_output = [[0 for _ in range(int(row[ind_ncols]))] for _ in range(int(row[ind_nrows]))]

                if to_tensor != 1 and previous[0] != float(row[0]):  # New person to consider
                    inputs.append([])
                    outputs.append([])
                    nb_id += 1

                inputs[-1].append(new_input)
                outputs[-1].append(new_output)

                previous = (float(row[0]), float(row[1]), float(row[2]))

            seat_col = int(row[ind_col_available])
            seat_row = int(row[ind_row_available])

            inputs[-1][-1][seat_row - 1][seat_col - 1] = 1

            # Update the output with the chosen place, except if it's on (0,0), which signify no place chosen:
            if (int(row[ind_chosen]) == 1) and seat_row != 0 and np.sum(outputs[-1][-1]) == 0:
                outputs[-1][-1][seat_row - 1][seat_col - 1] = 1
        if verbose:
            print("\nLoad {} examples as matrices for each of the {} customers\n".format(nb_inputs, nb_id))

        if nb_clients is not None:
            inputs = inputs[:nb_clients]
            outputs = outputs[:nb_clients]

        if to_tensor != 1:

            inputs = inputs[1:]
            outputs = outputs[1:]

            min_nb_choices = min([len(to) for to in outputs])

            if nb_choice is None:
                nb_choice = min_nb_choices
            else:
                assert nb_choice < min_nb_choices

            for i in range(len(inputs)):
                inputs[i] = inputs[i][:nb_choice]
                outputs[i] = outputs[i][:nb_choice]

        torch_inputs = []
        torch_outputs = []

        if padding != 0:

            for ind_id, inputs_id in enumerate(inputs):
                inp, out = torch_with_padding(inputs_id, padding, room_size, outputs[ind_id])
                torch_inputs.append(inp)
                torch_outputs.append(out)

            torch_inputs, torch_outputs = torch.stack(torch_inputs), torch.stack(torch_outputs)

            if to_tensor:
                return torch_inputs, torch_outputs

            else:
                return torch.squeeze(torch_inputs).numpy(), \
                       torch_outputs.view(nb_id, min_nb_choices, room_size, room_size).numpy()

        else:
            if to_tensor == 0:
                return np.asarray(inputs), np.asarray(outputs)

            elif to_tensor == 2:
                torch_outputs = torch.Tensor(outputs).view(nb_id, min_nb_choices, room_size * room_size)
                torch_inputs = torch.Tensor(inputs).view(nb_id, min_nb_choices, int(room_size), int(room_size), 1)

        if to_tensor:
            return torch.Tensor(inputs), torch.Tensor(outputs)

        else:
            return torch.squeeze(torch_inputs).numpy(), \
                   torch_outputs.view(nb_id, min_nb_choices, room_size, room_size).numpy()

import numpy as np
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def are_next_to(orders):
    """
    Used for Orchestra Hall preprocessing, to indicate if two locations are next to each others.
    :param orders: the orchestra hall pandas dataframe of orders.
    :return: a boolean
    """
    resu = orders["Row Number"].values[0] == orders["Row Number"].values[1] and \
           abs(orders["Seat Number"].values[0] - orders["Seat Number"].values[1]) == 1
    return resu


def keep_left_seat(room):
    """
    Modify a room so as to make the seats that are to the left
    of unavailable (or non-existent) seats unavailable.
    Used for couple seat orders
    :param room: An input data, 2D Array
    :return: The transformed room
    """
    for i in range(len(room)):
        for j in range(1, len(room[0])):
            if room[i][j - 1] == 1 and room[i][j] != 1:
                room[i][j - 1] = -1
        if room[i][-1] == 1:
            room[i][-1] = -1
    return room


def keep_right_seat(room):
    """
    Modify a room so as to make the seats that are to the right
    of unavailable (or non-existent) seats unavailable.
    Used for couple seat orders
    :param room: An input data, 2D Array
    :return: The transformed room
    """
    return np.flip(keep_left_seat(np.flip(room, axis=1)), axis=1)


def inverse_keep_left_seat(room):
    """
    Do the inverse operation of keep_left_seat, to obtain
    the original from the transformed one.
    :param room: a 2D Array
    :return: a 2D Array
    """
    room = np.flip(room, axis=1)
    for i in range(len(room)):
        for j in range(1, len(room[0])):
            if room[i][j - 1] == -1 and room[i][j] == 1:
                room[i][j - 1] = 1
    return np.flip(room, axis=1)


def keep_left_seat_torch(inputs):
    """
    Do the same thing as keep_left_seat, adapted for
    torch tensor inputs.
    :param inputs: a tensor of inputs
    :return: a tensor of inputs
    """
    room_size = inputs.shape[2]
    update_inputs = []
    for room in inputs.view(-1, room_size, room_size):
        to_add = torch.Tensor(keep_left_seat(room.cpu()))
        update_inputs.append(to_add)

    update_inputs = torch.stack(update_inputs).to(device)
    update_inputs = update_inputs.view(-1, room_size, room_size, 1)

    return update_inputs


def keep_right_seat_torch(inputs):
    """
    Do the same thing as keep_right_seat, adapted for
    torch tensor inputs.
    :param inputs: a tensor of inputs
    :return: a tensor of inputs
    """
    room_size = inputs.shape[2]
    update_inputs = []
    for room in inputs.view(-1, room_size, room_size):
        to_add = torch.Tensor(keep_right_seat(room.cpu()))
        update_inputs.append(to_add)

    update_inputs = torch.stack(update_inputs).to(device)
    update_inputs = update_inputs.view(-1, room_size, room_size, 1)

    return update_inputs


def inverse_keep_left_seat_torch(inputs):
    """
    Do the same thing as inverse_keep_left_seat, adapted for
    torch tensor inputs.
    :param inputs: a tensor of inputs
    :return: a tensor of inputs
    """
    room_size = inputs.shape[2]
    update_inputs = []
    for room in inputs.view(-1, room_size, room_size):
        to_add = torch.Tensor(inverse_keep_left_seat(room.cpu()))
        update_inputs.append(to_add)

    update_inputs = torch.stack(update_inputs).to(device)
    update_inputs = update_inputs.view(-1, room_size, room_size, 1)

    return update_inputs

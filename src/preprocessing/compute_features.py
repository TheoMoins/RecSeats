import numpy as np


# from sklearn.decomposition import TruncatedSVD, PCA


def compute_index(position, i, size_i, size_j):
    """
    compute the index of the matrix that is on a L1 distance
    equal to i of the position.
    :param position: tuple that contains the position of the coefficient
    :param i: size of the L1-radius
    :param size_i: number of rows
    :param size_j: number of lines
    :return: list that contains tuple for positions
    """
    resu = []
    y, x = position
    for k in range(y - i, y + i + 1):
        if 0 <= k < size_i:
            if x - i >= 0:
                resu.append((k, x - i))
            if x + i < size_j:
                resu.append((k, x + i))
    for l in range(x - i + 1, x + i):
        if 0 <= l < size_j:
            if y - i >= 0:
                resu.append((y - i, l))
            if y + i < size_i:
                resu.append((y + i, l))
    return resu


class Feature_Pipeline:

    def __init__(self, params):

        self.beta_position = (params.beta['beta_position'] == "True")
        self.beta_r1 = (params.beta['beta_r1'] == "True")
        self.beta_r2 = (params.beta['beta_r2'] == "True")
        self.beta_r3 = (params.beta['beta_r3'] == "True")
        self.beta_ps = (params.beta['beta_ps'] == "True")

        #        self.n_beta_pos = params.feature_size_position
        #        self.n_beta_ri = params.feature_size_Rn
        self.nb_features = int(self.beta_position) * 5 + \
                           int(self.beta_r1) + \
                           int(self.beta_r2) + \
                           int(self.beta_r3) + \
                           int(self.beta_ps) * 7

    def compute_x_ri(self, torch_input, position, i, size_i, size_j):
        resu = 0
        index_positions = compute_index(position, i, size_i, size_j)

        for (yi, xi) in index_positions:
            if torch_input[yi][xi] == 1:
                resu += 1
        resu = resu / len(index_positions)
        return [resu]

    def compute_x_position(self, position, size_i, size_j):
        y, x = position
        x_norm, y_norm = x / size_j, y / size_i
        return [x_norm, y_norm, x_norm * x_norm, y_norm * y_norm, x_norm * y_norm]

    def compute_x_ps(self, torch_input, position, size_i, size_j):
        y, x = position
        resu = []

        # x_left
        if x - 1 >= 0:
            if torch_input[y][x - 1] == 1:
                resu.append(1)
            else:
                resu.append(0)
        else:
            resu.append(0)

        # x_right
        if x + 1 < size_j:
            if torch_input[y][x+1] == 1:
                resu.append(1)
            else:
                resu.append(0)
        else:
            resu.append(0)

        # x_leftright
        if x - 1 >= 0 and x + 1 < size_j:
            if torch_input[y][x-1] == 1 and torch_input[y][x+1] == 1:
                resu.append(1)
            else:
                resu.append(0)
        else:
            resu.append(0)

        # x_front
        if y - 1 >= 0:
            if torch_input[y-1][x] == 1:
                resu.append(1)
            else:
                resu.append(0)
        else:
            resu.append(0)

        # x_back
        if y + 1 < size_i:
            if torch_input[y+1][x] == 1:
                resu.append(1)
            else:
                resu.append(0)
        else:
            resu.append(0)

        # x_front_corner
        x_fc = 0
        if x - 1 >= 0 and y - 1 >= 0:
            if torch_input[y - 1][x - 1] == 1:
                x_fc += 1
        if y - 1 >= 0 and x + 1 < size_j:
            if torch_input[y - 1][x + 1] == 1:
                x_fc += 1
        resu.append(x_fc / 2)

        # x_back_corner
        x_bc = 0
        if x - 1 >= 0 and y + 1 < size_i:
            if torch_input[y + 1][x - 1] == 1:
                x_bc += 1
        if x + 1 < size_j and y + 1 < size_i:
            if torch_input[y + 1][x + 1] == 1:
                x_bc += 1
        resu.append(x_bc / 2)

        return resu

    def compute_feature_position(self, position, input_, size_i, size_j):
        """
        Compute a feature for a given position on a given room.
        """
        f = []
        if self.beta_position:
            f += self.compute_x_position(position, size_i, size_j)
        if self.beta_r1:
            f += self.compute_x_ri(input_, position, 1, size_i, size_j)
        if self.beta_r2:
            f += self.compute_x_ri(input_, position, 2, size_i, size_j)
        if self.beta_r3:
            f += self.compute_x_ri(input_, position, 3, size_i, size_j)
        if self.beta_ps:
            f += self.compute_x_ps(input_, position, size_i, size_j)
        return f

    def compute_feature_matrix(self, input_, output_=None):
        """
        Compute the features of all positions in a room.
        """
        mat_in = []
        mat_out = []
        size_i = len(input_)
        size_j = len(input_[0])
        prob = 0
        for i in range(size_i):
            for j in range(size_j):
                if input_[i][j] == 1:
                    mat_in.append(self.compute_feature_position((i, j), input_, size_i, size_j))
                    if output_ is not None:
                        mat_out.append(output_[i][j])
                    prob += 1
        if output_ is not None:
            return mat_in, mat_out
        return mat_in

    def compute_feature(self, inputs_, outputs_=None):
        """
        Compute the features for all the rooms in a list.
        """
        resu_in, resu_out = [], []
        for i in range(inputs_.shape[0]):

            if outputs_ is not None:
                mat_in, mat_out = self.compute_feature_matrix(inputs_[i], outputs_[i])
                resu_out.append(mat_out)
            else:
                mat_in = self.compute_feature_matrix(inputs_[i], None)

            resu_in.append(mat_in)

        if outputs_ is not None:
            return np.asarray(resu_in), np.asarray(resu_out)
        return np.asarray(resu_in)

    def compute_feature_all_users(self, X, Y, n_clients):
        """
        Compute the features for a list of users.
        """
        train_valid_X, train_valid_Y = [], []
        for i in range(n_clients):
            train_valid_X_i, train_valid_Y_i = self.compute_feature(X[i], Y[i])
            train_valid_X.append(train_valid_X_i)
            train_valid_Y.append(train_valid_Y_i)

        return train_valid_X, train_valid_Y

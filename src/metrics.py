import numpy as np
import torch

PATH_BEST_MODEL = "./save/best_models/best_model_"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def top_n_accuracy(Y_true, Y_pred, n=1):
    """
    Return the top N accuracy for a given prediction function
    :param Y_pred: the Top-N classes (array of size N for each input)
    :param Y_true: the labels
    :param n: The N of Top N Accuracy
    :return: the accuracy (float)
    """
    acc = 0
    for i in range(len(Y_pred)):
        class_label = np.where(np.asarray(Y_true[i]) == 1)[0]
        if class_label in Y_pred[i][0:n]:
            acc += 1

    acc = acc / len(Y_pred)
    return acc


def weighted_l1(user_model, X, Y_true, sizes):
    """
    compute the weighted L1 distance of an user-specific model predictions
    :param user_model: the user_specific model
    :param X: the input
    :param Y_true: the labels
    :param sizes: sizes of all the rooms X (array of size nb_rooms x 2)
    :return: a float corresponding to the loss value
    """
    res = 0
    for i in range(len(Y_true)):
        pos_prob = user_model.predict_pos_proba([X[i]],
                                                size_i=sizes[i][0], size_j=sizes[i][1],
                                                N_MAX=sizes[i][0] * sizes[i][1],
                                                scaled_by="Sum")[0]
        label = np.where(np.asarray(Y_true[i]) == 1)[0][0]
        label_pos = (int(X[i][label][0] * sizes[i][1]), int(X[i][label][1] * sizes[i][0]))
        for p in range(len(pos_prob)):
            res += pos_prob[p][2] * (abs(label_pos[0] - pos_prob[p][0]) + abs(label_pos[1] - pos_prob[p][1]))
    return res / len(Y_true)


def torch_top_n_accuracy(outputs, labels, N=1):
    """
    Same function as top_n_accuracy, adapted for pytorch tensors and deep models.
    :param outputs: the prediction of the network (Y_pred, corresponding at the index of the chosen seat)
    :param labels: the labels (Y_true)
    :param N: le N de la topN accuracy
    :return: a float corresponding to the accuracy
    """
    topN = torch.sort(outputs, dim=1, descending=True)[1][:, 0:N]
    ciN = 0
    for i in range(labels.shape[0]):
        if labels[i] in topN[i]:
            ciN += 1
    return ciN / labels.shape[0]


def torch_weighted_l1(outputs, labels):
    """
    same function as weighted L1, adapted for pytorch architecture
    :param outputs: outputs: the prediction of the network (Y_pred, corresponding at the index of the chosen seat)
    :param labels: labels: the labels (Y_true)
    :return: a float corresponding to the accuracy
    """
    res = 0
    room_size = int(np.sqrt(outputs.shape[1]))

    sort_output = torch.sort(outputs, dim=1, descending=True)
    label_pred = sort_output[1]
    sort_prob = sort_output[0].exp()
    labels_pos = [(x.item() // room_size, x.item() % room_size) for x in labels]

    p_max = len(sort_prob[0])
    # if p_max > 1000:
    #     p_max = 1000

    for i in range(len(sort_output[0])):
        prob = (sort_prob[i] - torch.min(sort_prob[i]))/(torch.sum(sort_prob[i]-torch.min(sort_prob[i])))
        label_pred_i = [(x.item() // room_size, x.item() % room_size) for x in label_pred[i]]
        for p in range(p_max):
            res += prob[p] * (abs(labels_pos[i][0] - label_pred_i[p][0]) +
                                      abs(labels_pos[i][1] - label_pred_i[p][1]))
    return res.item() / len(sort_output[0])

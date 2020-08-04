import numpy as np
import csv


def get_ids_from_file(path, label):
    ids = []
    with open(path, "r") as csvfile:
        reader = csv.reader(csvfile, delimiter=",", quotechar='"')

        header = next(reader)

        previous = (-1, -1, -1)

        for i_row, row in enumerate(reader):

            if previous[0] != float(row[0]):
                ids.append([])

            if previous != (float(row[0]), float(row[1]), float(row[2])):
                ids[-1].append([float(row[0]), float(row[1]), float(row[2]), label])

            previous = (float(row[0]), float(row[1]), float(row[2]))

    return ids


def get_single_or_couple_label(path_single, path_couple, i):
    ids_single = get_ids_from_file(path_single, label=1)
    ids_couple = get_ids_from_file(path_couple, label=2)

    idx_s = 0
    idx_c = 0
    res = []

    while idx_s < len(ids_single) and idx_c < len(ids_couple):

        client_id_single = ids_single[idx_s][0][0]
        client_id_couple = ids_couple[idx_c][0][0]

        if client_id_couple < client_id_single and len(ids_couple[idx_c]) == i:
            res.append([2] * i)
            idx_c += 1

        elif ids_single[idx_s][0][0] == ids_couple[idx_c][0][0]:
            tmp = ids_single[idx_s] + ids_couple[idx_c]
            tmp = sorted(tmp, key=lambda x: (x[1], x[2]))
            res.append([x[3] for x in tmp])
            idx_s += 1
            idx_c += 1

        elif len(ids_single[idx_s]) == i:
            res.append([1] * i)
            idx_s += 1

    while idx_c < len(ids_couple):
        client_id_couple = ids_couple[idx_c][0][0]

        if len(ids_couple[idx_c]) == i:
            res.append([2] * i)
            idx_c += 1

    while idx_s < len(ids_single):
        client_id_single = ids_single[idx_s][0][0]

        if len(ids_single[idx_s]) == i:
            res.append([1] * i)
            idx_s += 1

    return res

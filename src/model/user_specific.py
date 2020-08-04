import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from scipy.special import softmax

import time


class Recommendation:

    def __init__(self, params, pipeline):

        self.params = params
        self.pipeline = pipeline
        self.classifier = None
        self.set_classifier(self.params.classifier)

    def set_classifier(self, classifier):
        if classifier == "LR":
            self.classifier = LogisticRegression(C=self.params.C,
                                                 class_weight=self.params.param_logistic['class_weight'],
                                                 fit_intercept=True,
                                                 penalty=self.params.param_logistic['penalty'],
                                                 solver=self.params.param_logistic['solver'],
                                                 random_state=15,
                                                 max_iter=300,
                                                 n_jobs=-1)

        elif classifier == "SVC":
            self.classifier = SVC(C=self.params.C,
                                  kernel=self.params.kernel_svc,
                                  probability=True,
                                  class_weight="balanced",
                                  max_iter=2000,
                                  random_state=15)

        elif classifier == "AdaBoost":
            self.classifier = AdaBoostClassifier(DecisionTreeClassifier(criterion='gini',
                                                                        splitter='best',
                                                                        max_depth=self.params.param_tree['max_depth'],
                                                                        min_samples_split=self.params.param_tree[
                                                                            'min_samples_split'],
                                                                        min_samples_leaf=self.params.param_tree[
                                                                            'min_samples_leaf'],
                                                                        random_state=15,
                                                                        class_weight='balanced',
                                                                        presort=False),
                                                 n_estimators=50,
                                                 random_state=15)

        elif classifier == "RF":
            self.classifier = RandomForestClassifier(n_estimators=self.params.param_forest['n_estimators'],
                                                     min_samples_split=self.params.param_tree['min_samples_split'],
                                                     min_samples_leaf=self.params.param_tree['min_samples_leaf'],
                                                     max_features=2,
                                                     max_depth=self.params.param_tree['max_depth'],
                                                     n_jobs=-1,
                                                     class_weight="balanced",
                                                     random_state=15)

        elif classifier == "MLP":
            self.classifier = MLPClassifier(hidden_layer_sizes=(50, 25, 10),
                                            activation='relu',
                                            solver='lbfgs',
                                            alpha=0.0001,
                                            max_iter=200,
                                            shuffle=True,
                                            random_state=15,
                                            tol=0.0001,
                                            verbose=True,
                                            early_stopping=True,
                                            validation_fraction=0.1,
                                            n_iter_no_change=10)

        elif classifier == "Tree":
            self.classifier = DecisionTreeClassifier(criterion='gini',
                                                     splitter='best',
                                                     max_depth=self.params.param_tree['max_depth'],
                                                     min_samples_split=self.params.param_tree['min_samples_split'],
                                                     min_samples_leaf=self.params.param_tree['min_samples_leaf'],
                                                     random_state=15,
                                                     class_weight='balanced',
                                                     presort=False)

        elif classifier == "GBT":
            self.classifier = GradientBoostingClassifier(learning_rate=0.1,
                                                         n_estimators=self.params.param_forest['n_estimators'],
                                                         subsample=1,
                                                         max_depth=self.params.param_tree['max_depth'],
                                                         min_samples_split=self.params.param_tree['min_samples_split'],
                                                         min_samples_leaf=self.params.param_tree['min_samples_leaf'],
                                                         random_state=15,
                                                         max_features=2)

        else:
            print("UNKNOWN CLASSIFIER : Choose between LR/SVC/AdaBoost/RF/MLP/Tree/GBT")

    def fit(self, train_X_i, train_Y_i):
        flat_train_X = np.concatenate(train_X_i, axis=0)
        flat_train_Y = np.concatenate(train_Y_i, axis=0)
        return self.classifier.fit(flat_train_X, flat_train_Y)

    def get_params(self, deep=False):
        return {"params": self.params,
                "pipeline": self.pipeline}

    def train_evaluate(self, X, Y, evals_scorer, test_X=None, test_Y=None, verbose=False):

        train_acc_list = []
        valid_acc_list = []
        cross_acc_list = []

        beta_list = []
        time_perf = []

        compute_test = True
        if test_X is None:
            compute_test = False

        n_clients = self.params.n_clients
        if n_clients == "None":
            n_clients = len(X)

        compute_cv = False
        if self.params.compute_cross_val == "True":
            compute_cv = True

        _X, _Y = self.pipeline.compute_feature_all_users(X, Y, n_clients)

        if compute_test:
            _test_X, _test_Y = self.pipeline.compute_feature_all_users(test_X, test_Y, n_clients)

        for i in range(n_clients):
            before = time.time()
            train_valid_X, train_valid_Y = _X[i], _Y[i]

            if compute_test:
                test_X_i, test_Y_i = _test_X[i], _test_Y[i]

                self.fit(train_valid_X, train_valid_Y)

                train_acc = [evals_scorer[s](self, train_valid_X, train_valid_Y)
                             for s in range(len(evals_scorer))]

                valid_acc = [evals_scorer[s](self, test_X_i, test_Y_i)
                             for s in range(len(evals_scorer))]

            else:
                train_X_i, valid_X_i, train_Y_i, valid_Y_i = train_test_split(train_valid_X,
                                                                              train_valid_Y,
                                                                              test_size=0.20,
                                                                              random_state=15)
                self.fit(train_X_i, train_Y_i)

                train_acc = [evals_scorer[s](self, train_X_i, train_Y_i)
                             for s in range(len(evals_scorer))]

                valid_acc = [evals_scorer[s](self, valid_X_i, valid_Y_i)
                             for s in range(len(evals_scorer))]

            train_acc_list.append(train_acc)
            valid_acc_list.append(valid_acc)

            if compute_cv:
                cross_acc = [np.mean(cross_val_score(self, train_valid_X, train_valid_Y,
                                                     cv=5, scoring=evals_scorer[s], n_jobs=-1))
                             for s in range(len(evals_scorer))]
                cross_acc_list.append(cross_acc)

            if verbose:
                print("\nClient n°{}:".format(i + 1))
                print("  Training Set:")
                print("      Acc "+' Acc '.join('{}: {:.2%}'.format(*k) for k in enumerate(train_acc)))
                print("  Validation Set:")
                print("      Acc "+' Acc '.join('{}: {:.2%}'.format(*k) for k in enumerate(valid_acc)))

                if compute_cv:
                    print("  Cross-Validation:")
                    print("      Acc "+' Acc '.join('{}: {:.2%}'.format(*k) for k in enumerate(cross_acc)))

            if self.params.classifier in ["LR"]:
                beta_list.append(self.classifier.coef_[0])
                if verbose:
                    print("  Mean Beta coefficient : {:.3f}".format(np.mean(self.classifier.coef_[0])))

            now = time.time()
            time_perf.append(now - before)
            if not verbose:
                print("\r", end="")
                print("\r{}/{}, {:8.1f} seconds remaining"
                      .format(i + 1, n_clients, (n_clients - i - 1) * np.mean(time_perf)), end="")

        print("")
        print("####### Mean of Accuracies #######")

        mean_train = np.mean(train_acc_list, axis=0)
        mean_valid = np.mean(valid_acc_list, axis=0)
        print("Mean Training Acc:")
        print("      Acc "+' Acc '.join('{}: {:.2%}'.format(*k) for k in enumerate(mean_train)))
        print("Mean Validation Acc:")
        print("      Acc "+' Acc '.join('{}: {:.2%}'.format(*k) for k in enumerate(mean_valid)))

        if compute_cv:
            mean_cross = np.mean(cross_acc_list, axis=0)
            print("Mean Cross-Validation Acc:")
            print("      Acc "+' Acc '.join('{}: {:.2%}'.format(*k) for k in enumerate(mean_cross)))
        print("")

        return np.asarray(beta_list), train_acc_list, valid_acc_list, cross_acc_list

    def predict(self, X):
        """
        Predict the top_N most coveated seats, for all rooms.
        If the label = k, then it corresponds to the k^th line in the feature matrix
        :param X: The input, a numpy array
        :return: a list, containing predictions
        """
        N_MAX = 30
        res = []
        for (i, room) in enumerate(X):
            prob = np.array(self.classifier.predict_proba(room))[:, 1]
            sort_prob = np.flip(np.sort(prob))[0:N_MAX]
            res.append([list(prob).index(p) for p in sort_prob])
        return res

    def predict_pos(self, X, size_i, size_j, N_MAX=5):
        """
        Predict the 2D position of the top_N most coveated seats, for all rooms.
        :param X: The input, a numpy array
        :param size_i: the number of line of the room
        :param size_j: the number of column of the room
        :param N_MAX: The maximum number on which we compute Top N Accuracy
        :return: a list containing coordinates of top N most coveated seats
        """
        res = []
        for room in X:
            prob = np.array(self.classifier.predict_proba(room))[:, 1]
            N_final = N_MAX if N_MAX < len(prob) else len(prob)
            sort_prob = np.flip(np.sort(np.unique(prob)))

            index_label = []
            for p in sort_prob:
                index_label.extend([i for i, x in enumerate(list(prob)) if x == p])
            index_label = index_label[0:N_final]

            res.append([(int(room[i][0] * size_j), int(room[i][1] * size_i)) for i in index_label])
        return res

    def predict_proba(self, X, N_MAX=5):
        """
        Predict the probability of the top_N most coveated seats, for all rooms.
        :param X: The input, a numpy array
        :param N_MAX: The maximum number on which we compute Top N Accuracy
        :return: a list containing top N probabilities for all rooms in X
        """
        res = []
        for room in X:
            prob = np.array(self.classifier.predict_proba(room))[:, 1]
            sort_prob = np.flip(np.sort(prob))[0:N_MAX]
            res.append(list(sort_prob))
        return res

    def predict_pos_proba(self, X, size_i, size_j, N_MAX=5, scaled_by="MinMax"):
        """
        Predict the 2D position + SCALED probability of the top_N most coveated seats,
        for all rooms.
        :param X: Input matrix (after the pipeline)
        :param size_i: number of line
        :param size_j: number of column
        :param N_MAX: the number of value to compute (if it's greater than the number of
        available seats, it become the number of available seats)
        :param scaled_by: If "MinMax": return the min-max scaling of the prob (useful for plotting)
                       If "Sum": return the prob divided by the sum of all the prob
                       else: no transformations
        :return: A list of triplet [(xi, yi, pi)]
        """
        res = []
        for room in X:
            prob = np.array(self.classifier.predict_proba(room))[:, 1]
            N_final = N_MAX if N_MAX < len(prob) else len(prob)
            sort_prob = np.flip(np.sort(prob))[0:N_final]

            index_label = []
            for p in np.flip(np.sort(np.unique(prob))):
                index_label.extend([i for i, x in enumerate(list(prob)) if x == p])
            index_label = index_label[0:N_final]

            if scaled_by == "MinMax":
                sort_prob = [(x - sort_prob[-1]) / (sort_prob[0] - sort_prob[-1]) for x in sort_prob]
            if scaled_by == "Sum":
                sort_prob = [x/sum(sort_prob) for x in sort_prob]
            res.append([(int(room[i][0] * size_j),
                         int(room[i][1] * size_i),
                         sort_prob[idx])
                        for idx, i in enumerate(index_label)])
        return res

    def predict_heatmap(self, X, sizes=None, size_with_padding=57, scaled_by="Sum"):
        """
        Predict the probability of the top_N most coveated seats, for all rooms.
        :param X: Input matrix (after the pipeline)
        :param sizes: a list containing the size of all rooms
        :param size_with_padding: an int giving the size after padding on all rooms
        :param scaled_by: the way we scale probabilities
        :return: a numpy array of heatmaps
        """
        res = []
        if sizes is not None:
            sizes = sizes.detach().cpu().numpy()
        else:
            sizes = [(57, 57) for _ in range(len(X))]

        for (idx_room, room) in enumerate(X):
            mat = np.zeros((size_with_padding, size_with_padding))

            prob = np.array(self.classifier.predict_proba(room))[:, 1]
            if scaled_by == "MinMax":
                prob = [(x - prob[-1]) / (prob[0] - prob[-1]) for x in prob]
            if scaled_by == "Sum":
                s = sum(prob)
                prob = [x/s for x in prob]

            for i in range(len(prob)):
                pos_x, pos_y = (int(room[i][0] * sizes[idx_room][1]),
                                int(room[i][1] * sizes[idx_room][0]))

                if sizes[idx_room][0] < size_with_padding:
                    pos_y += (size_with_padding - sizes[idx_room][0]) // 2
                if sizes[idx_room][1] < size_with_padding:
                    pos_x += (size_with_padding - sizes[idx_room][1]) // 2

                mat[int(pos_y)][int(pos_x)] = prob[i]

            res.append(mat)
        return np.asarray(res)

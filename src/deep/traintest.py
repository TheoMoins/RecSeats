import torch
import torch.nn.functional as F
import numpy as np

from copy import deepcopy

from src.room_transformation import inverse_keep_left_seat_torch, keep_right_seat_torch

PATH_BEST_MODEL = "./save/best_models/best_model_"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TrainTest:
    def __init__(self, model, train_set, valid_set,
                 optimizer, loss_fn, eval_fns, two_inputs=False):

        self.model = model
        self.train_set = train_set
        self.valid_set = valid_set
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.eval_fns = eval_fns
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.two_inputs = two_inputs

    def predict(self, x):
        predictions = []
        with torch.no_grad():
            for inputs in x:
                inputs = inputs.to(self.device)

                outputs = self.model(inputs)
                predictions.extend(outputs)
        return np.asarray(predictions).reshape(-1)

    def evaluate(self):
        with torch.no_grad():
            valid_loss = 0
            valid_acc = [0 for _ in range(len(self.eval_fns))]
            for data in self.valid_set:
                if self.two_inputs:
                    inputs, sizes, labels = data
                    inputs, sizes, labels = inputs.to(self.device), sizes.to(self.device), labels.to(self.device)
                    inputs = inputs, sizes
                else:
                    inputs, labels = data
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                for i, eval_fn in enumerate(self.eval_fns):
                    valid_acc[i] += eval_fn(outputs, labels)
                valid_loss += self.loss_fn(outputs, labels).item()
            return valid_loss / len(self.valid_set), [v / len(self.valid_set) for v in valid_acc]

    def evaluate_couple(self):
        with torch.no_grad():
            valid_loss = 0
            valid_acc = [0 for _ in range(len(self.eval_fns))]
            for data in self.valid_set:
                if self.two_inputs:
                    inputs, sizes, labels = data
                    inputs, sizes, labels = inputs.to(self.device), sizes.to(self.device), labels.to(self.device)
                    inputs = inputs, sizes
                else:
                    inputs, labels = data
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    inputs_right = deepcopy(inputs)
                    inputs_right = keep_right_seat_torch(inverse_keep_left_seat_torch(inputs_right))

                outputs_left = self.model(inputs)

                outputs_right = self.model(inputs_right)
                room_size = int(np.sqrt(outputs_right.shape[1]))
                outputs_right = outputs_right.view(-1, room_size, room_size)
                to_add = torch.zeros(outputs_right.shape[0], room_size, 1).to(device)
                outputs_right = torch.cat(
                    (outputs_right[:, :, 1:], to_add), 2)
                outputs_right = outputs_right.view(-1, room_size * room_size)

                outputs = 0.5 * (outputs_left + outputs_right)
                for i, eval_fn in enumerate(self.eval_fns):
                    valid_acc[i] += eval_fn(outputs, labels)

                valid_loss += self.loss_fn(outputs, labels).item()
            return valid_loss / len(self.valid_set), [v / len(self.valid_set) for v in valid_acc]

    def train(self, patience=5, max_it=100, verbose=True):

        counter, train_loss = 0, None
        best_val_acc = [-1, -1, -1]
        history = [[], [], [], []]

        if verbose:
            print("{:5s} | {:10s} | {:10s} | {:6s} | {:6s} | {:6s}".format(
                "epoch", "train_loss", "valid_loss", "Acc1", "Acc2", "Acc3"))
        for epoch in range(max_it):
            running_loss = 0
            # early stopping
            counter += 1
            if counter > patience - 1:
                break
            for i, data in enumerate(self.train_set, 1):
                # get the inputs
                if self.two_inputs:
                    inputs, sizes, labels = data
                    inputs, sizes, labels = inputs.to(self.device), sizes.to(self.device), labels.to(self.device)
                    inputs = inputs, sizes
                else:
                    inputs, labels = data
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model.forward(inputs)
                train_loss = self.loss_fn(outputs, labels)
                running_loss += train_loss.item()
                train_loss.backward()
                self.optimizer.step()

            val_loss, val_acc = self.evaluate()

            history[0].append(running_loss / len(self.train_set))
            history[1].append(val_loss)
            history[2].append(val_acc)

            # For updating the value of early stopping, we first look at the first accuracy,
            # then at the second in an equality case, and then at the third if another equality.
            is_better = val_acc[0] > best_val_acc[0] or \
                        (val_acc[0] == best_val_acc[0] and val_acc[1] > best_val_acc[1]) or \
                        (val_acc[0] == best_val_acc[0] and val_acc[1] == best_val_acc[1]
                         and val_acc[2] > best_val_acc[2])

            if verbose:
                # same things but I want to print the value of alpha :
                if self.two_inputs:
                    a = self.model.alpha.item()
                    print(
                        "{:5d} | {:10.5f} | {:10.5f} | {:5.2%} | {:5.2%} | {:5.2%}| {:5.2f} ".format(
                            epoch, running_loss / len(self.train_set),
                            val_loss, val_acc[0], val_acc[1], val_acc[2],
                            a),
                        end="")
                    if is_better:
                        print("\tsaved!", end="")
                    print("")
                else:
                    print(
                        "{:5d} | {:10.5f} | {:10.5f} | {:5.2%} | {:5.2%} | {:5.2%}".format(
                            epoch, running_loss / len(self.train_set),
                            val_loss, val_acc[0], val_acc[1], val_acc[2]),
                        end="")
                    if is_better:
                        print("\tsaved!", end="")
                    print("")

            if is_better:
                counter = 0
                best_val_acc[0] = val_acc[0]
                best_val_acc[1] = val_acc[1]
                best_val_acc[2] = val_acc[2]
                torch.save(self.model.state_dict(), PATH_BEST_MODEL + self.model.name)

        self.model.load_state_dict(torch.load(PATH_BEST_MODEL + self.model.name))
        return history

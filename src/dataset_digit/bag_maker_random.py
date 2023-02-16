import numpy as np
import torch
import torch.utils.data as data_utils
from torchvision.datasets import MNIST, SVHN


class BagMaker(data_utils.Dataset):
    def __init__(self, positive_number, negative_number, mean_bag_length, var_bag_length, seed):
        self.positive_number = positive_number
        self.negative_number = negative_number
        self.mean_bag_length = mean_bag_length
        self.var_bag_length = var_bag_length
        self.r = np.random.RandomState(seed)

    @staticmethod
    def load_mnist(train):
        numbers, labels = [], []
        mnist_data = MNIST("MNIST", train=train, download=True)
        for img, label in mnist_data:
            numbers.append(np.array(img))
            labels.append(label)

        numbers = np.asarray(numbers)
        numbers = np.tile(numbers, (3,1,1,1))
        numbers = numbers.transpose(1, 0, 2, 3).astype(np.float32)
        labels = np.asarray(labels)

        return numbers, labels

    @staticmethod
    def load_svhn(train):
        numbers, labels = [], []
        split = "train" if train else "test"
        svhn_data = SVHN("SVHN", split=split, download=True)
        for img, label in svhn_data:
            img = np.array(img)[2:30, 2:30, :]
            numbers.append(img)
            labels.append(label)

        numbers = np.asarray(numbers)
        numbers = numbers.transpose(0, 3, 1, 2).astype(np.float32)
        labels = np.asarray(labels)

        return numbers, labels

    def random_bags_form(self, dataset_name, nums_bag, train):
        if dataset_name == 'mnist':
            numbers, labels = self.load_mnist(train)
        elif dataset_name == 'svhn':
            numbers, labels = self.load_svhn(train)
        else:
            raise ValueError(f"Invalid dataset {dataset_name}")

        data = torch.tensor(numbers)
        labels = torch.tensor(labels)

        data_pos = data[labels == self.positive_number]
        data_neg = data[labels == self.negative_number] if self.negative_number != -1 else data[labels != self.positive_number]
        print(f"{dataset_name}_{'train' if train else 'test'}-> all data: {len(data)}, positive: {len(data_pos)}, negative: {len(data_neg)}")
        data_pos, data_neg = self.r.permutation(data_pos), self.r.permutation(data_neg)
        # print(data_pos.shape, data_neg.shape)

        current_bag_label = 0
        pos_idx = neg_idx = 0
        valid_data_list, valid_label_list = [], []

        for i in range(nums_bag):
            bag_length = max(int(self.r.normal(self.mean_bag_length, self.var_bag_length, 1)), 1)

            if current_bag_label == 0:
                curent_bag = data_neg[neg_idx:neg_idx + bag_length]
                neg_idx += bag_length
                valid_data_list.append(curent_bag)
                valid_label_list.append(torch.tensor([0] * bag_length))
            else:
                # pos_length = self.r.randint(1, bag_length + 1)
                pos_mean, pos_var = self.mean_bag_length / 10, self.mean_bag_length / 10
                pos_length = min(max(int(self.r.normal(pos_mean, pos_var, 1)), 1), bag_length)
                neg_length = bag_length - pos_length

                current_bag = data_pos[pos_idx:pos_idx + pos_length]
                current_label = np.ones(pos_length)
                pos_idx += pos_length

                # current_bag = torch.cat((current_bag, data_neg[neg_idx:neg_idx + neg_length]))
                current_bag = np.append(current_bag, data_neg[neg_idx:neg_idx + neg_length], axis=0)
                current_bag = torch.tensor(current_bag)
                current_label = np.append(current_label, np.zeros(neg_length))
                current_label = torch.tensor(current_label)
                neg_idx += neg_length

                perm = list(np.random.permutation(bag_length))
                current_bag, current_label = current_bag[perm], current_label[perm]

                valid_data_list.append(current_bag)
                valid_label_list.append(current_label)

            current_bag_label ^= 1

        print('{}_{} pos: {}, neg: {}'.format(dataset_name, "train" if train else "test", pos_idx, neg_idx))
        # print(len(valid_data_list), len(valid_label_list))

        return valid_data_list, valid_label_list

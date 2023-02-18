import numpy as np
import torch
import torch.utils.data as data_utils
from torchvision.datasets import MNIST, SVHN
from utils import logger


class BagMaker(data_utils.Dataset):
    def __init__(self, positive_number, negative_number, bag_size_mean, bag_size_var, seed):
        self.positive_number = positive_number
        self.negative_number = negative_number
        self.bag_size_mean = bag_size_mean
        self.bag_size_var = bag_size_var
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
            data, labels = self.load_mnist(train)
        elif dataset_name == 'svhn':
            data, labels = self.load_svhn(train)
        else:
            raise ValueError(f"Invalid dataset {dataset_name}")

        data_pos = data[labels == self.positive_number]
        if self.negative_number != -1:
            data_neg = data[labels == self.negative_number]
        else:
            data_neg = data[labels != self.positive_number]
        data_pos, data_neg = self.r.permutation(data_pos), self.r.permutation(data_neg)
        logger.info(
            f"{dataset_name}_{'train' if train else 'test'}"
            f"-> all data: {len(data)}, positive: {len(data_pos)}, negative: {len(data_neg)}")

        pos_bag_num = nums_bag // 2
        neg_bag_num = nums_bag - pos_bag_num
        pos_idx = neg_idx = 0
        bag_list, label_list = [], []

        for _ in range(neg_bag_num):
            bag_length = max(int(self.r.normal(self.bag_size_mean, self.bag_size_var, 1)), 1)

            current_bag = data_neg[neg_idx:neg_idx + bag_length]
            neg_idx += bag_length
            bag_list.append(torch.tensor(current_bag))
            label_list.append(torch.tensor([0] * bag_length))

        for _ in range(pos_bag_num):
            bag_length = max(int(self.r.normal(self.bag_size_mean, self.bag_size_var, 1)), 1)

            pos_mean, pos_var = self.bag_size_mean / 10, self.bag_size_var / 10
            pos_length = min(max(int(self.r.normal(pos_mean, pos_var, 1)), 1), bag_length)
            neg_length = bag_length - pos_length

            pos_bag = data_pos[pos_idx: pos_idx + pos_length]
            neg_bag = data_neg[neg_idx: neg_idx + neg_length]
            current_bag = np.append(pos_bag, neg_bag, axis=0)
            current_label = np.append(np.ones(pos_length), np.zeros(neg_length))

            perm = self.r.permutation(bag_length)
            current_bag, current_label = current_bag[perm], current_label[perm]
            bag_list.append(torch.tensor(current_bag))
            label_list.append(torch.tensor(current_label))
            pos_idx += pos_length
            neg_idx += neg_length

        logger.info(f"{dataset_name}_{'train' if train else 'test'} -> pos: {pos_idx}, neg: {neg_idx}")

        return bag_list, label_list

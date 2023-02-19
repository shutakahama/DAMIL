import torch
import torch.utils.data as data_utils
from sklearn.model_selection import train_test_split
from .bag_maker import BagMaker


class DigitBagLoader(data_utils.Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        self.indexes = []

        count = 0
        for i in range(len(images)):
            current_index = torch.arange(len(images[i])) + count
            count += len(images[i])
            self.indexes.append(current_index)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        bag = self.images[index]
        label = [max(self.labels[index]), self.labels[index]]
        idx = self.indexes[index]
        return bag, label, idx


class DigitDataFactory:
    def __init__(
        self, positive_number, negative_number, bag_size_mean, bag_size_var, valid_rate, seed):

        self.valid_rate = valid_rate
        self.bag_maker = BagMaker(
            positive_number, negative_number, bag_size_mean, bag_size_var, seed)
        self.loader_kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}

    def get_train_data_loader(self, dataset_name, nums_bag):
        images, labels = self.bag_maker.random_bags_form(dataset_name, nums_bag, train=True)
        bag_labels = [max(_l) for _l in labels]
        train_images, valid_images, train_labels, valid_labels = train_test_split(
            images, labels, test_size=self.valid_rate, stratify=bag_labels)
        train_data_loader = data_utils.DataLoader(
            DigitBagLoader(train_images, train_labels),
            batch_size=1,
            shuffle=True,
            **self.loader_kwargs)
        valid_data_loader = data_utils.DataLoader(
            DigitBagLoader(valid_images, valid_labels),
            batch_size=1,
            shuffle=True,
            **self.loader_kwargs)

        return train_data_loader, valid_data_loader

    def get_test_data_loader(self, dataset_name, nums_bag):
        images, labels = self.bag_maker.random_bags_form(dataset_name, nums_bag, train=False)
        data_loader = data_utils.DataLoader(
            DigitBagLoader(images, labels),
            batch_size=1,
            shuffle=False,
            **self.loader_kwargs)

        return data_loader

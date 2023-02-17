import torch
import torch.utils.data as data_utils
from .bag_maker_random import BagMaker


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
    def __init__(self, positive_number, negative_number, batch_size, mean_bag_length, var_bag_length, seed):
        self.batch_size = batch_size
        self.bag_maker = BagMaker(positive_number, negative_number, mean_bag_length, var_bag_length, seed)
        self.loader_kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}

    def get_data_loader(self, dataset_name, nums_bag, train=True):
        images, labels = self.bag_maker.random_bags_form(dataset_name, nums_bag, train)
        data_loader = data_utils.DataLoader(
            DigitBagLoader(images, labels), batch_size=self.batch_size, shuffle=train, **self.loader_kwargs)

        return data_loader

import torch
import torch.utils.data as data_utils
from .bag_maker_random import BagMaker


class DigitBagLoader(data_utils.Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        bag = self.images[index]
        label = [max(self.labels[index]), self.labels[index]]
        return bag, label


class DigitBagLoaderIndex(DigitBagLoader):
    def __init__(self, images, labels):
        super(DigitBagLoaderIndex, self).__init__(images, labels)
        count = 0
        self.indexes = []
        for i in range(len(images)):
            current_index = torch.arange(len(images[i])) + count
            count += len(images[i])
            self.indexes.append(current_index)

    def __getitem__(self, index):
        bag = self.images[index]
        label = [max(self.labels[index]), self.labels[index]]
        idx = self.indexes[index]
        return bag, label, idx


class DigitDataLoader:
    def __init__(self, positive_number, negative_number, batch_size, mean_bag_length, var_bag_length, num_class, seed):
        self.batch_size = batch_size
        self.num_class = num_class
        self.bag_maker = BagMaker(positive_number, negative_number, mean_bag_length, var_bag_length, seed)
        self.loader_kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}

    def data_to_dataloader(self, images, labels, train=True, dataloader_type="single"):
        if dataloader_type == "single":
            data_loader = data_utils.DataLoader(DigitBagLoader(images, labels),
                                                batch_size=self.batch_size,
                                                shuffle=train,
                                                **self.loader_kwargs)
        elif dataloader_type == "index":
            data_loader = data_utils.DataLoader(DigitBagLoaderIndex(images, labels),
                                                batch_size=self.batch_size,
                                                shuffle=train,
                                                **self.loader_kwargs)
        else:
            raise NameError

        return data_loader

    def get_data_bags(self, dataset_name, nums_bag, train=True):
        images, labels = self.bag_maker.random_bags_form(dataset_name, nums_bag, train)
        return images, labels

    def get_data_loader(self, dataset_name, nums_bag, train=True, dataloader_type="single"):
        images, labels = self.get_data_bags(dataset_name, nums_bag, train)
        dataloader = self.data_to_dataloader(images, labels, train, dataloader_type)

        return dataloader


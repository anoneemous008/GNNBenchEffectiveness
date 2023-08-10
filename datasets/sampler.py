import torch
from torch.utils.data import sampler

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
	
class ImbalancedDatasetSampler(WeightedRandomSampler):
    def __init__(self, dataset, num_samples=None):
        self.num_samples = len(dataset) if num_samples is None else num_samples

        # 计算每个样本的权重
        targets = [d.y for d in dataset]
        print('tragets len"', len(targets))
        print('tragets 0: ', targets[0])
        class_sample_count = torch.unique(torch.tensor(targets), return_counts=True)[1]
        weight = 1. / class_sample_count.float()
        # need normalize.
        print('weight: ', weight)
        samples_weight = weight[targets]
        self.samples_weight = torch.FloatTensor(samples_weight)

        super().__init__(self.samples_weight, self.num_samples, replacement=True)


class RandomSampler(sampler.RandomSampler):
    """
    This sampler saves the random permutation applied to the training data,
    so it is available for further use (e.g. for saving).
    The permutation is saved in the 'permutation' attribute.
    The DataLoader can now be instantiated as follows:

    >>> data = Dataset()
    >>> dataloader = DataLoader(dataset=data, batch_size=32, shuffle=False, sampler=RandomSampler(data))
    >>> for batch in dataloader:
    >>>     print(batch)
    >>> print(dataloader.sampler.permutation)

    For convenience, one can create a method in the dataloader class to access the random permutation directly, e.g:

    class MyDataLoader(DataLoader):
        ...
        def get_permutation(self):
            return self.sampler.permutation
        ...
    """

    def __init__(self, data_source, num_samples=None, replacement=False):
        super().__init__(data_source, replacement=replacement, num_samples=num_samples)
        self.permutation = None

    def __iter__(self):
        n = len(self.data_source)
        self.permutation = torch.randperm(n).tolist()
        return iter(self.permutation)

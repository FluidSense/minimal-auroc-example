import torch

from pandas import DataFrame, Series
from torch.utils.data import Dataset
from typing import Tuple

from normalized_labels import Labels

class STLFakeDataset(Dataset):
    def __init__(self, dataset: DataFrame):
        predictors, targets = self.split_predictor(dataset)
        self.predictors = torch.tensor(predictors.values, dtype=torch.float)
        self.targets = targets

    def __len__(self):
        return len(self.predictors)

    def __getitem__(self, index):
        return (self.predictors[index], self.targets[index].unsqueeze(0))

    def split_predictor(self, data: DataFrame) -> Tuple[DataFrame, Series]:
        predictions = self.tensor_predictor(data.pop("Targets").to_list())
        return data, predictions

    def tensor_predictor(self, data: Series) -> torch.Tensor:
        data = torch.tensor(data, dtype=torch.float)
        return data


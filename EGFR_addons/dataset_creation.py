import torch
from torch_geometric.data import InMemoryDataset, Data
import pandas as pd
from ogb.utils.mol import smiles2graph


class EGFR_Dataset_Mem(InMemoryDataset):
    """Generate custom dataset of EGFR compounds following OGB standards

    Expect EGFR_compounds_lipinsky.csv file saved in datasets/EGFR/raw
    Outputs EGFR_compounds_lipinsky.pt to datasets/EGFR/processed

    """

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "EGFR_compounds_lipinsky.csv"

    @property
    def processed_file_names(self):
        return "EGFR_compounds_lipinsky.pt"

    def download(self):
        pass

    def process(self):
        self.dataframe = pd.read_csv(self.raw_paths[0], index_col=0)

        data_list = []

        for i in range(self.dataframe.shape[0]):
            mol_smile = self.dataframe.loc[self.dataframe.index[i], "smiles"]
            graph = smiles2graph(mol_smile)
            pyg_graph = Data(
                x=torch.tensor(graph["node_feat"], dtype=torch.long),
                edge_index=torch.tensor(graph["edge_index"], dtype=torch.long),
                edge_attr=torch.tensor(graph["edge_feat"], dtype=torch.long),
                y=self.dataframe.loc[self.dataframe.index[i], "pIC50"],
            )
            data_list.append(pyg_graph)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


if __name__ == "__main__":
    EGFR_Dataset_Mem("../datasets/EGFR")
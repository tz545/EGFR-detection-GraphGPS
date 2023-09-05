import pytest 
import torch

from dataset_creation import EGFR_Dataset_Mem


@pytest.fixture
def dataset():
    return EGFR_Dataset_Mem("../datasets/EGFR")

def test_atom_types_allowed(dataset):
    """ 
    Atoms are encoded by 9 values, representing atom type, charge, etc. 
    The values must be respectively less than: 119, 4, 12, 12, 10, 6, 6, 2, 2 to fit into the embeddings for AtomEncoder

    """
    max_atom_nos = [0] * 9 
    for i in range(len(dataset)):
        assert dataset[i].x.shape[1] == 9
        for j in range(9):
            atom_no = max(dataset[i].x[:,j])
            max_atom_nos[j] = max(atom_no, max_atom_nos[j])

    assert max_atom_nos[0] < 119
    assert max_atom_nos[1] < 4
    assert max_atom_nos[2] < 12
    assert max_atom_nos[3] < 12
    assert max_atom_nos[4] < 10
    assert max_atom_nos[5] < 6
    assert max_atom_nos[6] < 6
    assert max_atom_nos[7] < 2 
    assert max_atom_nos[8] < 2


def test_bond_types_allowed(dataset):
    """
    Bonds are encoded by 3 values, representing bond type, chirality, etc. 
    The values must be respectively less than: 5, 6, 2 to fit into the embeddings for BondEncoder

    """
    max_edges = [0] * 3
    for i in range(len(dataset)):
        ## it is possible for the edge attributes tensor to be empty
        if dataset[i].edge_attr.numel() != 0: 
            assert dataset[i].edge_attr.shape[1] == 3
            for j in range(3):
                edge = max(dataset[i].edge_attr[:,j])
                max_edges[j] = max(edge, max_edges[j])

    assert max_edges[0] < 5
    assert max_edges[1] < 6
    assert max_edges[2] < 2


def test_y_size_and_dtype(dataset):
    for i in range(len(dataset)):
        y = dataset[i].y
        assert y.dtype == torch.float64
        assert y.shape == torch.Size([1])
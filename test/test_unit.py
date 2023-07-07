import os

import pandas as pd
import pytest


def test_data_preparation():
    assert os.path.isfile("../datasets/housing/housing.csv")


def test_load_data():
    try:
        df = pd.read_csv(r"../datasets/housing/housing.csv")
        df.shape
    except:
        assert False


def test_artifact_dir():
    assert os.path.isdir("../artifacts/")


def test_pickle_file():
    assert os.path.isfile("../artifacts/lin_reg_op.pkl")

from ch02.fetch_data import fetch_housing_data, load_housing_data
from ch02.look_at_data import take_quick_look_at_data, show_housing_hist
from ch02.test_set import split_train_test, split_train_test_by_id
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import numpy as np
import pandas as pd


def run_ch02():
    fetch_housing_data()
    housing = load_housing_data()
    # take_quick_look_at_data(housing)
    # show_housing_hist(housing)
    # train_set, test_set = split_train_test(housing, 0.2)
    # print(len(train_set))
    # print(len(test_set))
    # housing_with_id = housing.reset_index()
    # train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")
    # housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
    # train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")
    # train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
    # print(len(train_set))
    # print(len(test_set))
    housing["income_cat"] = pd.cut(housing["median_income"], bins=[0.,1.5,3.0,4.5, 6., np.inf], labels=[1,2,3,4,5])
    # housing["income_cat"].hist()
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    # strata_ratio = strat_test_set["income_cat"].value_counts() / len(strat_test_set)
    # print(strata_ratio)
    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)






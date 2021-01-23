from ch02.fetch_data import fetch_housing_data, load_housing_data
from ch02.look_at_data import take_quick_look_at_data, show_housing_hist
from ch02.test_set import split_train_test, split_train_test_by_id


def run_ch02():
    fetch_housing_data()
    housing = load_housing_data()
    # take_quick_look_at_data(housing)
    # show_housing_hist(housing)
    # train_set, test_set = split_train_test(housing, 0.2)
    # print(len(train_set))
    # print(len(test_set))
    housing_with_id = housing.reset_index()
    # train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")
    housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
    train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")
    print(len(train_set))
    print(len(test_set))


from ch02.fetch_data import load_housing_data


def take_quick_look_at_data():
    housing = load_housing_data()
    print("looking at top rows of the data")
    print(housing.head())
    print("get quick description of the data")
    print(housing.info())

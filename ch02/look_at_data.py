from ch02.fetch_data import load_housing_data


def take_quick_look_at_data():
    housing = load_housing_data()
    print(housing.head())

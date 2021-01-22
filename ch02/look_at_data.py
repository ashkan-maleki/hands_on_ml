from ch02.fetch_data import load_housing_data


def take_quick_look_at_data():
    housing = load_housing_data()
    print("looking at top rows of the data")
    print(housing.head())
    print("get quick description of the data")
    print(housing.info())
    print('ocean_proximity categories')
    print(housing["ocean_proximity"].value_counts())
    print("let's look at numerical values")
    print(housing.describe())

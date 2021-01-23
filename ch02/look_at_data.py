import matplotlib.pyplot as plt


def take_quick_look_at_data(housing):
    print("looking at top rows of the data")
    print(housing.head())
    print("get quick description of the data")
    print(housing.info())
    print('ocean_proximity categories')
    print(housing["ocean_proximity"].value_counts())
    print("let's look at numerical values")
    print(housing.describe())


def show_housing_hist(housing):
    housing.hist(bins=50, figsize=(20, 15))
    plt.show()

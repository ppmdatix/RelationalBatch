import pandas as pd
import csv
import sys
from one_hot_encode import one_hot_encode


directory = sys.argv[1]
dataset = sys.argv[2]

if dataset.lower() == "kdd":
    path = directory + "fetch_kddcup99.csv"
    target = "labels"
elif dataset.lower() == "forest_cover":
    path = directory + "forest_cover.csv"
    target = "Cover_Type"
elif dataset.lower() == "adult_income":
    path = directory + "adult.csv"
    target = "income"
elif dataset.lower() == "dont_get_kicked":
    path = directory + "dontgetkicked.csv"
    target = "IsBadBuy"
elif dataset.lower() == "used_cars":
    path = directory + "cars.csv"
    target = "price_usd"
elif dataset.lower() == "compas":
    path = directory + "compas-scores-two-years.csv"
    target = "is_recid"

else:
    raise Exception('no such dataset')
category_features_file = "categorical_features.csv"


with open(directory + category_features_file, newline='') as f:
    reader = csv.reader(f)
    cat_feat = list(reader)[0]

df = pd.read_csv(path)
df["target"] = df[target]
df = df.drop(target, axis=1)
oldNames = df.columns
output = df.target.values
labels = set(output)

for c in df.columns:
    if (not c in cat_feat) and (c != "target"):
        df = df.drop(c, axis=1)
print(dataset)
for col in df.columns:
    if col != "target":
        df = one_hot_encode(df, col)
        df = df.drop(col, axis=1)
print('###########')

df["target"] = df["target"].apply(lambda x: str(x))

# df.to_csv(directory + "training_processed.csv", index=False)



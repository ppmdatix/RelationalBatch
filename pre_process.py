import pandas as pd
import csv
import sys
from one_hot_encode import one_hot_encode
from data import data as dta


directory = sys.argv[1]
dataset = sys.argv[2].lower()


path    = directory + dta.sourceFiles[dataset]
target  = dta.sourceTargets[dataset]


with open(directory + dta.category_features_file, newline='') as f:
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

df["target"] = df["target"].apply(lambda x: str(x))

df.to_csv(directory + dta.output_file, index=False)



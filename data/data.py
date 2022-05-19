import torch

device = torch.device('cpu')

datasets = ["kdd", "forest_cover", "adult_income", "dont_get_kicked", "used_cars", "compas"]
colors   = ['pink', 'lightblue', 'silver', 'bisque', 'fushia', 'crimson']


task_types = {
                "kdd":             "multiclass",
                "forest_cover":    "multiclass",
                "adult_income":    "binclass",
                "dont_get_kicked": "binclass",
                "used_cars":       "regression",
                "compas":          "binclass" }


folderName = {
                "kdd":             "KDD99",
                "forest_cover":    "Forest_Cover",
                "adult_income":    "Adult_Income",
                "dont_get_kicked": "Dont_Get_Kicked",
                "used_cars":       "Usedcarscatalog",
                "compas":          "compas" }

sourceFiles = {
                "kdd":             "fetch_kddcup99.csv",
                "forest_cover":    "forest_cover.csv",
                "adult_income":    "adult.csv",
                "dont_get_kicked": "dontgetkicked.csv",
                "used_cars":       "cars.csv",
                "compas":          "compas-scores-two-years.csv" }


sourceTargets = {
                "kdd":             "labels",
                "forest_cover":    "Cover_Type",
                "adult_income":    "income",
                "dont_get_kicked": "IsBadBuy",
                "used_cars":       "price_usd",
                "compas":          "is_recid" }


targets = {
                "kdd":             "labels",
                "forest_cover":    "Cover_Type",
                "adult_income":    "target",
                "dont_get_kicked": "target",
                "used_cars":       "price_usd",
                "compas":          "is_recid" }


models = ["mlp", "resnet"]
optims = ["SGD", "adam", "adagrad" ]
batch_sizes = [64,512, 1024]
epochs = 10
reproduction = 10
png_prefix = "withOptimoo"
category_features_file = "categorical_features.csv"
output_file = "training_processed.csv"
nrows = None

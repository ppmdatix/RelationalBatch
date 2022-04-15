datasets = ["kdd", "forest_cover", "adult_income", "dont_get_kicked", "used_cars", "compas"]
colors = ['pink', 'lightblue', 'silver', 'bisque', 'fushia', 'crimson']


task_types = {
                "kdd":             "muliclass",
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


targets = {
                "kdd":             "labels",
                "forest_cover":    "Cover_Type",
                "adult_income":    "target",
                "dont_get_kicked": "target",
                "used_cars":       "price_usd",
                "compas":          "is_recid" }




models = ["mlp", "resnet"]
optims = ["sparse_adam" ]# , "SGD", "adam", "adagrad" ]
batch_sizes = [1, 8,32, 128]
epochs = 10
reproduction = 10

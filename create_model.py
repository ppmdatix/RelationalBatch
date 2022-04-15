import rtdl
import torch
import torch.nn.functional as F
import rtdl

device = torch.device('cpu')


def create_model(
        X_all,
        n_classes=None,
        task_type="regression",
        model_name="mlp",
        optim="adam",
        lr=0.001,
        weight_decay=0.0,
        first_layer=4,
        middle_layers=8,
        dropout=0.1):

    if task_type == "multiclass":
        d_out = n_classes
    else:
        d_out = n_classes or 1

    if model_name == "mlp":
        _model = rtdl.MLP.make_baseline(
            d_in=X_all.shape[1],
            d_layers=[first_layer, middle_layers, first_layer],
            dropout=dropout,
            d_out=d_out
        )
    elif model_name == "resnet":
        _model = rtdl.ResNet.make_baseline(
            d_in=X_all.shape[1],
            d_main=128,
            d_hidden=256,
            dropout_first=0.2,
            dropout_second=0.0,
            n_blocks=2,
            d_out=d_out,)
    elif model_name == "transformer":
        _model = rtdl.FTTransformer.make_default(
            n_num_features=X_all.shape[1],
            cat_cardinalities=None,
            last_layer_query_idx=[-1],  # it makes the model faster and does NOT affect its output
            d_out=d_out,
        )

    _model.to(device)

    optimizer = None

    if optim.lower() == "adam":
        optimizer = torch.optim.Adam(_model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optim.lower() == "adagrad":
        optimizer = torch.optim.Adagrad(_model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optim.lower() == "sgd":
        optimizer = torch.optim.SGD(_model.parameters(), lr=lr, momentum=0, dampening=0, weight_decay=0, nesterov=False)
    elif optim.lower() == "lbfgs":
        optimizer = torch.optim.LBFGS(_model.parameters(), lr=lr)
    elif optim.lower() == "rmsprop":
        optimizer = torch.optim.RMSprop(_model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optim.lower() == "sparse_adam":
        optimizer = torch.optim.SparseAdam(list(_model.parameters()), lr=lr)

    loss_fn = (
        F.mse_loss
        if task_type == 'binclass'
        else F.cross_entropy
        if task_type == 'multiclass'
        else F.mse_loss
    )
    return _model, optimizer, loss_fn

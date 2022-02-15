import rtdl
import torch
import torch.nn.functional as F

device = torch.device('cpu')


def create_model(X_all, n_classes=None, task_type="regression", model_name="mlp", optim="adam", x=None, y=None):
    if task_type == "multiclass":
        d_out = n_classes
    else:
        d_out = n_classes or 1

    lr = 0.001
    weight_decay = 0.0

    first_layer = 4
    if model_name == "mlp":
        _model = rtdl.MLP.make_baseline(
            d_in=X_all.shape[1],
        #     d_layers=[first_layer, 256, 128],
            d_layers=[first_layer, 8, first_layer],
            dropout=0.1,
            d_out=d_out,
            # seed=42
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
    elif optim.lower() == "lbfgs":
        optimizer = torch.optim.LBFGS(_model.parameters(), lr=lr)
    elif optim.lower() == "rmsprop":
        optimizer = torch.optim.RMSprop(_model.parameters(), lr=lr, weight_decay=weight_decay)



    #optimizer = (
    #    _model.make_default_optimizer()
    #    if isinstance(_model, rtdl.FTTransformer)
    #     else torch.optim.AdamW(_model.parameters(), lr=lr, weight_decay=weight_decay)
    #    else torch.optim.Adam(_model.parameters(), lr=lr, weight_decay=weight_decay)
    #)

    loss_fn = (
        F.mse_loss
        # F.binary_cross_entropy_with_logits
        if task_type == 'binclass'
        else F.cross_entropy
        if task_type == 'multiclass'
        else F.mse_loss
    )

    def closure():
        y_pred = _model(x)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        return loss

    optimizer.step(closure)


    return _model, optimizer, loss_fn

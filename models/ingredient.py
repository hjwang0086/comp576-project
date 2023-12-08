from .matcher import MatchERT
from sacred import Ingredient
model_ingredient = Ingredient('model')


@model_ingredient.config
def config():
    name = None
    num_global_features = 2048
    num_local_features = 128
    seq_len = None
    dim_K = None
    dim_feedforward = None
    nhead = None
    num_encoder_layers = None
    dropout = 0.0
    activation = "relu"
    normalize_before = False
    use_bottleneck = False # see experiment.py
    use_duplicate = False # see experiment.py

@model_ingredient.named_config
def RRT():
    name = 'rrt'
    seq_len = 1004
    dim_K = 256
    dim_feedforward = 1024
    nhead = 4
    num_encoder_layers = 6
    dropout = 0.0 
    activation = "relu"
    normalize_before = False

@model_ingredient.named_config
def RRT_h8():
    name = 'rrt'
    seq_len = 1004
    dim_K = 256
    dim_feedforward = 1024
    nhead = 8
    num_encoder_layers = 6
    dropout = 0.0 
    activation = "relu"
    normalize_before = False

@model_ingredient.named_config
def RRT_h8ext():
    name = 'rrt'
    seq_len = 1004
    dim_K = 256
    num_local_features = 256
    dim_feedforward = 1024
    nhead = 8
    num_encoder_layers = 6
    dropout = 0.0 
    activation = "relu"
    normalize_before = False

@model_ingredient.named_config
def RRT_h16():
    name = 'rrt'
    seq_len = 1004
    dim_K = 256
    dim_feedforward = 1024
    nhead = 16
    num_encoder_layers = 6
    dropout = 0.0 
    activation = "relu"
    normalize_before = False

@model_ingredient.named_config
def RRT_h32():
    name = 'rrt'
    seq_len = 1004
    dim_K = 256
    dim_feedforward = 1024
    nhead = 32
    num_encoder_layers = 6
    dropout = 0.0 
    activation = "relu"
    normalize_before = False

@model_ingredient.named_config
def RRT_h64():
    name = 'rrt'
    seq_len = 1004
    dim_K = 256
    dim_feedforward = 1024
    nhead = 64
    num_encoder_layers = 6
    dropout = 0.0 
    activation = "relu"
    normalize_before = False

@model_ingredient.named_config
def RRT_h2():
    name = 'rrt'
    seq_len = 1004
    dim_K = 256
    dim_feedforward = 1024
    nhead = 2
    num_encoder_layers = 6
    dropout = 0.0 
    activation = "relu"
    normalize_before = False

@model_ingredient.named_config
def RRT_L():
    name = 'rrt'
    seq_len = 1004
    dim_K = 256
    dim_feedforward = 1024
    nhead = 4
    num_encoder_layers = 8
    dropout = 0.0 
    activation = "relu"
    normalize_before = False

@model_ingredient.named_config
def RRT_S():
    name = 'rrt'
    seq_len = 1004
    dim_K = 256
    dim_feedforward = 1024
    nhead = 4
    num_encoder_layers = 4
    dropout = 0.0 
    activation = "relu"
    normalize_before = False

@model_ingredient.named_config
def RRT_T():
    name = 'rrt'
    seq_len = 1004
    dim_K = 256
    dim_feedforward = 1024
    nhead = 4
    num_encoder_layers = 2
    dropout = 0.0 
    activation = "relu"
    normalize_before = False


@model_ingredient.capture
def get_model(num_global_features, num_local_features, seq_len, dim_K, dim_feedforward, nhead, num_encoder_layers, dropout, 
                activation, normalize_before, use_bottleneck, use_duplicate):
    return MatchERT(d_global=num_global_features, d_model=num_local_features, seq_len=seq_len, d_K=dim_K, nhead=nhead, num_encoder_layers=num_encoder_layers, 
            dim_feedforward=dim_feedforward, dropout=dropout, activation=activation, normalize_before=normalize_before, 
                use_bottleneck=use_bottleneck, use_duplicate=use_duplicate)
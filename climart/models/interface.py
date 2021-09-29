from climart.models.GNs.graph_network import GN_Trainer, GraphNetwork, GN_withReadout
from climart.models.MLP import MLP_Trainer, MLPNet
from climart.models.CNN import CNN_Trainer, CNN_Net, CNN_Multiscale
from climart.models.base_model import BaseModel, BaseTrainer

def get_model(model_name: str, only_class: bool = False, *args, **kwargs) -> BaseModel:
    model_name = model_name.strip().lower()
    if model_name == 'mlp':
        model = MLPNet
    elif model_name in ['gcn', 'gnn']:
        from climart.models.GNNs.GCN import GCN
        model = GCN
    elif model_name in ['gcn+readout', 'gnn+readout']:
        from climart.models.GNNs.GCN import GCN_withReadout
        model = GCN_withReadout
    elif model_name in ['gn', 'graph_net']:
        model = GraphNetwork
    elif model_name in ['gn+readout', 'graph_net+readout']:
        model = GN_withReadout
    elif model_name == 'cnn':
        model = CNN_Net
    elif model_name == 'cnnms':
        model = CNN_Multiscale
    else:
        raise ValueError(f"Unknown model {model_name}")
    if only_class:
        return model
    if 'name' not in kwargs:
        kwargs['name'] = model_name
    return model(*args, **kwargs)


def get_trainer(model_name, *args, **kwargs) -> BaseTrainer:
    model_name = model_name.strip().lower()
    if model_name == 'mlp':
        trainer = MLP_Trainer
    elif is_gnn(model_name):
        from climart.models.GNNs.GCN import GCN_Trainer
        trainer = GCN_Trainer
    elif is_graph_net(model_name):
        trainer = GN_Trainer
    elif is_cnn(model_name):
        trainer = CNN_Trainer
    else:
        raise ValueError()
    if 'name' not in kwargs:
        kwargs['name'] = model_name
    return trainer(*args, **kwargs)


def is_gnn(name):
    return name.lower() in ['gcn', 'gnn',
                            'gcn+readout', 'gnn+readout']


def is_graph_net(name):
    return name.lower() in ['gn', 'graph_net', 'gn+readout', 'graph_net+readout']


def is_cnn(name):
    return name.lower() in ['cnn', 'cnnms']


def get_input_transform(model_class: BaseModel, batched: bool):
    if batched:
        return model_class._batched_input_transform
    else:
        return model_class._input_transform

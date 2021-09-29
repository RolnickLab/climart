def get_model_name(name: str) -> str:
    if 'CNN' in name:
        return 'CNN'
    elif 'MLP' in name:
        return 'MLP'
    elif 'GCN' in name:
        return 'GCN'
    elif 'GN' in name:
        return 'GraphNet'
    else:
        raise ValueError(name)
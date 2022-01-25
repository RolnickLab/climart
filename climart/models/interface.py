from climart.models.base_model import BaseModel



def get_input_transform(model_class: BaseModel, batched: bool):
    if batched:
        return model_class._batched_input_transform
    else:
        return model_class._input_transform

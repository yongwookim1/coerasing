from .Internlm import InternLM
model_dict = {
    'Internlm': InternLM,

}
def get_model(model_name, **kwargs):
    return model_dict[model_name](**kwargs)
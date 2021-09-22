from models.uvgg19 import uvgg19
from models.uvgg19_linear import uvgg19_linear

models_dict = {
    "uvgg19": uvgg19,
    "uvgg19_linear": uvgg19_linear
}


def get_models_dict():
    return models_dict

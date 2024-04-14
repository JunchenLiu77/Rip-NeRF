from .rf_model import RFModel
from .ripnerf_model import RipNerfModel


def get_model(model_name: str = 'Rip-NeRF') -> RFModel:
    if 'Rip-NeRF' == model_name:
        return RipNerfModel
    else:
        raise NotImplementedError

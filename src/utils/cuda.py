import torch


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('CUDA is available, using the following GPU: ', torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print('CUDA is not available. Using the CPU')
    return device

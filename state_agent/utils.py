import torch

def init_device(cuda_gpu = 0):
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print('using mps')
    elif torch.cuda.is_available():
        device = torch.device(f'cuda:{cuda_gpu}')
        print('using cuda')
    else:
        device = torch.device('cpu')
        print('using cpu')
    return device
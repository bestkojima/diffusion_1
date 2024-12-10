
from dataclasses import dataclass, field
@dataclass
class model_config():
    dim = 64
    dim_mults = (1, 2, 4, 8)
    channels=1

@dataclass
class mnist_config():
    img_size=(28,28)
    channels=1
    numsteps=1000


@dataclass
class denoise_mnist_train():
    time_steps: int = 50
    train_steps: int = 700000
    save_folder: str = './results_mnist'
    load_path: str = None
    data_path: str = '../root_mnist/'
    train_routine: str = 'Final'
    sampling_routine: str = 'default'
    remove_time_embed: bool = False
    residual: bool = False
    loss_type: str = 'l1'


if __name__ == "__main__":
    
    print(denoise_mnist_train.time_steps)
    print(type(c.dim_mults))
    print(type(c.channels))
from dataclasses import dataclass
from typing import Union
import torch


@dataclass
class Params:
    q: int = 10
    r: int = 1
    seed: int = 1234

    num_workers: int = 10
    num_episode: int = 1000
    batch_size: int = 2048
    mini_batch_size: int = 256
    num_epoch: int = 10
    num_test: int = 10

    lr_a: float = 5e-4
    lr_c: float = 5e-4
    gamma: float = 0.99
    gae_lam: float = 0.97
    clipped_value_loss_param: Union[float, None] = None
    loss_value_coef: float = 0.5
    loss_entropy_coef: float = 0.01
    target_kl: Union[float, None] = None
    clip: float = 0.2
    max_grad_norm: float = 0.5
    EPS = 1e-5

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    # Here, use the percentage to represent the noise gotten into the state. e.g. 0.05 <--> 5% noise in state  
    adversary_attack: bool = False
    adversary_attack_var: Union[float, None] = 0.1
    loss_upper_kl_coef: float = 0.05

    early_stop: bool = True
    early_stop_episode: int = 9999

    model_path = './model/'
    net_structure_log = False
    net_structure_logger_path = './network/'
    tensorboard_log = True
    tensorboard_logger_path = './tensorboard_results_1218/'
    save_data = True
    data_path = './noise_results_1218/'


@dataclass
class CSTR_Params(Params):
    '''
    input variables: 
        CAf:    concentration of A in feed stream 
        Tf:     temperature of feed stream 
        Tc:     temperation of jacket coolant (controlled variable)
    output variables: 
        CA:     concentration of A in the reactor 
        T:      temperature of stream in the reactor
    Others:
        F:      volumetric flow rate 
        V:      reator volume
        CA0:    inital concentration of A in reactor 
        Tc0:    inital temprature in the reactor
        CA_tar: target concentration of A in reactor 
    '''
    sample_time: float = 0.1
    F_bound: float = 0.1
    Tc_bound: float = 5

    F: float = 1.0  # control 
    V: float = 1.0
    CAf: float = 10.0
    Tf: float = 300.0
    CA0: float = 8.5698  
    T0: float = 311.2639
    Tc0: float = 292.0  # control 
    CA_tar: float = 7
    T_tar: float = 311.2639

    # F: float = 100.0  # control 
    # V: float = 100.0
    # CAf: float = 1
    # Tf: float = 350.0
    # CA0: float = 0.7 
    # T0: float = 338.5276
    # Tc0: float = 303.0438  # control 
    # CA_tar: float = 0.8
    # T_tar: float = T0


@dataclass 
class Weight_Params:
    qr_list = [
        # (10, 1), (5, 1), (1, 1), (1, 2)
        # (1, 1), (2, 1), (3, 1)
        (1, 1),(2, 1), (3, 1), (8, 1),
        # (5, 1)
    ]

    noise: Union[float, None] = 0.05
    base_agent_verbose: bool = True



from .dpfedavg import DPFedAvgClient
from .fedavg import FedAvgClient
from .fednova import FedNovaClient, fednova_aggregate
from .fedprox import FedProxClient
from .scaffold import SCAFFOLDClient, SCAFFOLDServer

REGISTRY = {
    "FedAvg": FedAvgClient,
    "FedProx": FedProxClient,
    "SCAFFOLD": SCAFFOLDClient,
    "FedNova": FedNovaClient,
    "DP-FedAvg": DPFedAvgClient,
}

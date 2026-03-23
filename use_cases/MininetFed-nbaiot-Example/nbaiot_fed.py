import os

from mininetfed.core.dto.metrics import MetricType
from mininetfed.core.fed_options import (
    ServerOptions,
    ClientAcceptorType,
    ClientSelectorType,
    AggregatorType,
)
from mininetfed.sim.net import MininetFed
from mininetfed.sim.nodes import FedServerNode, FedClientNode, FedBrokerNode
from mininetfed.sim.util.docker_utils import build_fed_node_docker_image


CLIENTS_ROOT = "clients"
CLIENT_SCRIPT = "nbaiot_trainer.py"
SERVER_SCRIPT = "nbaiot_server.py"


def topology():
    device_names = sorted([
        d for d in os.listdir(CLIENTS_ROOT)
        if os.path.isdir(os.path.join(CLIENTS_ROOT, d))
    ])

    if not device_names:
        raise RuntimeError("Nenhuma pasta de cliente encontrada.")

    #n_clients = len(device_names)
    n_clients = 7
    print(f"[*] {n_clients} clientes encontrados")

    server_args = {
        ServerOptions.MIN_CLIENTS      : n_clients,
        ServerOptions.NUM_ROUNDS       : 100,
        ServerOptions.STOP_VALUE       : 1.00,
        ServerOptions.PATIENT          : 10,
        ServerOptions.TARGET_METRIC    : MetricType.F1,
        ServerOptions.CLIENT_ACCEPTOR  : ClientAcceptorType.ALL_CLIENTS,
        ServerOptions.CLIENT_SELECTOR  : ClientSelectorType.ALL_CLIENTS,
        ServerOptions.MODEL_AGGREGATOR : AggregatorType.FED_AVG,
    }

    client_dimage = build_fed_node_docker_image(
        "nbaiot_client",
        "client_code/client_requirements.txt",
    )["tag"]

    net = MininetFed()
    try:
        s1 = net.addSwitch(name="s1", failMode="standalone")

        broker = net.addHost("broker", cls=FedBrokerNode)
        net.addLink(s1, broker)

        server = net.addHost(
            name="server",
            cls=FedServerNode,
            script=SERVER_SCRIPT,
            server_folder="./server/",
            server_args=server_args
        )

        #server = net.addHost(
        #    name="server",
        #    cls=FedServerNode,
        #    server_args=server_args
        #)
        net.addLink(s1, server)

        # 👇 nomes curtos de host + pasta real do device
        for i in range(n_clients):
            host_name = f"c{i}"   # <= nome curto e seguro
            client_folder = os.path.join(CLIENTS_ROOT, device_names[i])

            print(f"[*] Host {host_name} -> device {device_names[i]}")

            c = net.addHost(
                name=host_name,
                cls=FedClientNode,
                script=CLIENT_SCRIPT,
                dimage=client_dimage,
                client_folder=client_folder,
            )
            net.addLink(s1, c)

        print("\n*** Starting network...\n")
        net.build()
        net.addNAT(name="nat0", linkTo="s1", ip="192.168.210.254").configDefault()
        s1.start([])
        net.runFed()

    finally:
        net.stop()


if __name__ == "__main__":
    topology()

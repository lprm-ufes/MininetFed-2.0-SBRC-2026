from mininetfed.core.dto.metrics import MetricType
from mininetfed.core.fed_options import ServerOptions, ClientAcceptorType, ClientSelectorType, AggregatorType
from mininetfed.sim.net import MininetFed
from mininetfed.sim.nodes import FedServerNode, FedClientNode, FedBrokerNode
from mininetfed.sim.util.clients_generator import create_federated_client_datasets
from mininetfed.sim.util.docker_utils import build_fed_node_docker_image

n_clients = 4 # número de clientes
client_code_path = "client_code/" # pasta com o código cliente

# configurações do treinamento
server_args = {
    ServerOptions.MIN_CLIENTS: n_clients,
    ServerOptions.NUM_ROUNDS: 100,
    ServerOptions.STOP_VALUE: 0.99,
    ServerOptions.PATIENT: 10,
    ServerOptions.TARGET_METRIC: MetricType.F1,
    ServerOptions.CLIENT_ACCEPTOR: ClientAcceptorType.ALL_CLIENTS,
    ServerOptions.CLIENT_SELECTOR: ClientSelectorType.ALL_CLIENTS,
    ServerOptions.MODEL_AGGREGATOR: AggregatorType.FED_AVG
}

def run_experiment():

    # Carrega o dataset, o divide entre 4 clientes
    # seguindo a mesma distribuicao do original
    #client_paths = create_federated_client_datasets(
    #    dataset_source="dataset/wustl-ehms-2020_with_attacks_categories.csv",
    #    target_col="Label",
    #    n_clients=n_clients,
    #    split_mode="iid",
    #    code_src_dir=client_code_path,
    #)

    # Carrega o dataset, o divide entre 4 clientes
    # seguindo uma distribuicao desbalanceada em relação
    # a original
    client_paths = create_federated_client_datasets(
        dataset_source="dataset/wustl-ehms-2020_with_attacks_categories.csv",
        target_col="Label",
        n_clients=n_clients,
        split_mode="non_iid",
        alpha=0.8,
        code_src_dir=client_code_path,
    )

    # Cria uma imagem com o framework MininetFed 2.0 ja
    # instalado, alem dos pacotes dos quais o codigo cliente e
    # dependente
    client_dimage = build_fed_node_docker_image("torch_client", client_code_path + "client_requirements.txt")["tag"]

    # Cria a simulacao usando o modelo baseado no Mininet
    net = MininetFed()
    try:
        # Adiciona um switch
        s1 = net.addSwitch(name="s1", failMode='standalone')

        # Adiciona o broker de comunicacao entre os clientes
        broker = net.addHost(name="broker", cls=FedBrokerNode)
        net.addLink(s1, broker)

        # Adiciona o servidor padrão do MininetFed 2.0
        server = net.addHost(
            name="server",
            cls=FedServerNode,
            server_args=server_args
        )
        net.addLink(s1, server)

        # Adiciona os clientes federados. Cada cliente e mapeado para
        # uma das pastas retornada por create_federated_client_datasets
        # e usa a imagem docker criada por build_fed_node_docker_image
        for i in range(n_clients):
            net.addHost(
                name=f"client{i}",
                cls=FedClientNode,
                script="ehms_trainer.py",
                dimage=client_dimage,
                client_folder=client_paths[i]
            )
            net.addLink(s1, net.get(f"client{i}"))

        # dispara a simulacao da rede Mininet
        print(f'*** Starting network...\n')
        net.build()
        net.addNAT(name='nat0', linkTo='s1', ip='192.168.210.254').configDefault()
        s1.start([])

        # roda o experimento de aprendizado federado
        net.runFed()
    finally:
        # termina o experimento quando ocorre o criterio
        net.stop()

if __name__ == "__main__":
    run_experiment()

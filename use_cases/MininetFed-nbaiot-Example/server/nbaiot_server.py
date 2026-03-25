from mininetfed.core.dto.client_state import ClientState
from mininetfed.core.nodes.fed_server import FedServer

# Para implementar políticas/algoritmos de FL personalizados,
# é preciso implementar o código de servidor também. Aqui mostramos
# um exemplo de uso de uma política de seleção diferente da política
# padrão do MininetFed 2.0
class NBIOTServer(FedServer):
    def __init__(self) -> None:
        super().__init__()
        self.index_selection = 0
        self.clients_per_round = 3 # número de clientes por rodada

    # Seleção dos clientes baseada em uma fila circular, com 3 clientes
    # selecionados em cada rodada
    def select_clients(self, clients_states: list[ClientState]) -> list[str]:
        selected_clients = []

        n = len(clients_states)
        for _ in range(self.clients_per_round):
            selected_clients.append(clients_states[self.index_selection].get_client_id())
            self.index_selection = (self.index_selection + 1) % n

        print("Selected clients for this round: ", selected_clients)
        return selected_clients
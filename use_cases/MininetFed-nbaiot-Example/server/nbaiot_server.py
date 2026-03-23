from mininetfed.core.dto.client_state import ClientState
from mininetfed.core.nodes.fed_server import FedServer


class NBIOTServer(FedServer):
    def __init__(self) -> None:
        super().__init__()
        self.index_selection = 0
        self.clients_per_round = 3

    def select_clients(self, clients_states: list[ClientState]) -> list[str]:
        selected_clients = []

        n = len(clients_states)
        for _ in range(self.clients_per_round):
            selected_clients.append(clients_states[self.index_selection].get_client_id())
            self.index_selection = (self.index_selection + 1) % n

        print("Selected clients for this round: ", selected_clients)
        return selected_clients
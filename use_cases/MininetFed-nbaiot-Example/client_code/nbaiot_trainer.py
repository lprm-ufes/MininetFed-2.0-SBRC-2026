import os
import numpy as np
from numpy import ndarray

from mininetfed.core.dto.client_info import ClientInfo
from mininetfed.core.dto.dataset_info import DatasetInfo
from mininetfed.core.dto.metrics import Metrics, MetricType
from mininetfed.core.nodes.fed_client import FedClient

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


# ============================
# Hiperparâmetros locais
# ============================

LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 4096
N_EPOCHS = 5     # em FL costuma ser menor por round (ajuste se quiser)

HIDDEN_LAYERS = (128, 64)
DROPOUT = 0.2

USE_CLASS_WEIGHTS = True  # recomendado (desbalanceamento)


# ============================
# Dataset tabular para PyTorch
# ============================

class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        # CrossEntropyLoss espera y long (N,)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ============================
# MLP
# ============================

class MLP(nn.Module):
    def __init__(self, input_dim: int, n_classes: int):
        super().__init__()
        layers = []
        prev = input_dim

        for h in HIDDEN_LAYERS:
            layers += [
                nn.Linear(prev, h),
                nn.ReLU(),
                nn.Dropout(DROPOUT),
            ]
            prev = h

        layers += [nn.Linear(prev, n_classes)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def class_weights_from_y(y: np.ndarray, n_classes: int) -> torch.Tensor:
    """
    Peso inversamente proporcional à frequência (normalizado pela média).
    Bom para classes desbalanceadas (multiclasse e binário).
    """
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    w = counts.sum() / counts
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


# ============================
# Cliente Federado MininetFed 2.0 (PyTorch)
# ============================

class TrainerNBAIOT(FedClient):
    """
    Cliente federado usando MLP em PyTorch para o dataset N-BAIOT,
    carregado a partir de .npz com X,y (gerado pelo seu script de split).
    """

    def __init__(self):
        super().__init__()
        self.model: MLP | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.X_train: np.ndarray | None = None
        self.X_test: np.ndarray | None = None
        self.y_train: np.ndarray | None = None
        self.y_test: np.ndarray | None = None

        self.train_loader: DataLoader | None = None
        self.test_loader: DataLoader | None = None

        self.criterion = None
        self.optimizer = None

        self.n_classes: int | None = None
        self.input_dim: int | None = None

    # Todo cliente MininetFed 2.0 implementa o método abaixo para
    # carregar os dados que serão usados no treinamento local.
    # Informações sobre o dataset são entregues ao servidor através da
    # estrutura de dados DatasetInfo
    def prepare_data(self, path_to_data: str) -> DatasetInfo:
        npz_files = [f for f in os.listdir(path_to_data) if f.endswith(".npz")]
        if not npz_files:
            raise FileNotFoundError(f"Nenhum arquivo .npz encontrado em {path_to_data}")

        ds_path = os.path.join(path_to_data, npz_files[0])
        print(f"[CLIENT {self.get_client_id()}] Carregando dados de: {ds_path}")

        data = np.load(ds_path)
        X = data["X"].astype("float32")
        y = data["y"].astype("int64")

        # sanity checks
        if X.ndim != 2:
            raise ValueError(f"X deve ser 2D. Recebido: shape={X.shape}")
        if y.ndim != 1:
            y = y.reshape(-1).astype("int64")

        self.input_dim = X.shape[1]
        self.n_classes = int(np.max(y) + 1)

        # Split local treino/teste
        strat = y if len(np.unique(y)) > 1 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=strat,
            shuffle=True,
        )

        # Normalização local
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train).astype("float32")
        X_test_scaled = scaler.transform(X_test).astype("float32")

        self.X_train, self.X_test = X_train_scaled, X_test_scaled
        self.y_train, self.y_test = y_train, y_test

        # Datasets e loaders
        train_ds = TabularDataset(self.X_train, self.y_train)
        test_ds = TabularDataset(self.X_test, self.y_test)

        self.train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        self.test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

        # Modelo
        self.model = MLP(input_dim=self.input_dim, n_classes=self.n_classes).to(self.device)

        # Loss + otimizador
        if USE_CLASS_WEIGHTS:
            cw = class_weights_from_y(self.y_train, self.n_classes).to(self.device)
            print(f"[CLIENT {self.get_client_id()}] class_weights: {cw.detach().cpu().numpy()}")
            self.criterion = nn.CrossEntropyLoss(weight=cw)
        else:
            self.criterion = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
        )

        # logs de distribuição
        unique, counts = np.unique(self.y_train, return_counts=True)
        dist = {int(k): int(v) for k, v in zip(unique, counts)}
        print(f"[CLIENT {self.get_client_id()}] input_dim={self.input_dim} n_classes={self.n_classes}")
        print(f"[CLIENT {self.get_client_id()}] train_dist={dist}")

        return DatasetInfo(client_id=self.get_client_id(), num_samples=self.X_train.shape[0])

    # Todo cliente MininetFed 2.0 implementa o método abaixo para
    # enviar ao servidor informações através da estrutura de dados
    # ClientInfo, e que podem ser utilizadas como
    # critérios para a política de aceitação. Aqui nenhuma informação
    # é enviada.
    def set_client_info(self, client_info: ClientInfo):
        return ClientInfo(self.get_client_id())

    # Todo cliente MininetFed 2.0 implementa o método abaixo para
    # realizar o treinamento de uma rodada. returna True caso o
    # treinamento tenha sido realizado com sucesso e False caso contrário.
    def fit(self) -> bool:
        if self.model is None or self.train_loader is None:
            print(f"[CLIENT {self.get_client_id()}] Model ou train_loader não inicializados.")
            return False

        try:
            self.model.train()
            for epoch in range(1, N_EPOCHS + 1):
                running_loss = 0.0

                for X_batch, y_batch in self.train_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)

                    self.optimizer.zero_grad(set_to_none=True)
                    logits = self.model(X_batch)
                    loss = self.criterion(logits, y_batch)
                    loss.backward()
                    self.optimizer.step()

                    running_loss += loss.item() * X_batch.size(0)

                avg_loss = running_loss / len(self.train_loader.dataset)
                print(
                    f"[CLIENT {self.get_client_id()}] "
                    f"Epoch {epoch}/{N_EPOCHS} - loss={avg_loss:.6f}"
                )
            return True
        except Exception as e:
            print(f"Training failed in client {self.get_client_id()}: {e}")
            return False

    # Todo cliente MininetFed 2.0 implementa o método abaixo para
    # avaliar o modelo, gerando métricas como acurácia e F1-score. As
    # métricas são enviadas ao servidor para agregação através da
    # estrutura de dados Metrics.
    def evaluate(self) -> Metrics:
        if self.model is None or self.test_loader is None:
            print(f"[CLIENT {self.get_client_id()}] Model ou test_loader não inicializados.")
            return Metrics(client_id=self.get_client_id(), metrics={})

        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for X_batch, y_batch in self.test_loader:
                X_batch = X_batch.to(self.device)
                logits = self.model(X_batch)
                preds = torch.argmax(logits, dim=1).cpu().numpy()

                all_preds.append(preds)
                all_labels.append(y_batch.numpy())

        all_preds = np.concatenate(all_preds).astype("int32")
        all_labels = np.concatenate(all_labels).astype("int32")

        # Confusion Matrix (KxK)
        labels_order = list(range(self.n_classes)) if self.n_classes is not None else None
        cm = confusion_matrix(all_labels, all_preds, labels=labels_order)

        metrics = {
            MetricType.CONFUSION_MATRIX: cm.tolist(),  # JSON serializável
        }

        return Metrics(client_id=self.get_client_id(), metrics=metrics)

    # Todo cliente MininetFed 2.0 implementa o método abaixo para
    # atualizar os pesos do modelo local com os pesos do modelo
    # global recebidos do servidor
    def update_weights(self, global_weights: list[ndarray]):
        """
        Atualiza os pesos do modelo local com os pesos globais vindos do servidor.
        """
        if self.model is None:
            # Cria modelo “em branco” com base no tamanho local, se já houver dados:
            if self.X_train is not None and self.y_train is not None:
                self.input_dim = self.X_train.shape[1]
                self.n_classes = int(np.max(self.y_train) + 1)
                self.model = MLP(input_dim=self.input_dim, n_classes=self.n_classes).to(self.device)
            else:
                raise RuntimeError(
                    "Model ainda não inicializado e não há dados locais para inferir input_dim/n_classes."
                )

        with torch.no_grad():
            for param, w in zip(self.model.parameters(), global_weights):
                w_t = torch.from_numpy(w).to(self.device)
                param.data.copy_(w_t)

    # Todo cliente MininetFed 2.0 implementa o método abaixo para
    # para obter os pesos do modelo local e enviá-los para o servidor de
    # agregação.
    def get_weights(self) -> list[ndarray]:
        """
        Retorna os pesos locais como lista de ndarrays, para o servidor agregar.
        """
        if self.model is None:
            raise RuntimeError("Model ainda não inicializado em get_weights().")

        return [param.detach().cpu().numpy().copy() for param in self.model.parameters()]

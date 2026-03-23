import os
import numpy as np
import pandas as pd
from numpy import ndarray

from mininetfed.core.dto.client_info import ClientInfo
from mininetfed.core.dto.dataset_info import DatasetInfo
from mininetfed.core.dto.metrics import Metrics, MetricType
from mininetfed.core.nodes.fed_client import FedClient

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


# ============================
# Hiperparâmetros locais
# ============================

LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 256
N_EPOCHS = 10

DATA_FILE = "dataset_subset.csv"

LABEL_COL = "Label"
ID_COL = "instance_id"   # se existir, não entra como feature


# ============================
# Dataset tabular para PyTorch
# ============================

class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ============================
# MLP do artigo (EHMS ANN)
# ============================

class EHMSANN(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 40),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(40, 40),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(40, 20),
            nn.ReLU(),

            nn.Linear(20, 10),
            nn.ReLU(),

            nn.Linear(10, 10),
            nn.ReLU(),

            nn.Linear(10, 10),
            nn.ReLU(),

            nn.Linear(10, 10),
            nn.ReLU(),

            nn.Linear(10, 1),
        )

    def forward(self, x):
        return self.net(x)


# ============================
# Cliente Federado (PyTorch)
# ============================

class TrainerEHMS(FedClient):
    def __init__(self):
        super().__init__()
        self.model: EHMSANN | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.features: list[str] | None = None
        self.scaler: StandardScaler | None = None

        self.X_train: np.ndarray | None = None
        self.X_test: np.ndarray | None = None
        self.y_train: np.ndarray | None = None
        self.y_test: np.ndarray | None = None

        self.train_loader: DataLoader | None = None
        self.test_loader: DataLoader | None = None

        self.criterion = None
        self.optimizer = None

    def _prepare_ehms_dataframe(self, csv_path: str):
        df = pd.read_csv(csv_path)

        network_features = [
            "Sport", "Dport",
            "SrcBytes", "DstBytes", "SrcLoad", "DstLoad",
            "SrcGap", "DstGap", "SIntPkt", "DIntPkt",
            "SIntPktAct", "DIntPktAct", "SrcJitter", "DstJitter",
            "sMaxPktSz", "dMaxPktSz", "sMinPktSz", "dMinPktSz",
            "Dur", "Trans", "TotPkts", "TotBytes",
            "Load", "Loss", "pLoss", "pSrcLoss", "pDstLoss", "Rate"
        ]
        bio_features = [
            "Temp", "SpO2", "Pulse_Rate",
            "SYS", "DIA", "Heart_rate",
            "Resp_Rate", "ST",
        ]

        features = [c for c in (network_features + bio_features) if c in df.columns]

        numeric_features = []
        for c in features:
            if df[c].dtype == object:
                df[c] = df[c].astype(str).str.strip()

            df[c] = pd.to_numeric(df[c], errors="coerce")

            if df[c].notna().sum() > 0:
                numeric_features.append(c)
            else:
                print(f"[CLIENT {self.get_client_id()}] [WARN] Removendo coluna não numérica/inválida: {c}")

        features = numeric_features

        if LABEL_COL not in df.columns:
            raise RuntimeError(
                f"[CLIENT {self.get_client_id()}] A coluna '{LABEL_COL}' não existe no CSV."
            )

        df[LABEL_COL] = pd.to_numeric(df[LABEL_COL], errors="coerce")

        df_model = df[features + [LABEL_COL]].dropna().copy()

        if "instance_id" not in df_model.columns:
            df_model["instance_id"] = df_model.index.astype(np.int64)

        cols = ["instance_id"] + features + [LABEL_COL]
        df_model = df_model[cols]

        return df_model, features

    def prepare_data(self, path_to_data: str) -> DatasetInfo:
        data_path = os.path.join(path_to_data, DATA_FILE)

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Arquivo não encontrado: {data_path}")

        print(f"[CLIENT {self.get_client_id()}] Lendo dataset: {data_path}")

        df, features = self._prepare_ehms_dataframe(data_path)

        excluded = {LABEL_COL}
        if ID_COL in df.columns:
            excluded.add(ID_COL)

        features = [c for c in df.columns if c not in excluded]

        if not features:
            raise RuntimeError(
                f"[CLIENT {self.get_client_id()}] Nenhuma feature encontrada após excluir {excluded}."
            )

        self.features = features

        X = df[features].to_numpy(dtype=np.float32)
        y = df[LABEL_COL].to_numpy(dtype=np.int64)

        if len(np.unique(y)) < 2:
            raise RuntimeError(
                f"[CLIENT {self.get_client_id()}] O subset local possui apenas uma classe em '{LABEL_COL}'."
            )

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42,
                stratify=y,
                shuffle=True,
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42,
                shuffle=True,
            )

        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train).astype(np.float32)
        X_test = self.scaler.transform(X_test).astype(np.float32)

        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test

        train_ds = TabularDataset(self.X_train, self.y_train)
        test_ds = TabularDataset(self.X_test, self.y_test)

        self.train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        self.test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

        input_dim = self.X_train.shape[1]
        self.model = EHMSANN(input_dim=input_dim).to(self.device)

        n_pos = int((self.y_train == 1).sum())
        n_neg = int((self.y_train == 0).sum())
        pos_weight_value = n_neg / max(n_pos, 1)
        pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32).to(self.device)

        print(f"[CLIENT {self.get_client_id()}] Train samples: {len(self.y_train)}")
        print(f"[CLIENT {self.get_client_id()}] Test samples : {len(self.y_test)}")
        print(f"[CLIENT {self.get_client_id()}] Features     : {input_dim}")
        print(f"[CLIENT {self.get_client_id()}] Classe positiva (1): {n_pos}")
        print(f"[CLIENT {self.get_client_id()}] Classe negativa (0): {n_neg}")
        print(f"[CLIENT {self.get_client_id()}] pos_weight usado: {pos_weight_value:.4f}")

        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=LR,
            weight_decay=WEIGHT_DECAY,
        )

        return DatasetInfo(
            client_id=self.get_client_id(),
            num_samples=int(self.X_train.shape[0]),
        )

    def set_client_info(self, client_info: ClientInfo):
        return ClientInfo(self.get_client_id())

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

                    self.optimizer.zero_grad()
                    logits = self.model(X_batch)
                    loss = self.criterion(logits, y_batch)
                    loss.backward()
                    self.optimizer.step()

                    running_loss += loss.item() * X_batch.size(0)

                avg_loss = running_loss / len(self.train_loader.dataset)
                print(f"[CLIENT {self.get_client_id()}] Epoch {epoch}/{N_EPOCHS} - loss={avg_loss:.4f}")

            return True

        except Exception as e:
            print(f"[CLIENT {self.get_client_id()}] Training failed: {e}")
            return False

    def evaluate(self) -> Metrics:
        if self.model is None or self.test_loader is None:
            print(f"[CLIENT {self.get_client_id()}] Model ou test_loader não inicializados.")
            return Metrics(client_id=self.get_client_id(), metrics={})

        self.model.eval()

        all_probs = []
        all_labels = []

        with torch.no_grad():
            for X_batch, y_batch in self.test_loader:
                X_batch = X_batch.to(self.device)

                logits = self.model(X_batch)
                probs = torch.sigmoid(logits).cpu().numpy()

                all_probs.append(probs)
                all_labels.append(y_batch.cpu().numpy())

        all_probs = np.vstack(all_probs).ravel()
        all_labels = np.vstack(all_labels).ravel().astype("int32")

        preds = (all_probs >= 0.5).astype("int32")

        cm = confusion_matrix(all_labels, preds, labels=[0, 1])

        metrics = {
            MetricType.CONFUSION_MATRIX: cm.tolist()
        }

        return Metrics(
            client_id=self.get_client_id(),
            metrics=metrics
        )

    def update_weights(self, global_weights: list[ndarray]):
        if self.model is None:
            if self.X_train is not None:
                input_dim = self.X_train.shape[1]
                self.model = EHMSANN(input_dim=input_dim).to(self.device)
            else:
                raise RuntimeError("Model ainda não inicializado e não há dados locais para inferir input_dim.")

        with torch.no_grad():
            for param, w in zip(self.model.parameters(), global_weights):
                w_t = torch.from_numpy(w).to(self.device)
                if param.data.shape != w_t.shape:
                    raise RuntimeError(
                        f"[CLIENT {self.get_client_id()}] Shape mismatch ao atualizar pesos: "
                        f"param={tuple(param.data.shape)} vs w={tuple(w_t.shape)}"
                    )
                param.data.copy_(w_t)

    def get_weights(self) -> list[ndarray]:
        if self.model is None:
            raise RuntimeError("Model ainda não inicializado em get_weights().")

        return [p.detach().cpu().numpy().copy() for p in self.model.parameters()]
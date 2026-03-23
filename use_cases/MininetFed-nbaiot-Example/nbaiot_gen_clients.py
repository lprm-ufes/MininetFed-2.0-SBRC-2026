import argparse
import os
import glob
import shutil
import subprocess
import numpy as np
import pandas as pd


# ======================================================
# Script de criação dos clientes
# ======================================================
#
# Exemplo de comando
# python nbaiot_gen_clients.py --data_root ./dataset --out_dir ./clients --py_src_dir ./client_code -N 9 --mode binary --extract_rar

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def get_py_files(py_src_dir: str):
    if not os.path.isdir(py_src_dir):
        raise ValueError(f"PY_SRC_DIR inválido: {py_src_dir}")
    return [
        os.path.join(py_src_dir, f)
        for f in os.listdir(py_src_dir)
        if f.endswith(".py")
    ]

def try_extract_rar(rar_path: str, out_dir: str) -> bool:
    ensure_dir(out_dir)

    # 1) unrar
    try:
        r = subprocess.run(
            ["unrar", "x", "-o+", rar_path, out_dir],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        if r.returncode == 0:
            return True
    except FileNotFoundError:
        pass

    # 2) 7z
    try:
        r = subprocess.run(
            ["7z", "x", "-y", f"-o{out_dir}", rar_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        if r.returncode == 0:
            return True
    except FileNotFoundError:
        pass

    return False

def read_csv_numeric(path: str, max_rows: int = 0) -> pd.DataFrame:
    nrows = None if max_rows == 0 else max_rows
    df = pd.read_csv(path, nrows=nrows)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df

def label_from_attack_csv_name(csv_path: str) -> str:
    base = os.path.basename(csv_path).lower().replace(".csv", "")
    base = base.replace(" ", "_").replace("-", "_")
    return base

def list_device_dirs(data_root: str) -> list[str]:
    device_dirs = sorted(
        [p for p in glob.glob(os.path.join(data_root, "*")) if os.path.isdir(p)]
    )
    if not device_dirs:
        raise FileNotFoundError(f"Nenhuma pasta de device encontrada em: {data_root}")
    return device_dirs

def gather_attack_csvs(device_dir: str, extract_rar: bool, extract_dirname: str) -> list[str]:
    rar_paths = sorted(set(glob.glob(os.path.join(device_dir, "*.rar"))))

    extracted_dirs = []
    if extract_rar and rar_paths:
        for rp in rar_paths:
            out_dir = os.path.join(
                device_dir, extract_dirname, os.path.basename(rp).replace(".rar", "")
            )
            ok = try_extract_rar(rp, out_dir)
            if ok:
                extracted_dirs.append(out_dir)

    search_dirs = set(extracted_dirs)

    # fallback: subpastas já existentes
    for sub in glob.glob(os.path.join(device_dir, "**"), recursive=True):
        if os.path.isdir(sub) and sub != device_dir:
            if glob.glob(os.path.join(sub, "*.csv")):
                search_dirs.add(sub)

    attack_csvs = []
    for d in sorted(search_dirs):
        for csvp in sorted(glob.glob(os.path.join(d, "*.csv"))):
            if os.path.basename(csvp).lower() == "benign_traffic.csv":
                continue
            attack_csvs.append(csvp)

    return attack_csvs

def build_device_xy(device_dir: str, mode: str, max_rows: int,
                    extract_rar: bool, extract_dirname: str):
    benign_path = os.path.join(device_dir, "benign_traffic.csv")
    if not os.path.exists(benign_path):
        raise FileNotFoundError(f"Não achei benign_traffic.csv em {device_dir}")

    df_benign = read_csv_numeric(benign_path, max_rows)
    attack_csvs = gather_attack_csvs(device_dir, extract_rar, extract_dirname)

    frames = [df_benign]
    y_str_parts = [np.array(["BENIGN"] * len(df_benign), dtype=object)]

    for csvp in attack_csvs:
        df_a = read_csv_numeric(csvp, max_rows)
        frames.append(df_a)
        if mode == "binary":
            y_str_parts.append(np.array(["ATTACK"] * len(df_a), dtype=object))
        else:
            y_str_parts.append(
                np.array([label_from_attack_csv_name(csvp)] * len(df_a), dtype=object)
            )

    df_all = pd.concat(frames, ignore_index=True, sort=False).fillna(0.0)
    y_str = np.concatenate(y_str_parts, axis=0)

    if mode == "binary":
        y = (y_str != "BENIGN").astype(np.int64)
        classes = np.array(["BENIGN", "ATTACK"], dtype=object)
    else:
        classes, y = np.unique(y_str, return_inverse=True)
        y = y.astype(np.int64)

    X = df_all.values.astype(np.float32)
    return X, y, classes


# ======================================================
# MAIN
# ======================================================

def main():
    parser = argparse.ArgumentParser(
        description="Preparar clientes do MininetFed a partir do dataset N-BaIoT"
    )

    parser.add_argument("--data_root", required=True,
                        help="Pasta raiz com as pastas dos devices (N-BaIoT)")
    parser.add_argument("--out_dir", required=True,
                        help="Pasta de saída para os clientes")
    parser.add_argument("--py_src_dir", required=True,
                        help="Pasta com arquivos .py a serem copiados para cada cliente")

    parser.add_argument("-N", "--n_clients", type=int, required=True,
                        help="Número de clientes/devices a usar (primeiros N)")

    parser.add_argument("--mode", choices=["binary", "multiclass"],
                        default="binary", help="Modo de rótulo")

    parser.add_argument("--max_rows_per_csv", type=int, default=0,
                        help="Limite de linhas por CSV (0 = sem limite)")

    parser.add_argument("--extract_rar", action="store_true",
                        help="Extrair arquivos .rar automaticamente")
    parser.add_argument("--extract_dirname", default="_extracted",
                        help="Subpasta onde os .rar serão extraídos")

    args = parser.parse_args()

    device_dirs = list_device_dirs(args.data_root)
    selected = device_dirs[:args.n_clients]

    print(f"Devices encontrados: {len(device_dirs)}")
    print(f"Selecionados (primeiros {args.n_clients}): {len(selected)}")
    for d in selected:
        print(" -", os.path.basename(d))

    ensure_dir(args.out_dir)
    py_files = get_py_files(args.py_src_dir)

    info_path = os.path.join(args.out_dir, "clients_info.txt")
    with open(info_path, "w", encoding="utf-8") as f:
        f.write(f"MODE={args.mode}\n")
        f.write(f"DATA_ROOT={os.path.abspath(args.data_root)}\n")

    for dev_dir in selected:
        dev_name = os.path.basename(dev_dir)
        client_dir = os.path.join(args.out_dir, dev_name)
        ensure_dir(client_dir)

        print(f"\n=== Criando cliente: {dev_name} ===")

        X, y, classes = build_device_xy(
            dev_dir,
            mode=args.mode,
            max_rows=args.max_rows_per_csv,
            extract_rar=args.extract_rar,
            extract_dirname=args.extract_dirname,
        )

        npz_path = os.path.join(client_dir, "nbaiot_subset.npz")
        np.savez_compressed(npz_path, X=X, y=y)

        print(
            f"[{dev_name}] salvo: {npz_path} | "
            f"X={X.shape} y={y.shape} bincount={np.bincount(y).tolist()}"
        )

        # salvar classes
        classes_path = os.path.join(client_dir, "classes.txt")
        with open(classes_path, "w", encoding="utf-8") as f:
            for i, name in enumerate(classes):
                f.write(f"{i}\t{name}\n")

        # copiar .py
        for src in py_files:
            shutil.copy2(src, client_dir)

        print(f"[{dev_name}] copiados {len(py_files)} arquivo(s) .py")

        # registrar info
        with open(info_path, "a", encoding="utf-8") as f:
            f.write(
                f"\nclient={dev_name}\n"
                f"  X_shape={X.shape}\n"
                f"  y_bincount={np.bincount(y).tolist()}\n"
            )

    print(f"\nConcluído! Clientes criados em: {args.out_dir}")
    print(f"Resumo: {info_path}")


if __name__ == "__main__":
    main()

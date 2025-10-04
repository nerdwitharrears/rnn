# debug_load_h5.py
import h5py
import numpy as np
import traceback
import json
import os

H5_PATH = "simple_rnn_imdb.h5"
EXPORT_NPZ = "simple_rnn_imdb_weights_dump.npz"
SAVEDMODEL_DIR = "simple_rnn_imdb_savedmodel_rebuilt"

def recurse_print(group, prefix=""):
    for key in group.keys():
        item = group[key]
        path = f"{prefix}/{key}" if prefix else key
        if isinstance(item, h5py.Dataset):
            print(f"DATASET: {path} shape={item.shape} dtype={item.dtype}")
        elif isinstance(item, h5py.Group):
            print(f"GROUP:   {path}")
            recurse_print(item, path)

def inspect_h5(path):
    print(f"\n--- Inspecting HDF5: {path} ---")
    with h5py.File(path, "r") as f:
        print("Top-level keys:", list(f.keys()))
        # Common keys: 'model_weights', 'layer_names', 'model_config', 'optimizer_weights', etc.
        if "model_config" in f:
            try:
                cfg = f["model_config"][()]
                # model_config may be bytes; try decode
                if isinstance(cfg, (bytes, bytearray)):
                    cfg = cfg.decode("utf-8")
                print("\nmodel_config (truncated):\n", cfg[:1000])
            except Exception as e:
                print("Could not read model_config:", e)
        if "model_weights" in f:
            print("\n--- model_weights groups ---")
            # print layer names
            mw = f["model_weights"]
            print("Layers in model_weights:", list(mw.keys()))
            # print each group's datasets and shapes
            for ln in mw.keys():
                print(f"\nLayer group: {ln}")
                group = mw[ln]
                recurse_print(group, prefix=f"model_weights/{ln}")
        else:
            print("\nNo 'model_weights' group found. Printing all groups recursively:")
            recurse_print(f, prefix="")

def export_weights_to_npz(h5_path, npz_path):
    arrays = {}
    with h5py.File(h5_path,"r") as f:
        # If model_weights exists prefer that structure
        container = f.get("model_weights", f)
        # We'll iterate recursively and save datasets with full path names
        def collect(g, prefix=""):
            for key in g.keys():
                item = g[key]
                name = f"{prefix}/{key}" if prefix else key
                if isinstance(item, h5py.Dataset):
                    arrays[name] = item[()]
                else:
                    collect(item, name)
        collect(container, "")
    if not arrays:
        raise RuntimeError("No dataset arrays found to export.")
    # Save to an npz — names become keys (replace slashes with $$$ to keep valid keys)
    safe_arrays = {k.replace("/","$$$"): v for k,v in arrays.items()}
    np.savez_compressed(npz_path, **safe_arrays)
    print(f"Exported {len(safe_arrays)} arrays to {npz_path}")
    return npz_path

# Attempt to rebuild common RNN architecture and load weights from npz mapping
def try_rebuild_and_load(npz_path):
    print("\n--- Attempting to rebuild a typical IMDB SimpleRNN model and load weights ---")
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
    except Exception as e:
        print("TensorFlow import failed:", e)
        return False

    # You MUST set these to what you used during original training. Try common defaults:
    vocab_size = 10000
    embedding_dim = 32
    rnn_units = 32
    input_length = 500

    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=input_length, name="embedding"),
        SimpleRNN(rnn_units, name="simple_rnn"),
        Dense(1, activation="sigmoid", name="output")
    ])
    print("Rebuilt model summary:")
    model.summary()

    # Load npz arrays into memory
    arrs = np.load(npz_path)
    keys = list(arrs.keys())
    print("npz keys (sample):", keys[:10])

    # Heuristic: for each model layer, find arrays whose key contains the layer.name
    success_layers = []
    failed_layers = []
    for layer in model.layers:
        lname = layer.name
        # Search for candidate keys containing the layer name
        candidates = [k for k in arrs.files if lname in k]
        # Sort by key name length (shorter likely matches top-level weight tensors first)
        candidates = sorted(candidates, key=lambda x: len(x))
        if not candidates:
            print(f"[WARN] No arrays found in NPZ for layer '{lname}'")
            failed_layers.append(lname)
            continue
        # Extract arrays in naive order — this is heuristic and may need manual mapping
        try:
            # Try to pick the arrays that match shapes expected by the layer weights
            expected_count = len(layer.get_weights())
            chosen = []
            # Try to match by shapes
            for k in candidates:
                arr = arrs[k]
                if len(chosen) < expected_count:
                    chosen.append(arr)
            if len(chosen) != expected_count:
                print(f"[WARN] For layer {lname} expected {expected_count} arrays but found {len(chosen)} candidates. Trying layer.set_weights anyway.")
            layer.set_weights(chosen)
            print(f"[OK] Set weights for layer {lname} (used {len(chosen)} arrays).")
            success_layers.append(lname)
        except Exception as ex:
            print(f"[ERROR] Failed set_weights for layer {lname}: {ex}")
            traceback.print_exc()
            failed_layers.append(lname)

    print("\nResult: success_layers=", success_layers)
    print("        failed_layers=", failed_layers)

    # If a lot succeeded, save the model in SavedModel format to reuse
    if len(success_layers) >= 1 and len(failed_layers) == 0:
        try:
            model.save(SAVEDMODEL_DIR)
            print(f"Saved rebuilt model to {SAVEDMODEL_DIR}")
            return True
        except Exception as se:
            print("Failed to save SavedModel:", se)
            return False
    else:
        print("Not all layers loaded cleanly — manual mapping will be required.")
        return False

if __name__ == "__main__":
    if not os.path.exists(H5_PATH):
        print("H5 file not found at path:", H5_PATH)
        raise SystemExit(1)

    inspect_h5(H5_PATH)
    try:
        npz = export_weights_to_npz(H5_PATH, EXPORT_NPZ)
        print("Proceeding to attempt rebuild+load from exported npz...")
        ok = try_rebuild_and_load(npz)
        if ok:
            print("Recovery succeeded: SavedModel created.")
        else:
            print("Automated recovery incomplete. Please paste the printed HDF5 layer names and group/dataset listing here for manual mapping.")
    except Exception as e_all:
        print("Export or rebuild failed:")
        traceback.print_exc()
        raise

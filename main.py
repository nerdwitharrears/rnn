# Step 1: Import Libraries and Load the Model (robust version)
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
# from tensorflow.keras.models import load_model   # not imported here (we'll import inside try)
import h5py
import traceback

# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Try to load full model first (may fail due to serialization differences)
model = None
h5_path = 'simple_rnn_imdb.h5'

try:
    from tensorflow.keras.models import load_model
    model = load_model(h5_path)
    print("Loaded full model via load_model()")
except Exception as e:
    print("load_model() failed — will try rebuilding architecture and loading weights.")
    traceback.print_exc()

    # === IMPORTANT: set these to the values you used when training ===
    vocab_size = 10000       # e.g. 10000
    embedding_dim = 32       # e.g. 32
    rnn_units = 32           # e.g. 32 (units in SimpleRNN)
    input_length = 500       # sequence length used during training
    # ================================================================

    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

    # Recreate the same architecture used in training (names help by_name matching)
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=input_length, name="embedding"),
        SimpleRNN(rnn_units, name="simple_rnn"),
        Dense(1, activation="sigmoid", name="output")
    ])

    # Try loading weights by name (safe if names match)
    try:
        model.load_weights(h5_path, by_name=True)
        print("Loaded weights with model.load_weights(..., by_name=True)")
    except Exception as e2:
        print("model.load_weights(..., by_name=True) failed — attempting manual HDF5 inspect + set_weights")
        traceback.print_exc()

        # Manual fallback: open HDF5 and copy datasets into layer.set_weights
        try:
            with h5py.File(h5_path, 'r') as f:
                print("HDF5 top-level keys:", list(f.keys()))
                if 'model_weights' in f:
                    mw = f['model_weights']
                    for layer in model.layers:
                        if layer.name in mw:
                            group = mw[layer.name]
                            # HDF5 stores weight datasets under names like 'kernel:0', 'bias:0', etc.
                            weight_list = []
                            # maintain HDF5 insertion order for weights
                            for name in group.keys():
                                ds = group[name]
                                weight_list.append(ds[()])
                            try:
                                layer.set_weights(weight_list)
                                print(f"Set weights for layer: {layer.name}")
                            except Exception as set_err:
                                print(f"Failed to set weights for {layer.name}: {set_err}")
                else:
                    # maybe file only contains 'weights' or other layout
                    print("No 'model_weights' group found. Top-level groups:", list(f.keys()))
                    # Dump further details for debugging
                    for k in f.keys():
                        print(k, "->", list(f[k].keys()))
            print("Manual weight-loading attempt completed.")
        except Exception as e3:
            print("Manual HDF5 inspection/load failed:")
            traceback.print_exc()
            raise RuntimeError("Unable to load model or weights from HDF5. Please re-save model as SavedModel or weights-only from original training environment.") from e3

# At this point `model` should be usable (or an exception already raised).

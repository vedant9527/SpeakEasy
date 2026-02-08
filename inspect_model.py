from pathlib import Path
import json
import h5py

from tensorflow.keras.models import load_model
from tensorflow import keras
from tensorflow.keras import initializers as keras_initializers

APP_ROOT = Path(__file__).resolve().parent
MODEL_CANDIDATES = [APP_ROOT / "action.h5", APP_ROOT / "m1.h5"]


def main() -> None:
    model_path = next((p for p in MODEL_CANDIDATES if p.exists()), None)
    if model_path is None:
        raise SystemExit("No model found. Place action.h5 or m1.h5 in the project root.")

    def _clean_config(obj):
        if isinstance(obj, dict):
            return {k: _clean_config(v) for k, v in obj.items() if k not in {"module", "registered_name"}}
        if isinstance(obj, list):
            return [_clean_config(x) for x in obj]
        return obj

    def _load_model_flexible(path: Path):
        # Try normal load.
        try:
            return load_model(path, compile=False)
        except Exception:
            pass

        # Fallback: clean config and load weights manually.
        with h5py.File(path, "r") as f:
            cfg = f.attrs.get("model_config")
            if cfg is None:
                raise ValueError("model_config missing in h5 file")
            if isinstance(cfg, bytes):
                cfg = cfg.decode("utf-8")
        cfg_dict = json.loads(cfg)
        cfg_clean = _clean_config(cfg_dict)
        model = keras.models.model_from_json(json.dumps(cfg_clean))
        model.load_weights(path)
        return model

    model = _load_model_flexible(model_path)
    print(f"Loaded: {model_path.name}")
    model.summary()


if __name__ == "__main__":
    main()

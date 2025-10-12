import os
import itertools
import copy
from scripts.utils import load_config
from scripts.train import main as train_main

# Simple grid search using config overrides
def override_config(cfg, updates):
    new_cfg = copy.deepcopy(cfg)
    for k, v in updates.items():
        new_cfg[k] = v
    return new_cfg

def run_tuning():
    base = load_config()
    grid = {
        "learning_rate": base["tuning"]["learning_rates"],
        "batch_size": base["tuning"]["batch_sizes"],
        "weight_decay": base["tuning"]["weight_decays"],
        "dropout": base["tuning"]["dropout_rates"],
        "freeze_backbone": base["tuning"]["freeze_backbone_options"],
    }

    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))

    print(f"Total trials: {len(combos)}")
    for idx, combo in enumerate(combos, 1):
        updates = {k: v for k, v in zip(keys, combo)}
        cfg_override_path = f"configs/run_{idx}.yaml"

        # Write temp config
        import yaml
        new_cfg = override_config(base, updates)
        with open(cfg_override_path, "w") as f:
            yaml.safe_dump(new_cfg, f)

        print(f"\nTrial {idx}: {updates}")
        train_main(cfg_override_path)

        # After training, you can parse outputs/metrics or logs to pick best run

if __name__ == "__main__":
    run_tuning()

from pathlib import Path
from typing import List, Optional
import yaml


class TrainingConfig:
    """

    """

    def __init__(self, yaml_path: str):
        self.yaml_path = yaml_path
        self._load_yaml(yaml_path)

    def _load_yaml(self, yaml_path):
        cfg = yaml.safe_load(open(yaml_path, "r"))

        # source_file
        self.source_path: Path = Path(cfg["source_file"]["path"]).resolve()
        self.column_power: str = cfg["source_file"].get("column_power_name")
        self.column_time_stamp: str = cfg["source_file"].get("column_timestamps")

        # estimators
        self.n_estimators: int = cfg["model"].get("n_estimators")
        self.max_depth: int = cfg["model"].get("max_depth")

        # estimators
        self.hourly_horizon: int = cfg["task"].get("hourly_horizon")
        self.step_per_hour: int = cfg["task"].get("step_per_hour")

        self.split_date: str = cfg["split_train_test"].get("date_split")

    def override(self, **kwargs):
        """
        Override values from CLI or other sources.
        """
        for k, v in kwargs.items():
            if v is not None and hasattr(self, k):
                setattr(self, k, v)

    def __repr__(self):
        return f"<column power={self.column_power} columns time stamp={self.column_time_stamp} n estimators={self.n_estimators} n epochs={self.max_depth}>"

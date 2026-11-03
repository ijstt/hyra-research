import yaml
from pathlib import Path
from typing import Any, Dict, Union


class SpecValidationError(Exception):
    """Raised when an experiment specification violates contract invariants."""
    pass


def load_and_validate_spec(spec_path: Union[str, Path]) -> Dict[str, Any]:
    """Load and validate an experiment specification contract (spec.yaml).
    
    Args:
        spec_path: Path to the YAML specification file.
        
    Returns:
        Validated dictionary of specification parameters.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        SpecValidationError: If mandatory fields are missing or malformed.
    """
    path = Path(spec_path)
    if not path.is_file():
        raise FileNotFoundError(f"Experiment specification file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        try:
            data = yaml.safe_load(f)
        except yaml.YAMLError as err:
            raise SpecValidationError(f"Invalid YAML syntax in {path}: {err}") from err

    if not isinstance(data, dict):
        raise SpecValidationError(f"Specification in {path} must be a dictionary mapping.")

    required_keys = [
        "experiment_id",
        "title",
        "hypothesis",
        "falsification",
        "compute_budget",
    ]
    for key in required_keys:
        if key not in data:
            raise SpecValidationError(f"Missing mandatory specification section: '{key}'")

    falsification = data.get("falsification")
    if not isinstance(falsification, dict):
        raise SpecValidationError("Block 'falsification' must be a dictionary mapping.")

    for f_key in ["target_metric", "baseline_metric", "required_gain"]:
        if f_key not in falsification:
            raise SpecValidationError(f"Missing mandatory field in 'falsification': '{f_key}'")

    compute_budget = data.get("compute_budget")
    if not isinstance(compute_budget, dict):
        raise SpecValidationError("Block 'compute_budget' must be a dictionary mapping.")

    return data

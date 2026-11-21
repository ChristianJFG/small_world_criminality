"""Validated experiment settings; no computation or file writes at import time."""
from dataclasses import dataclass, field, asdict
from pathlib import Path
import math
try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

from .networks import make_graph

TOPOLOGIES = ("ring", "square", "triangular")


def integer(name, value, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def validate_probabilities(values):
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError("probabilities must be a nonempty list")
    if any(type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1 for p in values):
        raise ValueError("probabilities must be finite numbers in [0, 1]")
    if len(set(values)) != len(values):
        raise ValueError("probabilities must be distinct")


@dataclass
class ExperimentConfig:
    preset: str = "quick"
    topologies: list[str] = field(default_factory=lambda: list(TOPOLOGIES))
    nodes: int = 200
    seeds: int | None = None
    seed_start: int = 0
    ring_k: int = 4
    p_points: int | None = None
    probabilities: dict[str, list[float]] = field(default_factory=dict)
    output: str = "results"
    format: str = "png"

    def validate(self):
        if self.preset not in ("quick", "manuscript"):
            raise ValueError("preset must be quick or manuscript")
        if not isinstance(self.topologies, list) or not self.topologies or any(t not in TOPOLOGIES for t in self.topologies):
            raise ValueError(f"topologies must be a nonempty list drawn from {TOPOLOGIES}")
        if len(set(self.topologies)) != len(self.topologies):
            raise ValueError("topologies must not contain duplicates")
        integer("nodes", self.nodes, 3)
        if self.seeds is not None:
            integer("seeds", self.seeds, 1)
        integer("seed_start", self.seed_start)
        integer("ring_k", self.ring_k, 2)
        if self.ring_k % 2 or ("ring" in self.topologies and self.ring_k >= self.nodes):
            raise ValueError("ring_k must be even and less than nodes for a ring")
        if self.p_points is not None:
            integer("p_points", self.p_points, 1)
            if self.preset != "quick":
                raise ValueError("p_points applies only to the quick preset; use probabilities for custom manuscript grids")
            limit = 35 if any(t in self.topologies for t in ("ring", "triangular")) else 105
            if self.p_points > limit:
                raise ValueError(f"p_points must be <= {limit} to avoid duplicate historical samples")
        if not isinstance(self.probabilities, dict):
            raise ValueError("probabilities must be a table keyed by topology")
        for topology, values in self.probabilities.items():
            if topology not in self.topologies:
                raise ValueError(f"probabilities contains unselected topology: {topology}")
            validate_probabilities(values)
        if not isinstance(self.output, str) or not self.output.strip():
            raise ValueError("output must be a nonempty path")
        if self.format not in ("png", "svg", "pdf"):
            raise ValueError("format must be png, svg, or pdf")
        # Validate all lattice dimensions before any expensive computation/output.
        for topology in self.topologies:
            make_graph(topology, self.nodes, 0, self.seed_start, self.ring_k)
        return self

    @property
    def seed_count(self):
        return self.seeds if self.seeds is not None else (10 if self.preset == "manuscript" else 2)

    def resolved(self):
        from .experiments import sweep_probabilities
        result = asdict(self)
        result["output"] = str(Path(self.output).resolve())
        result["seeds"] = self.seed_count
        result["seed_values"] = list(range(self.seed_start, self.seed_start + self.seed_count))
        result["probabilities"] = {
            t: sweep_probabilities(t, self.preset, self.p_points, self.probabilities.get(t)).tolist()
            for t in self.topologies
        }
        return result


def load_config(path=None, overrides=None):
    data = {}
    if path is not None:
        path = Path(path).resolve()
        with path.open("rb") as stream:
            data = tomllib.load(stream)
        unknown = set(data) - set(ExperimentConfig.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown configuration keys: {', '.join(sorted(unknown))}")
        output = data.get("output", "results")
        if isinstance(output, str):
            data["output"] = str(path.parent / output)
    data.update({k: v for k, v in (overrides or {}).items() if v is not None})
    return ExperimentConfig(**data).validate()

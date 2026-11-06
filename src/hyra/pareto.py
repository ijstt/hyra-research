"""Multi-Objective Pareto Frontier Dominance Engine for Scientific Discovery.

Evaluates trade-offs across competing scientific and engineering metrics:
e.g. Accuracy vs. Model Size vs. Latency vs. Memory vs. Stability.

Provides:
1. Exact non-dominated sorting (Pareto ranking / layers).
2. Multi-objective dominance checking with custom optimization directions.
3. Pareto frontier identification and trade-off reporting.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Set, Tuple


@dataclass(frozen=True)
class Objective:
    """Specification of an optimization objective."""
    name: str
    direction: Literal["maximize", "minimize"] = "maximize"

    def is_better(self, val_a: float, val_b: float) -> bool:
        """Return True if val_a is strictly better than val_b."""
        if self.direction == "maximize":
            return val_a > val_b
        return val_a < val_b

    def is_at_least_as_good(self, val_a: float, val_b: float) -> bool:
        """Return True if val_a is at least as good as val_b (greater/less or equal)."""
        if self.direction == "maximize":
            return val_a >= val_b
        return val_a <= val_b


@dataclass
class CandidatePoint:
    """An evaluated model, configuration, or run candidate."""
    candidate_id: str
    metrics: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    pareto_rank: Optional[int] = None
    is_on_frontier: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "metrics": self.metrics,
            "metadata": self.metadata,
            "pareto_rank": self.pareto_rank,
            "is_on_frontier": self.is_on_frontier,
        }


@dataclass
class ParetoReport:
    """Comprehensive analysis report of the multi-objective Pareto frontier."""
    objectives: List[Objective]
    total_candidates: int
    frontier_candidates: List[CandidatePoint]
    dominated_candidates: List[CandidatePoint]
    layers_count: int
    summary_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "objectives": [{"name": o.name, "direction": o.direction} for o in self.objectives],
            "total_candidates": self.total_candidates,
            "frontier_count": len(self.frontier_candidates),
            "dominated_count": len(self.dominated_candidates),
            "layers_count": self.layers_count,
            "frontier_candidates": [c.to_dict() for c in self.frontier_candidates],
            "dominated_candidates": [c.to_dict() for c in self.dominated_candidates],
            "summary_text": self.summary_text,
        }

    def format_terminal(self) -> str:
        """Render publication-grade ASCII report of Pareto trade-offs."""
        lines: List[str] = []
        border = "=" * 80
        lines.append(border)
        lines.append("HYRA MULTI-OBJECTIVE PARETO FRONTIER REPORT")
        lines.append(border)

        obj_desc = ", ".join(f"{o.name} ({o.direction})" for o in self.objectives)
        lines.append(f"Objectives: {obj_desc}")
        lines.append(f"Total Candidates: {self.total_candidates} | Frontier Points: {len(self.frontier_candidates)}")
        lines.append("-" * 80)

        # Header
        obj_names = [o.name for o in self.objectives]
        header = f"{'Candidate ID':<22} | {'Rank':<6} | " + " | ".join(f"{name:<12}" for name in obj_names)
        lines.append(header)
        lines.append("-" * 80)

        # Frontier candidates first
        for c in self.frontier_candidates:
            vals = " | ".join(f"{c.metrics.get(name, float('nan')):<12.4f}" for name in obj_names)
            lines.append(f"{c.candidate_id:<22} | {'1 (★)':<6} | {vals}")

        # Dominated candidates
        if self.dominated_candidates:
            lines.append("-" * 80)
            lines.append("Dominated Candidates:")
            for c in self.dominated_candidates:
                rank_str = str(c.pareto_rank) if c.pareto_rank else "-"
                vals = " | ".join(f"{c.metrics.get(name, float('nan')):<12.4f}" for name in obj_names)
                lines.append(f"{c.candidate_id:<22} | {rank_str:<6} | {vals}")

        lines.append("-" * 80)
        lines.append(f"Summary: {self.summary_text}")
        lines.append(border)
        return "\n".join(lines)


def dominates(
    cand_a: CandidatePoint,
    cand_b: CandidatePoint,
    objectives: Sequence[Objective],
) -> bool:
    """Determine whether Candidate A Pareto-dominates Candidate B.
    
    A dominates B iff:
    1. A is at least as good as B in all objectives.
    2. A is strictly better than B in at least one objective.
    """
    at_least_as_good_in_all = True
    strictly_better_in_at_least_one = False

    for obj in objectives:
        val_a = cand_a.metrics.get(obj.name)
        val_b = cand_b.metrics.get(obj.name)

        if val_a is None or val_b is None:
            # Cannot establish dominance if a metric is missing
            return False

        if not obj.is_at_least_as_good(val_a, val_b):
            at_least_as_good_in_all = False
            break

        if obj.is_better(val_a, val_b):
            strictly_better_in_at_least_one = True

    return at_least_as_good_in_all and strictly_better_in_at_least_one


def compute_pareto_ranking(
    candidates: Sequence[CandidatePoint],
    objectives: Sequence[Objective],
) -> List[List[CandidatePoint]]:
    """Perform non-dominated sorting across candidates.
    
    Returns:
        List of candidate layers [Rank_1, Rank_2, ...], where Rank_1 is the Pareto frontier.
    """
    if not candidates or not objectives:
        return []

    cands = [CandidatePoint(
        candidate_id=c.candidate_id,
        metrics=dict(c.metrics),
        metadata=dict(c.metadata),
    ) for c in candidates]

    n = len(cands)
    domination_counts = [0] * n
    dominated_sets: List[List[int]] = [[] for _ in range(n)]

    # Compute pairwise dominance
    for i in range(n):
        for j in range(i + 1, n):
            if dominates(cands[i], cands[j], objectives):
                dominated_sets[i].append(j)
                domination_counts[j] += 1
            elif dominates(cands[j], cands[i], objectives):
                dominated_sets[j].append(i)
                domination_counts[i] += 1

    layers: List[List[CandidatePoint]] = []
    current_front_indices = [i for i, count in enumerate(domination_counts) if count == 0]

    current_rank = 1
    while current_front_indices:
        current_layer: List[CandidatePoint] = []
        next_front_indices: List[int] = []

        for i in current_front_indices:
            cands[i].pareto_rank = current_rank
            cands[i].is_on_frontier = (current_rank == 1)
            current_layer.append(cands[i])

            for j in dominated_sets[i]:
                domination_counts[j] -= 1
                if domination_counts[j] == 0:
                    next_front_indices.append(j)

        layers.append(current_layer)
        current_rank += 1
        current_front_indices = next_front_indices

    return layers


def find_pareto_frontier(
    candidates: Sequence[CandidatePoint],
    objectives: Sequence[Objective],
) -> ParetoReport:
    """Identify the non-dominated Pareto frontier and compile comprehensive report."""
    layers = compute_pareto_ranking(candidates, objectives)

    if not layers:
        return ParetoReport(
            objectives=list(objectives),
            total_candidates=0,
            frontier_candidates=[],
            dominated_candidates=[],
            layers_count=0,
            summary_text="No valid candidates provided for Pareto analysis.",
        )

    frontier = layers[0]
    dominated: List[CandidatePoint] = []
    for layer in layers[1:]:
        dominated.extend(layer)

    if len(frontier) == len(candidates):
        summary = (
            f"All {len(candidates)} candidates form a mutually non-dominated trade-off frontier. "
            "No candidate strictly outperforms another across all specified objectives."
        )
    else:
        frontier_ids = ", ".join(c.candidate_id for c in frontier)
        summary = (
            f"Identified {len(frontier)} non-dominated Pareto-optimal architecture(s) "
            f"out of {len(candidates)} evaluated: [{frontier_ids}]."
        )

    return ParetoReport(
        objectives=list(objectives),
        total_candidates=len(candidates),
        frontier_candidates=frontier,
        dominated_candidates=dominated,
        layers_count=len(layers),
        summary_text=summary,
    )

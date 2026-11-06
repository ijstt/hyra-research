"""Hypothesis Directed Acyclic Graph (DAG) and Automated Ablation Engine.

Enables deep scientific inquiry through:
1. Multi-branch hypothesis lineage tracking (parent hypotheses, refinements, counter-hypotheses).
2. Systematic 2^k and leave-one-out ablation variant generation.
3. Causal component attribution analysis: determines whether each mathematical term
   is CRITICAL, MARGINAL, NEUTRAL, or PARASITIC.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


@dataclass
class HypothesisNode:
    node_id: str
    title: str
    parent_id: Optional[str] = None
    relation_type: str = "ROOT"  # ROOT, ABLATION, REFINEMENT, COUNTER_HYPOTHESIS
    components: List[str] = field(default_factory=list)
    status: str = "DRAFT"  # DRAFT, RUNNING, CONFIRMED, FALSIFIED, RETRACTED
    metric_value: Optional[float] = None
    formal_proof_verified: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HypothesisNode":
        return cls(**data)


@dataclass
class AblationVariant:
    variant_id: str
    active_components: List[str]
    removed_components: List[str]
    variant_type: str  # FULL, LEAVE_ONE_OUT, ISOLATED
    metric_value: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CausalComponentAttribution:
    component: str
    marginal_delta: float
    importance_classification: str  # CRITICAL, MARGINAL, NEUTRAL, PARASITIC
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AblationMatrixReport:
    base_experiment_id: str
    full_metric: float
    higher_is_better: bool
    variants: List[AblationVariant] = field(default_factory=list)
    attributions: List[CausalComponentAttribution] = field(default_factory=list)
    critical_components: List[str] = field(default_factory=list)
    parasitic_components: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "base_experiment_id": self.base_experiment_id,
            "full_metric": self.full_metric,
            "higher_is_better": self.higher_is_better,
            "variants": [v.to_dict() for v in self.variants],
            "attributions": [a.to_dict() for a in self.attributions],
            "critical_components": self.critical_components,
            "parasitic_components": self.parasitic_components,
        }

    def format_terminal(self) -> str:
        lines: List[str] = []
        border = "=" * 78
        lines.append(border)
        lines.append(f"HYRA CAUSAL ABLATION MATRIX: {self.base_experiment_id}")
        lines.append(border)
        lines.append(f"Full Model Metric: {self.full_metric:.4f}")
        lines.append("-" * 78)
        lines.append(f"{'Component':<25} | {'Delta M':<10} | {'Classification':<12} | {'Impact'}")
        lines.append("-" * 78)

        for a in self.attributions:
            lines.append(
                f"{a.component:<25} | {a.marginal_delta:+9.4f} | {a.importance_classification:<12} | {a.explanation}"
            )

        lines.append("-" * 78)
        if self.parasitic_components:
            lines.append(f"PARASITIC TERMS DETECTED: {', '.join(self.parasitic_components)}")
            lines.append("Remediation: Prune parasitic components to prevent overfitting and model bloat.")
        else:
            lines.append("All components contribute positively or neutrally to performance.")

        lines.append(border)
        return "\n".join(lines)


class HypothesisDAG:
    """Manages the hierarchical tree/DAG of research hypotheses."""

    def __init__(self):
        self.nodes: Dict[str, HypothesisNode] = {}

    def add_node(self, node: HypothesisNode) -> None:
        if node.node_id in self.nodes:
            raise KeyError(f"Hypothesis node '{node.node_id}' already exists in DAG.")
        if node.parent_id and node.parent_id not in self.nodes:
            raise KeyError(f"Parent hypothesis '{node.parent_id}' not found in DAG.")
        self.nodes[node.node_id] = node

    def get_node(self, node_id: str) -> Optional[HypothesisNode]:
        return self.nodes.get(node_id)

    def get_children(self, parent_id: str) -> List[HypothesisNode]:
        return [node for node in self.nodes.values() if node.parent_id == parent_id]

    def get_lineage(self, node_id: str) -> List[HypothesisNode]:
        """Trace lineage back to root hypothesis."""
        lineage: List[HypothesisNode] = []
        curr = self.get_node(node_id)
        while curr:
            lineage.append(curr)
            curr = self.get_node(curr.parent_id) if curr.parent_id else None
        return list(reversed(lineage))

    def to_dict(self) -> Dict[str, Any]:
        return {"nodes": {k: v.to_dict() for k, v in self.nodes.items()}}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HypothesisDAG":
        dag = cls()
        nodes_dict = data.get("nodes", {})
        # Sort so parents are inserted before children if possible
        for k, v in nodes_dict.items():
            dag.nodes[k] = HypothesisNode.from_dict(v)
        return dag

    def save_to_file(self, file_path: Path) -> None:
        p = Path(file_path).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load_from_file(cls, file_path: Path) -> "HypothesisDAG":
        p = Path(file_path).resolve()
        if not p.is_file():
            return cls()
        data = json.loads(p.read_text(encoding="utf-8"))
        return cls.from_dict(data)


def generate_ablation_variants(
    exp_id: str,
    components: List[str],
    include_isolated: bool = True,
) -> List[AblationVariant]:
    """Generate systematic ablation configurations for a multi-component mathematical formulation.
    
    Args:
        exp_id: Base experiment identifier.
        components: List of independent mathematical or architectural components.
        include_isolated: Whether to include single-component isolation variants.
        
    Returns:
        List of AblationVariant instances.
    """
    if not components:
        raise ValueError("Cannot generate ablations for an empty list of components.")

    variants: List[AblationVariant] = []

    # 1. Full model
    variants.append(
        AblationVariant(
            variant_id=f"{exp_id}_full",
            active_components=list(components),
            removed_components=[],
            variant_type="FULL",
        )
    )

    # 2. Leave-One-Out (LOO) variants: removing component i
    for comp in components:
        active = [c for c in components if c != comp]
        variants.append(
            AblationVariant(
                variant_id=f"{exp_id}_no_{comp}",
                active_components=active,
                removed_components=[comp],
                variant_type="LEAVE_ONE_OUT",
            )
        )

    # 3. Isolated variants (single component only)
    if include_isolated and len(components) > 2:
        for comp in components:
            variants.append(
                AblationVariant(
                    variant_id=f"{exp_id}_only_{comp}",
                    active_components=[comp],
                    removed_components=[c for c in components if c != comp],
                    variant_type="ISOLATED",
                )
            )

    return variants


def compute_ablation_attributions(
    base_experiment_id: str,
    full_metric: float,
    variant_metrics: Dict[str, float],
    higher_is_better: bool = True,
) -> AblationMatrixReport:
    """Compute causal marginal contributions and classify importance of each term.
    
    Args:
        base_experiment_id: Experiment ID.
        full_metric: Metric achieved by the complete model.
        variant_metrics: Mapping from variant_id (e.g. 'EXP_no_jump') to evaluated metric.
        higher_is_better: True if larger metric indicates superior performance.
        
    Returns:
        AblationMatrixReport with classified components.
    """
    attributions: List[CausalComponentAttribution] = []
    critical_components: List[str] = []
    parasitic_components: List[str] = []

    def _extract_component(var_key: str) -> Optional[str]:
        # Exact match with base_experiment_id prefix: e.g. "EXP-001_no_jump" -> "jump"
        pref = f"{base_experiment_id}_no_"
        if var_key.startswith(pref):
            return var_key[len(pref):]

        # Check for full baseline keys to skip
        if var_key.lower() in ("full", f"{base_experiment_id.lower()}_full"):
            return None

        # Strip any variant prefix like "V1_", "variant_1_", or "<exp_id>_"
        cleaned = var_key
        if "_" in cleaned:
            parts = cleaned.split("_", 1)
            if re.match(r"^(v\d+|variant\d*|exp[-\w]*)$", parts[0], re.IGNORECASE):
                cleaned = parts[1]

        if cleaned.startswith("no_"):
            return cleaned[3:]
        if cleaned.startswith("without_"):
            return cleaned[8:]

        # Direct component name fallback
        return cleaned

    for var_id, var_metric in variant_metrics.items():
        comp_name = _extract_component(var_id)
        if comp_name:
            # Marginal delta: how much does removing the component hurt performance?
            if higher_is_better:
                # If higher is better: Delta = Full - Without_i
                # Positive delta means component was helping; negative delta means component was hurting (parasitic)!
                delta = full_metric - var_metric
            else:
                # If lower is better (e.g. loss): Delta = Without_i - Full
                delta = var_metric - full_metric

            if delta > 0.05:
                classification = "CRITICAL"
                explanation = "Indispensable: removal induces significant degradation."
                critical_components.append(comp_name)
            elif delta > 0.01:
                classification = "MARGINAL"
                explanation = "Moderate positive contribution."
            elif delta >= -0.01:
                classification = "NEUTRAL"
                explanation = "Inconsequential: removal has negligible effect."
            else:
                classification = "PARASITIC"
                explanation = "Harmful: model performs strictly better without this component."
                parasitic_components.append(comp_name)

            attributions.append(
                CausalComponentAttribution(
                    component=comp_name,
                    marginal_delta=delta,
                    importance_classification=classification,
                    explanation=explanation,
                )
            )

    # Sort attributions by delta descending
    attributions.sort(key=lambda a: a.marginal_delta, reverse=True)

    return AblationMatrixReport(
        base_experiment_id=base_experiment_id,
        full_metric=full_metric,
        higher_is_better=higher_is_better,
        attributions=attributions,
        critical_components=critical_components,
        parasitic_components=parasitic_components,
    )

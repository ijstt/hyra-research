"""Tests for hyra.hypothesis_dag module."""

from pathlib import Path
import pytest

from hyra.hypothesis_dag import (
    AblationMatrixReport,
    AblationVariant,
    CausalComponentAttribution,
    HypothesisDAG,
    HypothesisNode,
    compute_ablation_attributions,
    generate_ablation_variants,
)


def test_hypothesis_dag_lineage_and_persistence(tmp_path: Path):
    dag = HypothesisDAG()

    # 1. Root hypothesis
    root = HypothesisNode(
        node_id="EXP-001",
        title="Adaptive Threshold SNN",
        relation_type="ROOT",
        components=["jump_diff", "threshold_adapt", "leak"],
        status="CONFIRMED",
    )
    dag.add_node(root)

    # 2. Child ablation
    ablation = HypothesisNode(
        node_id="EXP-001.1",
        title="Ablation: No Jump Diffusion",
        parent_id="EXP-001",
        relation_type="ABLATION",
        components=["threshold_adapt", "leak"],
        status="FALSIFIED",
    )
    dag.add_node(ablation)

    # 3. Child refinement
    refinement = HypothesisNode(
        node_id="EXP-001.2",
        title="Refinement: Quadratic Leak",
        parent_id="EXP-001.1",
        relation_type="REFINEMENT",
        components=["threshold_adapt", "quad_leak"],
    )
    dag.add_node(refinement)

    # Test lineage
    lineage = dag.get_lineage("EXP-001.2")
    assert len(lineage) == 3
    assert lineage[0].node_id == "EXP-001"
    assert lineage[1].node_id == "EXP-001.1"
    assert lineage[2].node_id == "EXP-001.2"

    # Test error when parent does not exist
    with pytest.raises(KeyError, match="Parent hypothesis 'NON_EXISTENT' not found"):
        dag.add_node(HypothesisNode(node_id="BAD", title="Bad", parent_id="NON_EXISTENT"))

    # Test serialization
    dag_file = tmp_path / "hypotheses.json"
    dag.save_to_file(dag_file)
    assert dag_file.is_file()

    loaded_dag = HypothesisDAG.load_from_file(dag_file)
    assert len(loaded_dag.nodes) == 3
    assert loaded_dag.get_node("EXP-001.2").relation_type == "REFINEMENT"


def test_generate_ablation_variants():
    components = ["term_A", "term_B", "term_C"]
    variants = generate_ablation_variants("EXP-100", components, include_isolated=True)

    # 1 full + 3 leave-one-out + 3 isolated = 7 variants
    assert len(variants) == 7

    full = next(v for v in variants if v.variant_type == "FULL")
    assert full.active_components == components

    loo_b = next(v for v in variants if v.variant_id == "EXP-100_no_term_B")
    assert loo_b.active_components == ["term_A", "term_C"]
    assert loo_b.removed_components == ["term_B"]

    with pytest.raises(ValueError, match="empty list"):
        generate_ablation_variants("E", [])


def test_compute_ablation_attributions():
    # Base model F1 score = 0.90
    full_metric = 0.90
    variant_metrics = {
        "EXP-100_no_jump": 0.81,        # Delta = +0.09 -> CRITICAL
        "EXP-100_no_adapt": 0.87,       # Delta = +0.03 -> MARGINAL
        "EXP-100_no_color_noise": 0.90, # Delta = 0.00 -> NEUTRAL
        "EXP-100_no_complex_decay": 0.93, # Delta = -0.03 -> PARASITIC! (Model was better without it!)
    }

    report = compute_ablation_attributions(
        base_experiment_id="EXP-100",
        full_metric=full_metric,
        variant_metrics=variant_metrics,
        higher_is_better=True,
    )

    assert "jump" in report.critical_components
    assert "complex_decay" in report.parasitic_components

    attributions_map = {a.component: a for a in report.attributions}
    assert attributions_map["jump"].importance_classification == "CRITICAL"
    assert attributions_map["adapt"].importance_classification == "MARGINAL"
    assert attributions_map["color_noise"].importance_classification == "NEUTRAL"
    assert attributions_map["complex_decay"].importance_classification == "PARASITIC"

    # Terminal report output verification
    term = report.format_terminal()
    assert "HYRA CAUSAL ABLATION MATRIX: EXP-100" in term
    assert "PARASITIC TERMS DETECTED: complex_decay" in term


def test_compute_ablation_flexible_naming():
    # Test variant keys formatted as V1_no_..., without_..., or raw component name
    full_metric = 1.0
    variant_metrics = {
        "V1_no_residual": 0.85,
        "without_attention": 0.70,
        "normalization": 1.05,  # Parasitic
        "EXP-200_full": 1.0,    # Skipped full baseline
    }

    report = compute_ablation_attributions(
        base_experiment_id="EXP-200",
        full_metric=full_metric,
        variant_metrics=variant_metrics,
        higher_is_better=True,
    )

    comp_names = [a.component for a in report.attributions]
    assert "residual" in comp_names
    assert "attention" in comp_names
    assert "normalization" in comp_names
    assert "attention" in report.critical_components
    assert "normalization" in report.parasitic_components

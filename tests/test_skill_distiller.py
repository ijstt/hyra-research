from hyra.skill_distiller import distill_skill


def test_distill_skill(tmp_path):
    skill_file = distill_skill(
        skill_name="test_fast_integrator",
        description="Fast numerical integrator for stochastic SDEs.",
        usage_guidelines="Use when simulating jump-diffusion processes with dt <= 0.001.",
        python_example="def integrate(x): return x * 2",
        destination_dir=tmp_path,
    )

    assert skill_file.is_file()
    content = skill_file.read_text(encoding="utf-8")
    assert "name: test_fast_integrator" in content
    assert "Fast numerical integrator" in content
    assert "def integrate(x): return x * 2" in content

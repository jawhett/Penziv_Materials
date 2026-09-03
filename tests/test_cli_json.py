import json
import pytest
from click.testing import CliRunner
from penziv_materials.cli.main import evaluate_tea, validate_born, predict_forward

def test_evaluate_tea_json():
    runner = CliRunner()
    result = runner.invoke(evaluate_tea, ['NaCl', '--json'])
    assert result.exit_code == 0
    try:
        data = json.loads(result.output)
    except json.JSONDecodeError:
        pytest.fail(f"Output is not valid JSON: {result.output}")

    expected_keys = [
        "precursor_cost", "LCOS", "sintering_energy", "refining_HHI",
        "critical_minerals", "EPA_hazard_score", "carbon_footprint", "compliance_boolean"
    ]
    for key in expected_keys:
        assert key in data

def test_validate_born_json():
    runner = CliRunner()
    result = runner.invoke(validate_born, ['--c11', '260.0', '--c12', '160.0', '--c44', '110.0', '--system', 'cubic', '--json'])
    assert result.exit_code == 0
    try:
        data = json.loads(result.output)
    except json.JSONDecodeError:
        pytest.fail(f"Output is not valid JSON: {result.output}")

    assert "stability_status" in data
    assert "elastic_constants" in data
    assert "eigenvalues" in data
    assert "pass_fail_conditions" in data
    assert data["stability_status"] is True

def test_predict_forward_json():
    runner = CliRunner()
    result = runner.invoke(predict_forward, ['--material', 'TestMaterial', '--json'])
    assert result.exit_code == 0
    try:
        data = json.loads(result.output)
    except json.JSONDecodeError:
        pytest.fail(f"Output is not valid JSON: {result.output}")

    assert "name" in data
    assert data["name"] == "TestMaterial"
    assert "quantum" in data
    assert "atomistic" in data
    assert "mesoscale" in data
    assert "continuum" in data
    assert "process" in data

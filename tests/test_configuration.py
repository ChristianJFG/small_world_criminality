import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from turing_thesis.config import ExperimentConfig, load_config
from turing_thesis.experiments import run_sweep, sweep_probabilities, historical_p_values
from turing_thesis.__main__ import main
from turing_thesis.provenance import sha256


@pytest.mark.parametrize("kwargs", [
    {"seeds": 0}, {"seeds": True}, {"seed_start": -1}, {"nodes": 3.5},
    {"ring_k": 3}, {"nodes": 4, "ring_k": 4}, {"topologies": []},
    {"topologies": ["ring", "ring"]}, {"nodes": 17, "topologies": ["square"]},
    {"preset": "bad"}, {"format": "jpg"}, {"output": ""},
    {"p_points": 0}, {"p_points": 36}, {"preset": "manuscript", "p_points": 2},
    {"probabilities": {"ring": []}}, {"probabilities": {"ring": [0, 0]}},
    {"probabilities": {"ring": [float("nan")]}},
    {"probabilities": {"ring": [True]}}, {"probabilities": {"ring": [1.1]}},
    {"probabilities": {"unknown": [0.1]}},
])
def test_invalid_settings_fail_early(kwargs):
    with pytest.raises(ValueError):
        ExperimentConfig(**kwargs).validate()


def test_file_precedence_and_paths(tmp_path):
    path = tmp_path / 'experiment.toml'
    path.write_text('preset="manuscript"\nnodes=20\noutput="data"\n')
    config = load_config(path, {"nodes": 40, "seeds": None})
    assert config.nodes == 40
    assert config.seed_count == 10
    assert config.output == str(tmp_path / 'data')
    assert load_config(path, {"output": "cli-data"}).output == 'cli-data'
    path.write_text('seedz=2\n')
    with pytest.raises(ValueError, match='Unknown'):
        load_config(path)


def test_repository_config_matches_historical_defaults():
    config = load_config(Path(__file__).resolve().parents[1] / 'config.toml')
    assert config.seed_count == 2
    for topology in config.topologies:
        np.testing.assert_array_equal(config.resolved()['probabilities'][topology], sweep_probabilities(topology))
        np.testing.assert_array_equal(sweep_probabilities(topology, 'manuscript'), historical_p_values(topology))


def test_sampling_controls_are_effective():
    assert len(sweep_probabilities('square', p_points=2)) == 2
    assert len(sweep_probabilities('ring', p_points=2)) == 6
    with pytest.raises(ValueError):
        historical_p_values('ring', 104)
    with pytest.raises(ValueError):
        run_sweep('square', seeds=0)
    graphs, modes = run_sweep('ring', n=20, seeds=2, seed_start=7, ring_k=6, p_values=[0, 1])
    assert graphs.seed.tolist() == [7, 8, 7, 8]
    assert len(modes) == 80
    assert graphs.iloc[0].n_edges == 60


@pytest.mark.parametrize('topology', ['ring', 'square', 'triangular'])
def test_identical_sweeps_repeat(topology):
    kwargs = dict(topology=topology, n=20, seeds=2, seed_start=11, p_values=[0, .1, 1])
    first = run_sweep(**kwargs)
    second = run_sweep(**kwargs)
    for a, b in zip(first, second):
        pd.testing.assert_frame_equal(a, b, check_exact=True)


def test_cli_replay_and_plot_only_preserve_provenance(tmp_path):
    config = tmp_path / 'run.toml'
    config.write_text('topologies=["ring"]\nnodes=12\nseeds=1\noutput="first"\n[probabilities]\nring=[0.0,0.1,1.0]\n')
    main(['--config', str(config)])
    first = tmp_path / 'first' / 'ring'
    metadata = json.loads((first / 'metadata.json').read_text())
    assert metadata['status'] == 'complete'
    assert metadata['graphs'] == 3 and metadata['modes'] == 36
    assert metadata['config']['seed_values'] == [0]
    assert 'numpy' in metadata['environment']['dependencies']
    for filename, digest in metadata['output_sha256'].items():
        assert sha256(first / filename) == digest
    second = tmp_path / 'second'
    main(['--config', str(first / 'resolved_config.toml'), '--output', str(second)])
    for filename in ('graphs.csv', 'modes.csv'):
        assert (first / filename).read_bytes() == (second / 'ring' / filename).read_bytes()
    original = (first / 'metadata.json').read_bytes()
    main(['--config', str(config), '--plot-only'])
    assert (first / 'metadata.json').read_bytes() == original
    assert (first / 'replot_metadata.json').exists()


def test_failed_run_marks_metadata(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError('test interruption')
    monkeypatch.setattr('turing_thesis.__main__.run_sweep', fail)
    with pytest.raises(RuntimeError, match='test interruption'):
        main(['--topologies', 'ring', '--nodes', '12', '--output', str(tmp_path)])
    assert json.loads((tmp_path / 'ring' / 'metadata.json').read_text())['status'] == 'failed'

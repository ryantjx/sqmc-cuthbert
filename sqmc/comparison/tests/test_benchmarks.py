"""Acceptance checks for shared implementations and equal-budget comparison."""
import argparse
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sqmc.comparison import common, benchmark_qmc as qmc_benchmark
from sqmc.comparison import benchmark_hilbert_sort as hilbert_benchmark
from sqmc.comparison import benchmark_sqmc as paired
from sqmc.qmc.qmc import Sobol, Halton, normal_coordinates
from sqmc.sqmc import sqmc


@pytest.fixture(autouse=True)
def benchmark_precision():
    # Package tests deliberately restore x64=False; benchmarks require float64.
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


@pytest.mark.parametrize("sequence,cls", [("sobol", Sobol), ("halton", Halton)])
@pytest.mark.parametrize("mode", ["fresh"])
def test_qmc_runner_uses_public_sampler(sequence, cls, mode, monkeypatch):
    calls = []
    original = cls.sample

    def sample(self, n, **kwargs):
        calls.append((n, kwargs.get("state")))
        return original(self, n, **kwargs)

    monkeypatch.setattr(cls, "sample", sample)
    key = jax.random.key(5)
    runner = qmc_benchmark.make_runner(sequence, 8, 2, True, mode, jax.devices("cpu")[0], 4)
    actual = runner(jnp.uint32(0), key)
    expected = common.engine(sequence, 2, key if mode == "fresh" else jax.random.key(4), True).sample(8)
    np.testing.assert_array_equal(actual, expected)
    assert calls[0][0] == 8
    assert (calls[0][1] is not None) == (mode == "sample")


def test_hilbert_inputs_come_from_qmc():
    actual = hilbert_benchmark.generate_points("sobol", 2, 16, 7, "normal")
    expected = normal_coordinates(Sobol(2, scramble=True, key=jax.random.key(7), start_index=0).sample(16))
    np.testing.assert_array_equal(actual, expected)
    with pytest.raises(RuntimeError, match="permutation"):
        hilbert_benchmark.validate_permutation(np.zeros(8, dtype=int), 8)


def test_sqmc_uses_shared_filter_hilbert_and_scrambled_sobol(monkeypatch):
    counts = {"filter": 0, "hilbert": 0, "qmc": 0}
    original_build, original_sort, original_sample = sqmc.build_filter, sqmc.hilbert_sort, Sobol.sample

    def build(*args, **kwargs):
        counts["filter"] += 1
        assert isinstance(args[-1], Sobol) and args[-1].scramble
        return original_build(*args, **kwargs)

    def sort(points):
        counts["hilbert"] += 1
        return original_sort(points)

    def sample(self, n, **kwargs):
        counts["qmc"] += 1
        assert self.scramble and self.start_index == 0
        return original_sample(self, n, **kwargs)

    monkeypatch.setattr(sqmc, "build_filter", build)
    monkeypatch.setattr(sqmc, "hilbert_sort", sort)
    monkeypatch.setattr(Sobol, "sample", sample)
    runner = paired.make_runner(16, 2, 3)
    obs = jnp.ones((3, 2))
    means, loglik = runner(obs, jax.random.key(4))
    other, _ = runner(obs, jax.random.key(5))
    np.testing.assert_array_equal(means, runner(obs, jax.random.key(4))[0])
    assert not np.array_equal(means, other)
    assert means.shape == (3, 2) and np.isfinite(float(loglik))
    assert counts["filter"] == 1 and counts["hilbert"] > 0 and counts["qmc"] >= 2


def test_kalman_reference_matches_joint_gaussian():
    mean, variance, likelihood = paired.kalman(np.array([[1., -1.]]))
    np.testing.assert_allclose(mean, [[5/9, -5/9]])
    np.testing.assert_allclose(variance, [[5/9, 5/9]])
    assert likelihood == pytest.approx(-3.0932517270701183)
    y = np.array([[1., -1.], [.2, .7], [.5, -.2]])
    t = np.arange(1, 4)
    cov = 1 + .25 * np.minimum.outer(t, t) + np.eye(3)
    expected = -.5 * (2 * (3*np.log(2*np.pi) + np.linalg.slogdet(cov)[1]) + np.sum(y*np.linalg.solve(cov, y)))
    assert paired.kalman(y)[2] == pytest.approx(expected)


def row(n, seconds, selection, validation):
    return dict(dimension=2, backend="cpu", n=n, median_seconds=seconds, q75_seconds=seconds,
                samples_seconds=[seconds], selection_accuracy={"rmse": selection},
                validation_accuracy={"rmse": validation, "rmse_ci95": [validation*.8, validation*1.2]})


def test_budget_selection_uses_selection_set_not_held_out_error():
    rows = [row(8, .01, .3, .1), row(16, .02, .2, .4), row(32, .04, .1, .05)]
    summary = paired.budget_summary(rows, [2], ["cpu"], [.001, .02])
    assert summary[0]["status"] == "no_measured_configuration_within_budget"
    assert summary[1]["n"] == 16
    assert summary[1]["validation_rmse"] == .4
    assert summary[1]["median_seconds"] <= summary[1]["budget_seconds"]


def test_accuracy_averages_squared_errors_before_square_root():
    records = [dict(dataset=0, normalized_mse=e, loglik_error=0) for e in (1., 9.)]
    summary = paired.summarize_accuracy(records, 100, 3)
    assert summary["rmse"] == pytest.approx(np.sqrt(5))


def test_selection_validation_and_timing_keys_are_distinct():
    keys = [paired.trajectory_key(3, 2, 8, 0, 0, stream) for stream in (10, 20, 30)]
    assert len({tuple(np.asarray(jax.random.key_data(key))) for key in keys}) == 3


def test_failed_run_preserves_logs_and_status(tmp_path):
    output = tmp_path / "run"
    with pytest.raises(RuntimeError, match="failure"):
        with common.run_directory("test", argparse.Namespace(output_dir=output)):
            print("partial progress")
            raise RuntimeError("deliberate failure")
    assert "partial progress" in (output / "logs.txt").read_text()
    assert "deliberate failure" in (output / "logs.txt").read_text()
    assert json.loads((output / "status.json").read_text())["status"] == "failed"


def test_requested_gpu_never_falls_back(monkeypatch):
    def unavailable(backend):
        raise RuntimeError("unavailable")
    monkeypatch.setattr(jax, "devices", unavailable)
    with pytest.raises(RuntimeError, match="Requested gpu is unavailable"):
        common.devices(["gpu"])


def test_sqmc_requires_balanced_particle_counts():
    with pytest.raises(ValueError, match="powers of two"):
        common.validate_grid([2], [6], max_dimension=62, power_two=True)


@pytest.mark.parametrize('sequence', ['sobol', 'halton'])
def test_scipy_fresh_reproducible(sequence):
    run = qmc_benchmark.make_scipy_runner(sequence, 16, 2)
    a, b = run(4), run(5)
    np.testing.assert_array_equal(a, run(4))
    assert not np.array_equal(a, b)
    assert a.dtype == np.float64 and a.shape == (16, 2)
    assert np.isfinite(a).all() and ((a >= 0) & (a < 1)).all()


@pytest.mark.parametrize('argv', [['--no-scramble'], ['--modes', 'sample']])
def test_qmc_rejects_nonfresh(argv):
    with pytest.raises(SystemExit):
        qmc_benchmark.main(argv)


def test_qmc_three_way_pairing():
    base = dict(sequence='sobol', dimension=2, n=16, mode='fresh')
    rows = [base | dict(implementation=i, backend=b, median_seconds=t)
            for i, b, t in [('scipy', 'cpu', 6), ('jax', 'gpu', 2), ('jax', 'cpu', 4)]]
    jax_pairs, scipy_pairs = qmc_benchmark.comparisons(rows)
    assert jax_pairs[0]['cpu_over_gpu'] == 2
    assert {p['jax_backend']: p['scipy_over_jax'] for p in scipy_pairs} == {'cpu': 1.5, 'gpu': 3}
    with pytest.raises(ValueError, match='Duplicate'):
        qmc_benchmark.comparisons(rows + rows[:1])


@pytest.mark.parametrize('sequence', ['sobol', 'halton'])
def test_scipy_constructor_settings(sequence, monkeypatch):
    calls = []
    class Engine:
        def __init__(self, dimension, **kwargs):
            calls.append((dimension, kwargs))
        def random_base2(self, m):
            assert m == 4
            return np.zeros((16, 2))
        def random(self, n, *, workers):
            assert n == 16 and workers == 1
            return np.zeros((16, 2))
    monkeypatch.setattr(qmc_benchmark.scipy_qmc, sequence.title(), Engine)
    run = qmc_benchmark.make_scipy_runner(sequence, 16, 2)
    run(4); run(5)
    assert len(calls) == 2
    assert all(d == 2 and kw['scramble'] is True and kw['optimization'] is None for d, kw in calls)
    if sequence == 'sobol':
        assert all(kw['bits'] == 30 for _, kw in calls)

"""Self-contained CPU regression tests; no proprietary or benchmark data needed."""

import numpy as np
import pytest
import torch

from relstream_ode.config import DEFAULTS, VARIANTS, checkpoint_config, validate_config
from relstream_ode.data import ForecastHistory, HistoryBuilder, TemporalDataset
from relstream_ode.dynamics import JointVectorField
from relstream_ode.evaluation import evaluate, filtered_ranks
from relstream_ode.graph import EventGraph
from relstream_ode.model import RelStreamODE
from relstream_ode.solver import rk4_interval, integrate


def small_config(**kwargs):
    config = dict(DEFAULTS)
    config.update(entity_dim=8, relation_dim=8, time_dim=4,
                  transport_rank=2, initializer_layers=1, pair_hidden_dim=8,
                  relation_hidden_dim=8, transport_hidden_dim=8, dropout=0, **kwargs)
    validate_config(config)
    return config


def tiny_data(times=(2., 3.), test_rows=None):
    data = object.__new__(TemporalDataset)
    data.name, data.entities, data.relations, data.units = 'toy', 6, 1, 1
    data.raw = {
        'train': np.array([[0, 0, 1, 0]], dtype=np.int64),
        'valid': np.array([[0, 0, 2, 1]], dtype=np.int64),
        'test': np.array(test_rows if test_rows is not None
                         else [[0, 0, 3, 2], [1, 0, 4, 3]], dtype=np.int64),
    }
    data.event_times = {
        'train': np.array([0.]), 'valid': np.array([1.]),
        'test': np.asarray(times, dtype=np.float64),
    }
    data._index()
    return data


def graph_edges(graph):
    return set(zip(graph.source.tolist(), graph.relation.tolist(), graph.destination.tolist()))


def test_time_jitter_never_reveals_original_future_events():
    data = tiny_data()
    jittered = data.perturbed(np.array([2., 1.5]))
    history = HistoryBuilder(jittered, 'cpu', 100).build(2.)
    seen = graph_edges(history.initialization)
    for _, _, graph in history.intervals:
        seen |= graph_edges(graph)
    assert (1, 0, 4) not in seen
    assert jittered.event_times['test'][1] == 3.
    assert jittered.query_time_offsets.tolist() == [0., -1.5]


def test_time_jitter_does_not_change_multianswer_filter_group():
    data = tiny_data((2., 2.), [[0, 0, 3, 2], [0, 0, 4, 2]])
    jittered = data.perturbed(np.array([2., 2.1]))
    assert jittered.answers(2.)[(0, 0)] == {3, 4}
    assert data.answers(2.) == jittered.answers(2.)


def test_jitter_evaluation_preserves_targets_and_resets_time_code():
    data = tiny_data((2., 2.), [[0, 0, 3, 2], [0, 0, 4, 2]])
    model = RelStreamODE(6, 1, small_config())
    builder = HistoryBuilder(data, 'cpu', 2)
    orig = evaluate(model, data, builder, 'test', reuse_state=False)
    noisy = data.perturbed(np.array([1.5, 2.4]))
    collector = {}
    check = evaluate(model, noisy, HistoryBuilder(noisy, 'cpu', 2), 'test',
                     collector=collector)
    assert check['queries'] == orig['queries'] == 2
    assert set(collector['row_ids'].tolist()) == {0, 1}
    assert np.all(collector['original_label_days'] == 2.)
    assert np.allclose(sorted(collector['time_code_offsets']), [-0.5, 0.4])
    assert model.field.time_code_shift == 0.


def test_rk4_substeps_configurable_and_exact_constant_derivative():
    def field(t, state, graph):
        return torch.ones_like(state[0]), -torch.ones_like(state[1])
    for n in [1, 2, 3, 5, 8]:
        config = small_config(rk4_substeps=n)
        a, b = rk4_interval(field, (torch.zeros(3), torch.zeros(3)), 1., 4., None, n)
        assert torch.allclose(a, torch.full((3,), 3.), atol=1e-6)
        assert torch.allclose(b, torch.full((3,), -3.), atol=1e-6)
    with pytest.raises(ValueError):
        validate_config({**DEFAULTS, 'rk4_substeps': 0})


def test_four_core_and_new_ablations_have_finite_gradients():
    torch.set_num_threads(1)
    initial = EventGraph([[0, 0, 1], [1, 1, 2]], 6, 2, 'cpu')
    subsequent = EventGraph([[0, 0, 3]], 6, 2, 'cpu')
    history = ForecastHistory(initial, [(0., 1., initial), (1., 2., subsequent)],
                              0., 1., 2.)
    for variant in VARIANTS:
        model = RelStreamODE(6, 2, small_config(variant=variant))
        score = model(history, torch.tensor([[0, 0, 3]]))
        assert score.shape == (1, 6) and torch.isfinite(score).all()
        loss = torch.nn.functional.cross_entropy(score, torch.tensor([3]))
        loss.backward()
        assert any(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in model.parameters() if p.requires_grad)


def test_dense_and_factorized_transport_match_outputs_and_gradients():
    torch.manual_seed(19)
    cfg = small_config()
    graph = EventGraph([[0, 0, 1], [2, 0, 3], [3, 1, 4]], 6, 2, 'cpu')
    dense = JointVectorField(2, cfg).double()
    sparse = JointVectorField(2, {**cfg, 'transport_implementation': 'factorized'}).double()
    sparse.load_state_dict(dense.state_dict())
    h = torch.randn(6, 8, dtype=torch.double)
    r = torch.randn(2, 8, dtype=torch.double)
    output_dense = dense(.31, (h, r), graph)
    output_sparse = sparse(.31, (h, r), graph)
    for a, b in zip(output_dense, output_sparse):
        assert torch.allclose(a, b, rtol=1e-10, atol=1e-10)
    ld = sum(x.square().sum() for x in output_dense)
    ls = sum(x.square().sum() for x in output_sparse)
    ld.backward(); ls.backward()
    for (name_a, a), (name_b, b) in zip(dense.named_parameters(), sparse.named_parameters()):
        assert name_a == name_b
        if a.grad is not None:
            assert torch.allclose(a.grad, b.grad, atol=1e-9, rtol=1e-9), name_a


def test_transport_formula_switch_actually_changes_message_definition():
    torch.manual_seed(44)
    graph = EventGraph([[0, 0, 1], [0, 0, 2]], 4, 1, 'cpu')
    cfg = small_config()
    field = JointVectorField(1, cfg)
    h, r = torch.randn(4, 8), torch.randn(1, 8)
    exact, _ = field(.5, (h, r), graph)
    field.transport_formula = 'legacy'
    legacy, _ = field(.5, (h, r), graph)
    assert not torch.allclose(exact, legacy)


def test_old_checkpoint_configuration_preserves_actual_old_semantics():
    old = {k: v for k, v in DEFAULTS.items()
           if k not in ('history_steps', 'time_coordinate', 'transport_formula',
                        'transport_implementation', 'usage_aggregation')}
    old['history_days'] = 100.
    migrated = checkpoint_config(old)
    assert migrated['history_steps'] == 100.
    assert migrated['time_coordinate'] == 'legacy_days'
    assert migrated['transport_formula'] == 'legacy'
    assert migrated['transport_implementation'] == 'dense'


def test_truth_filter_keeps_target_and_excludes_alternative_correct_tails():
    scores = torch.tensor([[0.2, 0.3, 0.7, 0.8, 0.9]])
    queries = torch.tensor([[0, 0, 3]])
    filtered = filtered_ranks(scores, queries, {(0, 0): {3, 4}})
    assert float(filtered[0]) == 1.0


def test_data_index_preserves_original_target_ids():
    data = tiny_data((3., 2.), [[1, 0, 4, 3], [0, 0, 3, 2]])
    assert data.row_ids['test'].tolist() == [1, 0]
    perturbed = data.perturbed(np.array([1., 2.]))
    assert perturbed.row_ids['test'].tolist() == [1, 0]
    assert perturbed.observation_times.tolist() == data.observation_times.tolist()


def test_exported_baseline_scores_must_cover_original_queries(tmp_path):
    from relstream_ode.baseline_eval import evaluate_score_shards
    data = tiny_data((2., 2.), [[0, 0, 3, 2], [0, 0, 4, 2]])
    data.origin = 0.
    with (tmp_path / 't_2_part_000.npz').open('wb') as stream:
        np.savez_compressed(stream, row_ids=np.array([0, 1]),
                            scores=np.array([[0., 0., 0., 2., 3., 0.],
                                             [0., 0., 0., 3., 2., 0.]], dtype=np.float32))
    result, details, hashes = evaluate_score_shards(data, 'test', tmp_path)
    assert result['queries'] == 2 and result['MRR'] == 1.0
    assert set(details['row_ids']) == {0, 1}
    assert len(hashes) == 1
    with (tmp_path / 't_2_part_000.npz').open('wb') as stream:
        np.savez_compressed(stream, row_ids=np.array([0]),
                            scores=np.zeros((1, 6), dtype=np.float32))
    with pytest.raises(ValueError, match='Missing query IDs'):
        evaluate_score_shards(data, 'test', tmp_path)


def test_zero_perturbation_is_exact_base_evaluation():
    data = tiny_data((2., 2.), [[0, 0, 3, 2], [0, 0, 4, 2]])
    model = RelStreamODE(6, 1, small_config())
    baseline = evaluate(model, data, HistoryBuilder(data, 'cpu', 2))
    same = data.perturbed(data.event_times['test'].copy())
    noisy = evaluate(model, same, HistoryBuilder(same, 'cpu', 2))
    assert noisy['MRR'] == baseline['MRR']
    assert noisy['Hits@10'] == baseline['Hits@10']


def test_gdelt_time_unit_is_15_minutes_and_single_dataset_root_works(tmp_path, monkeypatch):
    import json
    import relstream_ode.data as dataset_module
    # Synthetic manifest is generated in a temporary root; no official data
    # hashes or dataset files are modified.
    monkeypatch.setattr(dataset_module, 'ROOT', tmp_path)
    monkeypatch.setitem(dataset_module.SPECS, 'GDELT', (6, 2, (3, 2, 2), 1440.0))
    source = tmp_path / 'official_like' / 'GDELT'
    source.mkdir(parents=True)
    (source / 'stat.txt').write_text('6 2\n')
    for name, rows in {
        'train': [[0, 0, 1, 0], [0, 1, 2, 15], [1, 0, 3, 30]],
        'valid': [[0, 0, 4, 45], [1, 0, 4, 60]],
        'test': [[0, 1, 3, 75], [1, 1, 5, 90]],
    }.items():
        np.savetxt(source / f'{name}.txt', rows, fmt='%d')
    cache = tmp_path / 'cache'
    manifest = dataset_module.prepare_dataset('GDELT', tmp_path / 'official_like', cache)
    (tmp_path / 'splits').mkdir()
    (tmp_path / 'splits' / 'GDELT.json').write_text(json.dumps(manifest))
    aligned = dataset_module.TemporalDataset('GDELT', tmp_path / 'official_like', cache)
    legacy = dataset_module.TemporalDataset('GDELT', tmp_path / 'official_like', cache,
                                             time_coordinate='legacy_days')
    assert aligned.units == 15.0
    assert np.allclose(aligned.event_times['test'], [5., 6.])
    assert legacy.units == 1440.0
    assert np.allclose(legacy.event_times['test'], [5/96., 6/96.])


def test_integration_uses_configured_number_of_vector_field_calls():
    torch.manual_seed(45)
    graph = EventGraph([[0, 0, 1]], 3, 1, 'cpu')
    state = (torch.randn(3, 8), torch.randn(1, 8))
    field = JointVectorField(1, small_config())
    field.nfe = 0
    integrate(field, state, [(0., 1., graph)], substeps=5)
    assert field.nfe == 20

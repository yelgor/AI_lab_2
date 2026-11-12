import torch

from cvlab.sampling import FlightTimeSampler


def records():
    return [
        {"id": family * 100 + frame, "frame_id": frame,
         "sequence_id": f"uav{family}_clip"}
        for family, count in [(1, 40), (2, 8), (3, 16)]
        for frame in range(count)
    ]


def test_short_prefix_covers_flights_and_time_without_duplicates():
    data = records()
    order = list(FlightTimeSampler(data, torch.Generator().manual_seed(42), bins=4))
    assert sorted(order) == list(range(len(data)))
    first = [data[i] for i in order[:12]]
    for family, count in [(1, 40), (2, 8), (3, 16)]:
        frames = [r['frame_id'] for r in first if r['sequence_id'] == f'uav{family}_clip']
        assert len(frames) == 4
        assert {frame * 4 // count for frame in frames} == {0, 1, 2, 3}


def test_checkpointed_generator_replays_next_epoch_but_epochs_differ():
    data = records()
    generator = torch.Generator().manual_seed(123)
    sampler = FlightTimeSampler(data, generator)
    first = list(sampler)
    state = generator.get_state()
    second = list(sampler)
    restored = torch.Generator()
    restored.set_state(state)
    assert second == list(FlightTimeSampler(data, restored))
    assert first != second


def test_excerpts_share_family_and_non_uav_sequences_stay_separate():
    data = [{"id": i, "sequence_id": seq} for i, seq in enumerate(
        ['uav1_a', 'uav1_b', 'other_a', 'other_b'])]
    sampler = FlightTimeSampler(data, torch.Generator().manual_seed(1))
    assert len(sampler.groups) == 3
    assert sorted(sampler) == list(range(4))


def test_loader_uses_sampler_only_for_training(monkeypatch):
    from cvlab import data

    items = [(i, i, {"id": i}) for i in range(6)]
    monkeypatch.setattr(data, "Frames", lambda *args, **kwargs: items)
    config = {
        "loader": {"workers": 0, "pin_memory": False, "batch_size": 1,
                   "sampling": "flight_time", "temporal_bins": 2},
        "data": {"root": "."}, "model": {"image_size": 32},
    }
    source = records()[:6]
    training = data.make_loader(config, source, True, torch.Generator().manual_seed(2))
    assert isinstance(training.sampler, FlightTimeSampler)
    assert sorted(batch[0][0] for batch in training) == list(range(6))
    validation = data.make_loader(config, source, False)
    assert [batch[0][0] for batch in validation] == list(range(6))

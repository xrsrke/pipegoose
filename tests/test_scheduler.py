import pytest

from pipegoose.scheduler import DetermisticScheduler


@pytest.mark.parametrize(
    "n_microbatches, n_patritions, expected_schedule",
    [
        (1, 1, [[(0, 0)]]),
        (2, 1, [[(0, 0)], [(1, 0)]]),
        (
            4,
            2,
            [
                [(0, 0)],  # noqa
                [(1, 0), (0, 1)],  # noqa
                [(2, 0), (1, 1)],  # noqa
                [(3, 0), (2, 1)],  # noqa
                [(3, 1)],  # noqa
            ],
        ),
    ],
)
def test_determistic_scheduler(n_microbatches, n_patritions, expected_schedule):
    scheduler = DetermisticScheduler()

    assert list(scheduler.generate(n_microbatches=n_microbatches, n_patritions=n_patritions)) == expected_schedule

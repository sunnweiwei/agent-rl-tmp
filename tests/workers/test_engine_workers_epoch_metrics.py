from verl.workers.engine_workers import _append_ppo_epoch_metrics


def test_append_ppo_epoch_metrics_keeps_actor_metrics_separate():
    metrics = {}

    _append_ppo_epoch_metrics(
        metrics,
        {
            "actor/ppo_kl": 0.1,
            "actor/pg_clipfrac": 0.2,
            "actor/pg_loss": 0.3,
            "grad_norm": 0.4,
            "loss": 0.5,
        },
        epoch_index=1,
    )

    assert metrics == {
        "actor/epoch_2/ppo_kl": [0.1],
        "actor/epoch_2/pg_clipfrac": [0.2],
        "actor/epoch_2/pg_loss": [0.3],
        "actor/epoch_2/grad_norm": [0.4],
        "actor/epoch_2/loss": [0.5],
    }


def test_append_ppo_epoch_metrics_ignores_non_actor_output():
    metrics = {}

    _append_ppo_epoch_metrics(metrics, {"critic/vf_loss": 0.2, "grad_norm": 0.3}, epoch_index=0)

    assert metrics == {}

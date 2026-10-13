import torch

from verl.utils.lora_sync_debug import fingerprint_lora_tensors


def test_lora_fingerprint_is_order_independent_and_exact():
    weights = [
        ("model.x.lora_A.default.weight", torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)),
        ("model.x.lora_B.default.weight", torch.tensor([[0.0], [3.0]], dtype=torch.bfloat16)),
    ]

    first = fingerprint_lora_tensors(weights)
    reordered = fingerprint_lora_tensors(reversed(weights))
    changed = fingerprint_lora_tensors([(weights[0][0], weights[0][1]), (weights[1][0], weights[1][1] + 1)])

    assert first == reordered
    assert first["sha256"] != changed["sha256"]
    assert first["tensor_count"] == 2
    assert first["total_numel"] == 4
    assert first["zero_tensors"] == 0


def test_lora_fingerprint_reports_zero_components():
    weights = [
        ("model.x.lora_A.default.weight", torch.ones((2, 2), dtype=torch.bfloat16)),
        ("model.x.lora_B.default.weight", torch.zeros((2, 2), dtype=torch.bfloat16)),
    ]

    result = fingerprint_lora_tensors(weights)

    assert result["zero_tensors"] == 1
    assert result["a_zero_tensors"] == 0
    assert result["b_zero_tensors"] == 1

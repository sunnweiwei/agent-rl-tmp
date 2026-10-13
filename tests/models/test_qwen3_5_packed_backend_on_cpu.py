from verl.models.transformers.qwen3_5 import _call_accepts_kwarg


def test_packed_backend_capability_requires_explicit_parameter():
    def kwargs_only(**kwargs):
        return kwargs

    def explicit(cu_seqlens=None, **kwargs):
        return cu_seqlens, kwargs

    assert not _call_accepts_kwarg(kwargs_only, "cu_seqlens")
    assert _call_accepts_kwarg(explicit, "cu_seqlens")

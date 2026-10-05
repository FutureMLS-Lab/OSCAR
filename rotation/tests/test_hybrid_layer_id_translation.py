"""HybridLinearKVPool must translate sparse global layer ids for the per-layer
geometry accessors, not just the buffer getters.

Qwen3.5 is a hybrid linear-attention model: only 1 in 4 layers has a KV cache
(full-attn layers 3, 7, 11, ...). gemma4 introduced per-layer geometry
accessors that ``dequantize_prefix_kv`` calls; without translation the inner
pool indexes a length-N list with a global id and raises IndexError.
"""

import types

from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool, MHATokenToKVPool

ACCESSORS = ("get_layer_head_num", "get_layer_head_dim", "get_layer_v_head_dim")


def _stub(num_full_layers=12, stride=4):
    """Duck-typed stand-in: global layer 3,7,11,... -> local 0,1,2,..."""
    mapping = {stride - 1 + stride * i: i for i in range(num_full_layers)}
    inner = types.SimpleNamespace(
        _layer_head_num=[4] * num_full_layers,
        _layer_head_dim=[128] * num_full_layers,
        _layer_v_head_dim=[128] * num_full_layers,
    )
    # mimic UnifiedInt2HPKVPool's local indexing (start_layer == 0)
    inner.get_layer_head_num = lambda i: inner._layer_head_num[i]
    inner.get_layer_head_dim = lambda i: inner._layer_head_dim[i]
    inner.get_layer_v_head_dim = lambda i: inner._layer_v_head_dim[i]
    return types.SimpleNamespace(
        full_kv_pool=inner,
        full_attention_layer_id_mapping=mapping,
        start_layer=0,
        _transfer_full_attention_id=HybridLinearKVPool._transfer_full_attention_id.__get__(
            types.SimpleNamespace(full_attention_layer_id_mapping=mapping)
        ),
    )


def test_accessors_defined_on_wrapper():
    for name in ACCESSORS:
        assert name in vars(HybridLinearKVPool), (
            f"{name} must be an explicit override; inheriting __getattr__ hands "
            "back the inner pool's bound method, which sees untranslated ids"
        )


def test_sparse_global_ids_translate():
    s = _stub()
    for name in ACCESSORS:
        fn = getattr(HybridLinearKVPool, name)
        # the last full-attn layer: global 47 -> local 11. Untranslated, 47
        # would run off the end of a 12-entry list.
        assert fn(s, 47) is not None
        assert fn(s, 3) is not None


def test_untranslatable_id_raises_clearly():
    s = _stub()
    for name in ACCESSORS:
        fn = getattr(HybridLinearKVPool, name)
        try:
            fn(s, 4)  # a linear-attn layer has no KV cache
        except ValueError as e:
            assert "not in full attention layers" in str(e)
        else:
            raise AssertionError(f"{name} accepted a linear-attn layer id")


def test_plain_mha_pool_does_not_advertise_accessors():
    # dequantize_prefix_kv branches on hasattr(kv_pool, "get_layer_head_dim");
    # a plain MHA pool must not claim to have it (it has no full_kv_pool).
    for name in ACCESSORS:
        assert name not in vars(MHATokenToKVPool), (
            f"{name} leaked into MHATokenToKVPool -- hasattr() will pick it up "
            "and it would AttributeError on self.full_kv_pool"
        )


PQ_ACCESSORS = ("pq_k_codebook", "pq_k_codebook2", "pq_v_codebook", "get_raw_key_buffer2")


def test_pq_codebook_accessors_translate_sparse_ids():
    """PQ codebooks are per full-attention layer in the inner pool; the wrapper
    must hand it the local index (the novakv arm on Qwen3.5-4B raised
    IndexError from pq_k_codebook with the global id)."""
    stub = _stub()
    inner = stub.full_kv_pool
    books = [f"book{i}" for i in range(12)]
    inner.pq_k_codebook = lambda i: books[i]
    inner.pq_k_codebook2 = lambda i: None
    inner.pq_v_codebook = lambda i: f"v{books[i]}"
    inner.get_raw_key_buffer2 = lambda i: f"codes2-{i}"
    stub._wait_for_layer = lambda layer_id: None
    for name in PQ_ACCESSORS:
        assert name in HybridLinearKVPool.__dict__, f"{name} must be a translated method on the wrapper"
    get = lambda name: HybridLinearKVPool.__dict__[name].__get__(stub)  # noqa: E731
    assert get("pq_k_codebook")(3) == "book0" and get("pq_k_codebook")(11) == "book2"
    assert get("pq_v_codebook")(47) == "vbook11"
    assert get("pq_k_codebook2")(7) is None
    assert get("get_raw_key_buffer2")(7) == "codes2-1"

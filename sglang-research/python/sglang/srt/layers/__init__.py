
# Kimi-K3 imports these as package attributes (`from sglang.srt.layers import
# k3_ar_fusion, ...`), which only resolves if the package binds them.
from sglang.srt.layers import (  # noqa: F401
    k3_ar_fusion,
    k3_gemm_ar,
    k3_sp_collective,
    zero_copy_context,
)

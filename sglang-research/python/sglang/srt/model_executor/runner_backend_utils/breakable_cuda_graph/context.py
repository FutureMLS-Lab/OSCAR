"""Breakable-CUDA-graph context: absent in this fork.

Upstream guards a few fast paths with `not is_in_breakable_cuda_graph()`. This
fork never captures that kind of graph, so the honest answer is False -- the
guarded path is the one that runs.
"""


def is_in_breakable_cuda_graph() -> bool:
    return False

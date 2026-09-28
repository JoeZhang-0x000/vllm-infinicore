"""Ascend-specialized operator adapters (C API bridge, routes, graph ops)."""

SUPPORTED_ROUTES = frozenset(
    {"RMSNorm", "SiluAndMul", "RoPE", "Embedding", "MatMul", "LMHead"}
)
GRAPH_POLICY = "ascend_graph_capturable"

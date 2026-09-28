"""Kunlun operator forwarding."""

SUPPORTED_ROUTES = frozenset({"RoPE", "Embedding", "MatMul", "LMHead"})
GRAPH_POLICY = "stream_bridge_graph_validated"

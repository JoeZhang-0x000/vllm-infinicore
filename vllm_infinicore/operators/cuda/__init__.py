"""NVIDIA CUDA operator forwarding."""

SUPPORTED_ROUTES = frozenset({"RMSNorm", "SiluAndMul", "RoPE", "Embedding", "MatMul", "LMHead"})
GRAPH_POLICY = "stream_bridge_graph_unverified"

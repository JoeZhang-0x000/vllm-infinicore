"""MetaX operator forwarding."""

SUPPORTED_ROUTES = frozenset(
    {"RMSNorm", "SiluAndMul", "RoPE", "Embedding", "MatMul", "LMHead"}
)
GRAPH_POLICY = "stream_bridge_graph_validated"

SUPPORTED_ROUTES = SUPPORTED_ROUTES | frozenset(
    {"StoreKVCache", "PagedAttentionPrefill", "PagedAttentionDecode"}
)

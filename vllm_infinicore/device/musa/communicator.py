"""MUSA communicator adapter for vLLM graph capture."""

from __future__ import annotations

from torch.distributed import ProcessGroup
from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator
from vllm.distributed.device_communicators.xpu_communicator import XpuCommunicator


class InfiniCoreMusaCommunicator(CudaCommunicator, XpuCommunicator):
    """Use MUSA torch.distributed collectives while satisfying vLLM graph checks."""

    def __init__(
        self,
        cpu_group: ProcessGroup,
        device=None,
        device_group: ProcessGroup | None = None,
        unique_name: str = "",
        *args: object,
        **kwargs: object,
    ) -> None:
        del args, kwargs
        XpuCommunicator.__init__(
            self,
            cpu_group=cpu_group,
            device=device,
            device_group=device_group,
            unique_name=unique_name,
        )
        self.ca_comm = None

    def all_reduce(self, input_):
        return XpuCommunicator.all_reduce(self, input_)

    def reduce_scatter(self, input_, dim: int = -1):
        return XpuCommunicator.reduce_scatter(self, input_, dim)

    def reduce_scatterv(self, input_, dim: int = -1, sizes=None):
        return XpuCommunicator.reduce_scatterv(self, input_, dim, sizes)

    def all_gatherv(self, input_, dim: int = 0, sizes=None):
        return XpuCommunicator.all_gatherv(self, input_, dim, sizes)

    def gather(self, input_, dst: int = 0, dim: int = -1):
        return XpuCommunicator.gather(self, input_, dst, dim)

    def broadcast(self, input_, src: int = 0) -> None:
        return XpuCommunicator.broadcast(self, input_, src)

    def dispatch_router_logits(
        self,
        hidden_states,
        router_logits,
        is_sequence_parallel: bool = False,
        extra_tensors=None,
    ):
        return XpuCommunicator.dispatch_router_logits(
            self,
            hidden_states,
            router_logits,
            is_sequence_parallel,
            extra_tensors,
        )

    def dispatch(
        self,
        hidden_states,
        topk_weights,
        topk_ids,
        is_sequence_parallel: bool = False,
        extra_tensors=None,
    ):
        return XpuCommunicator.dispatch(
            self,
            hidden_states,
            topk_weights,
            topk_ids,
            is_sequence_parallel,
            extra_tensors,
        )

    def combine(self, hidden_states, is_sequence_parallel: bool = False):
        return XpuCommunicator.combine(
            self,
            hidden_states,
            is_sequence_parallel,
        )

    def destroy(self) -> None:
        return None

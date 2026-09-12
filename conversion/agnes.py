from __future__ import annotations

import logging

import gguf

from .base import ModelBase
from .qwen import _LinearAttentionVReorderBase, _Qwen35MRopeMixin

logger = logging.getLogger("agnes")


@ModelBase.register("AgnesForConditionalGeneration")
class AgnesTextModel(_Qwen35MRopeMixin, _LinearAttentionVReorderBase):
    """Agnes 3.0: hybrid delta rule / global attention with a second FFN in parallel.

    The attention modules use the same math and the same norm conventions as Qwen3.5, they are only
    named differently (delta_attn / global_attn), so the names are rewritten and the Qwen3.5
    mapping is reused. The second FFN branch is exported as its own tensors.
    """

    model_arch = gguf.MODEL_ARCH.QWEN35

    @classmethod
    def filter_tensors(cls, item):
        name, gen = item
        name = name.replace(".delta_attn.", ".linear_attn.").replace(".global_attn.", ".self_attn.")
        return super().filter_tensors((name, gen))

    def set_gguf_parameters(self):
        # the shared Qwen3.5 parameters expect its layer type names and key names
        layer_types = self.hparams.get("layer_types")
        if layer_types is not None:
            self.hparams["layer_types"] = [
                "linear_attention" if t == "agnes_delta_attention" else "full_attention" for t in layer_types
            ]
        if "full_attention_interval" not in self.hparams and "global_attention_interval" in self.hparams:
            self.hparams["full_attention_interval"] = self.hparams["global_attention_interval"]

        super().set_gguf_parameters()

        n_ff_par = self.hparams.get("parallel_ffn_intermediate_size")
        if n_ff_par:
            logger.info("parallel FFN length: %d", n_ff_par)
            self.gguf_writer.add_feed_forward_parallel_length(int(n_ff_par))

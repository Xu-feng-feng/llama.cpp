from __future__ import annotations

from .base import ModelBase, gguf, logger
from .qwen import Qwen3Model



@ModelBase.register("LincalForCausalLM")
class Lincal3Model(Qwen3Model):

    model_arch = gguf.MODEL_ARCH.LINCAL3

    def set_gguf_parameters(self):
        super().set_gguf_parameters()

        sliding_window = self.hparams["sliding_window"]
        sliding_window_pattern = [
            layer_type == "sliding_attention"
            for layer_type in self.hparams["layer_types"]
        ]

        self.gguf_writer.add_sliding_window(sliding_window)
        self.gguf_writer.add_sliding_window_pattern(sliding_window_pattern)

        logger.info(f"gguf: sliding window = {sliding_window}")
        logger.info(f"gguf: sliding window pattern = {sliding_window_pattern}")

# put it in the lm_eval/models/
from typing import Optional, Union
import torch
import math
import torch.nn.functional as F
from easydict import EasyDict
import lm_eval.models.utils
from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM
import sys
import os
model_root = "/home/xwei/transformers/sequence_models/"
sys.path.insert(0, model_root)
import src.utils.config as util_config
import src.model
import src.task
import src.optim
import src.data  # to load all the things into registries
from src.utils.registry import task_registry
from src.utils import convert_load_ckpt
import hydra
from omegaconf import OmegaConf
prefix_path = "/home/xwei/fake_path/sequence_model/pretrain_1B_100BT_fineweb"
config_path = {
    "rat": "fineweb_edu/rat-xl",
    "rnn": "fineweb_edu/rnn-xl",
    "attention": "fineweb_edu/attention-xl",
    "rat_localattention": "fineweb_edu/rat_localattention_interleave-xl",
    "attention_localattention": "fineweb_edu/attention_localattention_interleave-xl",
}


def load_model_from_hydra_config(model_name, hydra_overrides):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.register_new_resolver("int", int)
    with hydra.initialize(config_path="../../../sequence_models/configs/", version_base=None):
        config = hydra.compose(
            config_name="config",
            overrides=[x for x in hydra_overrides.split(',')]
        )
    config = EasyDict(OmegaConf.to_container(config, resolve=True, throw_on_missing=True))
    return config


@register_model("sequence_llm")
class SequenceLM(HFLM):

    def __init__(self, pretrained="attention", **kwargs): # "attention", "rnn_pscan", "hybrid_rnn_l256"
        self.hydra_overrides = kwargs.pop("hydra_overrides")
        super().__init__(
            pretrained=pretrained,
            backend="causal",
            tokenizer="huggyllama/llama-7b",
            max_length=4096,
            **kwargs,
        )

    def _get_config(
        self,
        pretrained: str,
        **kwargs,
    ) -> None:
        self._config = load_model_from_hydra_config(pretrained, self.hydra_overrides)

    def _create_model(
        self,
        pretrained: str,
        dtype: Optional[Union[str, torch.dtype]] = "float16",
        # no `parallelize=True` options
        # no PEFT and quantization options
        # Mamba does not support arbitrary HF from_pretrained() args
        **kwargs,
    ) -> None:
        torch.serialization.add_safe_globals([EasyDict])
        self._model = (util_config.instantiate(task_registry, 
                                        config=self._config.task,
                                        model_config=self._config.model, device="cuda", dtype=torch.float32))
        if self._config.trainer.pretrained_path is not None:
            convert_load_ckpt.convert(self._model, self._config.trainer.pretrained_path)
        self._model = self._model.to("cuda").to(torch.float32)
        self._model = torch.compile(self._model)
        self._model.eval()
        print(self._model)
        dtype = torch.float16 if dtype == "auto" else lm_eval.models.utils.get_dtype(dtype),

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    def tok_encode(self, string: str):
        # note that add special tokens == False, because we don't add bos in the training
        encoding = self.tokenizer.encode(string, add_special_tokens=False) # no <s> at the beginning
        return encoding

    # TODO: tok_encode_batch and generate
    def _model_call(self, inps):
        """
        decoder-only model
        """
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                return self.model(inps)

    # def _model_generate(self, context, max_length, eos_token_id):
    #     pass

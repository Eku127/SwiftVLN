"""
Training entry point for Uni-NaVid fine-tuning on SatNav data.

Does NOT modify the original Uni-NaVid repo.  Instead it:
  1. Adds Uni-NaVid repo to sys.path
  2. Monkey-patches from_pretrained to enable flash_attention_2
  3. Monkey-patches make_supervised_data_module() to use SatNavUniNaVidDataset
  4. Calls the original train() function

Run via:  deepspeed baseline/uninavid/src/train_satnav.py  [args ...]
Or via:   bash baseline/uninavid/scripts/train_satnav.sh
"""

import os
import sys
from typing import Dict

import torch

# ------------------------------------------------------------------
# 1. Make Uni-NaVid and this baseline importable
# ------------------------------------------------------------------
_UNINAVID_ROOT = "/mnt/data1/home/jiangjiajun/workspace/Uni-NaVid"
if _UNINAVID_ROOT not in sys.path:
    sys.path.insert(0, _UNINAVID_ROOT)

_BASELINE_SRC = os.path.dirname(os.path.abspath(__file__))
if _BASELINE_SRC not in sys.path:
    sys.path.insert(0, _BASELINE_SRC)

# ------------------------------------------------------------------
# 2. Flash-attention patch (via transformers 4.34 native support)
#
# The original Uni-NaVid patch (llama_flash_attn_monkey_patch.py) was
# written for flash_attn 1.x (unpad_input returns 4 values).  Our env
# has flash_attn 2.x (returns 5 values), so that old patch crashes.
#
# Instead we use the transformers 4.34 built-in path:
#   from_pretrained(..., use_flash_attention_2=True)
# which selects LlamaFlashAttention2 internally and is fully compatible
# with flash_attn 2.x.  We monkey-patch LlavaLlamaAttForCausalLM so
# the original train() picks it up transparently.
# ------------------------------------------------------------------
from uninavid.model.language_model.llava_llama_vid import LlavaLlamaAttForCausalLM as _LlavaModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.trainer import is_sagemaker_mp_enabled

_orig_from_pretrained = _LlavaModel.from_pretrained
_orig_forward = _LlavaModel.forward
_USE_FLASH_ATTN = os.environ.get("UNINAVID_USE_FLASH_ATTN", "1").lower() not in {"0", "false", "no"}
_DEBUG_LOSS = os.environ.get("UNINAVID_DEBUG_LOSS", "0").lower() not in {"0", "false", "no"}
_DEBUG_RANK = os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0"))

@classmethod
def _flash_attn_from_pretrained(cls, *args, **kwargs):
    if _USE_FLASH_ATTN:
        kwargs.setdefault("use_flash_attention_2", True)
    return _orig_from_pretrained(*args, **kwargs)

_LlavaModel.from_pretrained = _flash_attn_from_pretrained


def _debug_forward(
    self,
    input_ids=None,
    attention_mask=None,
    past_key_values=None,
    inputs_embeds=None,
    labels=None,
    use_cache=None,
    output_attentions=None,
    output_hidden_states=None,
    images=None,
    prompts=None,
    return_dict=None,
):
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if not self.training:
        if images[0].device != self.device:
            images[0] = images[0].to(device=self.device)
        if input_ids.device != self.device:
            input_ids = input_ids.to(device=self.device)

    input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(
        input_ids, attention_mask, past_key_values, labels, images, prompts=prompts
    )

    if _DEBUG_LOSS and labels is not None:
        valid_after_mm = int((labels != -100).sum().item())
        print(f"[loss-debug][rank={_DEBUG_RANK}] valid_labels_after_mm={valid_after_mm}", flush=True)

    torch.cuda.empty_cache()

    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    hidden_states = outputs[0]
    logits = self.lm_head(hidden_states)

    loss = None
    if labels is not None:
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        valid_shift = int((shift_labels != -100).sum().item())
        loss_fct = torch.nn.CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1).to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

        if _DEBUG_LOSS:
            logits_finite = bool(torch.isfinite(shift_logits).all().item())
            loss_finite = bool(torch.isfinite(loss).item())
            print(
                f"[loss-debug][rank={_DEBUG_RANK}] valid_shift_labels={valid_shift} "
                f"logits_finite={logits_finite} loss={float(loss.detach().float().cpu())} "
                f"loss_finite={loss_finite}",
                flush=True,
            )
            if not loss_finite:
                raise RuntimeError(
                    f"Non-finite loss detected on rank={_DEBUG_RANK}: "
                    f"valid_shift_labels={valid_shift}, logits_finite={logits_finite}"
                )

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


if _DEBUG_LOSS:
    _LlavaModel.forward = _debug_forward


def _debug_training_step(self, model, inputs):
    model.train()
    inputs = self._prepare_inputs(inputs)

    raw_valid_labels = None
    if "labels" in inputs and inputs["labels"] is not None:
        raw_valid_labels = int((inputs["labels"] != -100).sum().item())

    if is_sagemaker_mp_enabled():
        return _orig_training_step(self, model, inputs)

    with self.compute_loss_context_manager():
        loss = self.compute_loss(model, inputs)

    if self.args.n_gpu > 1:
        loss = loss.mean()

    loss_finite = bool(torch.isfinite(loss).item())
    print(
        f"[loss-debug][rank={_DEBUG_RANK}] raw_valid_labels={raw_valid_labels} "
        f"loss_before_backward={float(loss.detach().float().cpu())} loss_finite={loss_finite}",
        flush=True,
    )
    if not loss_finite:
        raise RuntimeError(
            f"Non-finite loss before backward on rank={_DEBUG_RANK}, raw_valid_labels={raw_valid_labels}"
        )

    if self.do_grad_scaling:
        self.scaler.scale(loss).backward()
    elif self.use_apex:
        from apex import amp

        with amp.scale_loss(loss, self.optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        self.accelerator.backward(loss)

    return loss.detach() / self.args.gradient_accumulation_steps


# ------------------------------------------------------------------
# 3. Monkey-patch the data module builder
# ------------------------------------------------------------------
import uninavid.train.train as _train_mod
from dataset.satnav_dataset import SatNavUniNaVidDataset
from uninavid.train.llava_trainer import LLaVATrainer as _LLaVATrainer
from transformers.trainer import is_torch_tpu_available

_orig_training_step = _LLaVATrainer.training_step

if _DEBUG_LOSS:
    _LLaVATrainer.training_step = _debug_training_step


def _make_satnav_data_module(tokenizer, data_args):
    """Drop-in replacement that loads SatNav windowed data."""
    train_dataset = SatNavUniNaVidDataset(
        data_path=data_args.data_path,
        tokenizer=tokenizer,
        data_args=data_args,
    )
    data_collator = _train_mod.DataCollatorForSupervisedDataset(
        tokenizer=tokenizer,
    )
    return dict(
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
    )


_train_mod.make_supervised_data_module = _make_satnav_data_module


def _high_precision_maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
    """Preserve the original Trainer behavior but log an unrounded SatNav loss."""
    if self.control.should_log:
        if is_torch_tpu_available():
            import torch_xla.core.xla_model as xm
            xm.mark_step()

        logs: Dict[str, float] = {}
        tr_loss_scalar = self._nested_gather(tr_loss).mean().item()
        tr_loss -= tr_loss

        loss_value = tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged)
        logs["loss"] = round(loss_value, 4)
        logs["loss_raw"] = float(f"{loss_value:.8f}")
        logs["learning_rate"] = self._get_learning_rate()

        self._total_loss_scalar += tr_loss_scalar
        self._globalstep_last_logged = self.state.global_step
        self.store_flos()
        self.log(logs)

    metrics = None
    if self.control.should_evaluate:
        if isinstance(self.eval_dataset, dict):
            metrics = {}
            for eval_dataset_name, eval_dataset in self.eval_dataset.items():
                dataset_metrics = self.evaluate(
                    eval_dataset=eval_dataset,
                    ignore_keys=ignore_keys_for_eval,
                    metric_key_prefix=f"eval_{eval_dataset_name}",
                )
                metrics.update(dataset_metrics)
        else:
            metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
        self._report_to_hp_search(trial, self.state.global_step, metrics)

        if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            self.lr_scheduler.step(metrics[metric_to_check])

    if self.control.should_save:
        self._save_checkpoint(model, trial, metrics=metrics)
        self.control = self.callback_handler.on_save(self.args, self.state, self.control)


_LLaVATrainer._maybe_log_save_evaluate = _high_precision_maybe_log_save_evaluate

# ------------------------------------------------------------------
# 4. Run training (all model/trainer logic from original repo)
# ------------------------------------------------------------------
from uninavid.train.train import train

if __name__ == "__main__":
    train()

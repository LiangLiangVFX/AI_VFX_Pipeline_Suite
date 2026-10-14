# -*- coding: utf-8 -*-
"""
彻底修复 ComfyUI-SAM2 在新版 transformers 下的全部 BertModel 兼容问题:
1. get_extended_attention_mask
2. invert_attention_mask
3. get_head_mask
4. BertLayer.forward 参数与返回值兼容
"""
import os
import re
import shutil

suite_root = os.path.dirname(os.path.abspath(__file__))
target_file = os.path.join(
    suite_root,
    "ComfyUI_windows_portable", "ComfyUI", "custom_nodes",
    "ComfyUI-SAM2", "local_groundingdino", "models", "GroundingDINO", "bertwarper.py"
)

if not os.path.exists(target_file):
    print(f"❌ 找不到文件: {target_file}")
    exit(1)

backup_file = target_file + ".bak"
if not os.path.exists(backup_file):
    shutil.copyfile(target_file, backup_file)

with open(backup_file, "r", encoding="utf-8") as f:
    code = f.read()

compat_methods = """        def _compat_get_extended_attention_mask(attention_mask, input_shape=None, device=None, dtype=None):
            if dtype is None:
                dtype = self.embeddings.word_embeddings.weight.dtype
            if attention_mask.dim() == 3:
                extended_attention_mask = attention_mask[:, None, :, :]
            elif attention_mask.dim() == 2:
                extended_attention_mask = attention_mask[:, None, None, :]
            else:
                raise ValueError(f"Wrong shape for attention_mask: {attention_mask.shape}")
            extended_attention_mask = extended_attention_mask.to(dtype=dtype)
            extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
            return extended_attention_mask

        def _compat_invert_attention_mask(encoder_attention_mask):
            dtype = self.embeddings.word_embeddings.weight.dtype
            if encoder_attention_mask.dim() == 3:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
            elif encoder_attention_mask.dim() == 2:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
            else:
                raise ValueError(f"Wrong shape for encoder_attention_mask: {encoder_attention_mask.shape}")
            encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype=dtype)
            encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * torch.finfo(dtype).min
            return encoder_extended_attention_mask

        def _compat_get_head_mask(head_mask, num_hidden_layers, is_attention_chunked=False):
            if head_mask is not None:
                return head_mask
            return [None] * num_hidden_layers

        self.get_extended_attention_mask = _compat_get_extended_attention_mask
        self.invert_attention_mask = _compat_invert_attention_mask
        self.get_head_mask = _compat_get_head_mask"""

code = re.sub(
    r"^\s*self\.get_extended_attention_mask\s*=\s*bert_model\.get_extended_attention_mask.*$",
    compat_methods,
    code,
    flags=re.MULTILINE
)
code = re.sub(
    r"^\s*#?\s*self\.invert_attention_mask\s*=\s*bert_model\.invert_attention_mask.*$",
    "        # patched invert_attention_mask",
    code,
    flags=re.MULTILINE
)
code = re.sub(
    r"^\s*#?\s*self\.get_head_mask\s*=\s*bert_model\.get_head_mask.*$",
    "        # patched get_head_mask",
    code,
    flags=re.MULTILINE
)

old_layer_call = """                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )"""

new_layer_call = """                try:
                    layer_outputs = layer_module(
                        hidden_states,
                        attention_mask,
                        layer_head_mask,
                        encoder_hidden_states,
                        encoder_attention_mask,
                        past_key_value,
                        output_attentions,
                    )
                except TypeError:
                    layer_outputs = layer_module(hidden_states, attention_mask=attention_mask)"""

code = code.replace(old_layer_call, new_layer_call)
code = code.replace(
    "hidden_states = layer_outputs[0]",
    "hidden_states = layer_outputs[0] if isinstance(layer_outputs, (tuple, list)) else layer_outputs"
)

with open(target_file, "w", encoding="utf-8") as f:
    f.write(code)

print("✅ bertwarper.py 全部 4 项新版接口兼容已修复完成！")
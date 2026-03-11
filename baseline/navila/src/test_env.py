"""
NaVILA Baseline 环境组件测试脚本
验证 navila-baseline conda 环境中所有训练必要组件是否可正常导入和运行
"""
import sys
import traceback

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
WARN = "\033[93m[WARN]\033[0m"
INFO = "\033[94m[INFO]\033[0m"

results = []


def test(name, fn):
    try:
        info = fn()
        msg = f"{PASS} {name}"
        if info:
            msg += f"  →  {info}"
        print(msg)
        results.append((name, True, info or ""))
    except Exception as e:
        print(f"{FAIL} {name}")
        print(f"       {type(e).__name__}: {e}")
        traceback.print_exc()
        results.append((name, False, str(e)))


# ─────────────────────────────────────────────
# 1. 基础环境
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  [1] 基础运行环境")
print("=" * 60)


def t_python():
    v = sys.version
    assert sys.version_info >= (3, 10), f"Python >= 3.10 required, got {v}"
    return v.split()[0]


test("Python 3.10+", t_python)


def t_torch():
    import torch
    assert "2.3" in torch.__version__, f"Expected torch 2.3.x, got {torch.__version__}"
    return torch.__version__


test("PyTorch 2.3.x", t_torch)


def t_cuda():
    import torch
    assert torch.cuda.is_available(), "CUDA not available"
    n = torch.cuda.device_count()
    name = torch.cuda.get_device_name(0)
    return f"{n} GPU(s), device[0]={name}"


test("CUDA available", t_cuda)


def t_cuda_op():
    import torch
    a = torch.randn(128, 128, device="cuda")
    b = torch.randn(128, 128, device="cuda")
    c = torch.matmul(a, b)
    assert c.shape == (128, 128)
    return f"matmul (128x128) on cuda OK, dtype={c.dtype}"


test("CUDA tensor matmul", t_cuda_op)


def t_bf16():
    import torch
    a = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    b = torch.matmul(a, a)
    assert b.dtype == torch.bfloat16
    return "bfloat16 matmul OK"


test("BF16 CUDA op", t_bf16)

# ─────────────────────────────────────────────
# 2. FlashAttention
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  [2] FlashAttention")
print("=" * 60)


def t_flash_import():
    import flash_attn
    return flash_attn.__version__


test("flash_attn import", t_flash_import)


def t_flash_func():
    import torch
    from flash_attn import flash_attn_func
    B, S, H, D = 2, 64, 8, 64
    q = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
    out = flash_attn_func(q, k, v)
    assert out.shape == (B, S, H, D)
    return f"flash_attn_func OK, shape={out.shape}"


test("flash_attn_func forward", t_flash_func)


def t_flash_varlen():
    from flash_attn.flash_attn_interface import flash_attn_varlen_func
    return "flash_attn_varlen_func importable"


test("flash_attn_varlen_func", t_flash_varlen)

# ─────────────────────────────────────────────
# 3. 训练框架
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  [3] 训练框架")
print("=" * 60)


def t_transformers():
    import transformers
    assert transformers.__version__ == "4.37.2", f"Expected 4.37.2, got {transformers.__version__}"
    return transformers.__version__


test("transformers 4.37.2", t_transformers)


def t_transformers_patch():
    # 检查补丁是否生效（NaVILA 在 modeling_llama.py 中加了自定义 patch）
    from transformers.models.llama import modeling_llama
    import inspect
    src = inspect.getfile(modeling_llama)
    # 检查文件是否来自 site-packages (而非 NaVILA 源码目录)
    assert "site-packages" in src, f"Unexpected path: {src}"
    return f"modeling_llama from: {src.split('site-packages')[-1]}"


test("transformers LLaMA patch", t_transformers_patch)


def t_deepspeed():
    import deepspeed
    assert deepspeed.__version__ == "0.9.5", f"Expected 0.9.5, got {deepspeed.__version__}"
    return deepspeed.__version__


test("deepspeed 0.9.5", t_deepspeed)


def t_deepspeed_patch():
    from deepspeed.runtime.zero import mics
    import inspect
    src = inspect.getfile(mics)
    assert "site-packages" in src
    return f"deepspeed mics patch applied"


test("deepspeed mics patch", t_deepspeed_patch)


def t_accelerate():
    import accelerate
    return accelerate.__version__


test("accelerate", t_accelerate)


def t_peft():
    import peft
    return peft.__version__


test("peft", t_peft)


def t_bitsandbytes():
    import bitsandbytes as bnb
    import importlib.metadata
    ver = importlib.metadata.version("bitsandbytes")
    return ver


test("bitsandbytes", t_bitsandbytes)


def t_wandb():
    import wandb
    return wandb.__version__


test("wandb", t_wandb)

# ─────────────────────────────────────────────
# 4. VILA/LLaVA 框架核心模块
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  [4] VILA/LLaVA 框架核心模块")
print("=" * 60)


def t_llava_pkg():
    import llava
    import inspect
    src = inspect.getfile(llava)
    return f"from {src}"


test("llava package import", t_llava_pkg)


def t_llava_constants():
    from llava import constants
    return "constants OK"


test("llava.constants", t_llava_constants)


def t_llava_conversation():
    from llava.conversation import conv_templates
    assert "llama_3" in conv_templates, "llama_3 conv template not found"
    return f"conv_templates has 'llama_3'"


test("llava.conversation (llama_3 template)", t_llava_conversation)


def t_llava_mm_utils():
    from llava.mm_utils import tokenizer_image_token, process_images
    return "tokenizer_image_token, process_images OK"


test("llava.mm_utils", t_llava_mm_utils)

# ─────────────────────────────────────────────
# 5. 模型组件
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  [5] 模型组件")
print("=" * 60)


def t_llava_arch():
    from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
    return "LlavaMetaModel, LlavaMetaForCausalLM OK"


test("llava.model.llava_arch", t_llava_arch)


def t_llava_llama():
    from llava.model.language_model.llava_llama import LlavaLlamaModel, LlavaLlamaConfig
    return "LlavaLlamaModel, LlavaLlamaConfig OK"


test("llava.model.language_model.llava_llama", t_llava_llama)


def t_siglip_encoder():
    from llava.model.multimodal_encoder.siglip_encoder import SiglipVisionTower, SiglipVisionTowerS2
    return "SiglipVisionTower, SiglipVisionTowerS2 OK"


test("llava.model.multimodal_encoder.siglip_encoder", t_siglip_encoder)


def t_vision_encoder():
    from llava.model.multimodal_encoder.vision_encoder import VisionTower
    return "VisionTower OK"


test("llava.model.multimodal_encoder.vision_encoder", t_vision_encoder)


def t_mm_projector():
    from llava.model.multimodal_projector.builder import build_mm_projector
    return "build_mm_projector OK"


test("llava.model.multimodal_projector.builder", t_mm_projector)


def t_model_builder():
    from llava.model.builder import load_pretrained_model
    return "load_pretrained_model OK"


test("llava.model.builder", t_model_builder)

# ─────────────────────────────────────────────
# 6. 训练模块
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  [6] 训练模块")
print("=" * 60)


def t_train_args():
    from llava.train.args import TrainingArguments, DataArguments, ModelArguments
    return "TrainingArguments, DataArguments, ModelArguments OK"


test("llava.train.args", t_train_args)


def t_llava_trainer():
    from llava.train.llava_trainer import LLaVATrainer
    return "LLaVATrainer OK"


test("llava.train.llava_trainer", t_llava_trainer)


def t_train_utils():
    from llava.train.utils import prepare_config_for_training, get_checkpoint_path
    return "prepare_config_for_training, get_checkpoint_path OK"


test("llava.train.utils", t_train_utils)

# ─────────────────────────────────────────────
# 7. 数据模块
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  [7] 数据模块")
print("=" * 60)


def t_llava_data():
    from llava.data import datasets_mixture
    return "datasets_mixture OK"


test("llava.data.datasets_mixture", t_llava_data)


def t_datasets():
    import datasets
    return datasets.__version__


test("HuggingFace datasets", t_datasets)


def t_decord():
    import decord
    return decord.__version__


test("decord (video decoding)", t_decord)

# ─────────────────────────────────────────────
# 8. 视觉和工具库
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  [8] 视觉和工具库")
print("=" * 60)


def t_timm():
    import timm
    return timm.__version__


test("timm", t_timm)


def t_einops():
    import einops
    return einops.__version__


test("einops", t_einops)


def t_opencv():
    import cv2
    return cv2.__version__


test("opencv-python", t_opencv)


def t_pytorchvideo():
    import pytorchvideo
    return "pytorchvideo OK"


test("pytorchvideo", t_pytorchvideo)


def t_ninja():
    import ninja
    return "ninja OK"


test("ninja (CUDA extension build)", t_ninja)

# ─────────────────────────────────────────────
# 汇总
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  汇总")
print("=" * 60)
passed = [r for r in results if r[1]]
failed = [r for r in results if not r[1]]
print(f"  PASS: {len(passed)}/{len(results)}")
if failed:
    print(f"  FAIL: {len(failed)}")
    for name, _, err in failed:
        print(f"    - {name}: {err[:80]}")
print("=" * 60)
sys.exit(0 if not failed else 1)

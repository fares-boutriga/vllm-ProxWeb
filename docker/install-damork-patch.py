#!/usr/bin/env python3
"""Install Damork Python aliases into the base vLLM image."""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path


PATCH_ROOT = Path("/tmp/damork-patch/vllm")

DAMORK_CONFIG_ALIAS = '''


class DamorkTextConfig(Qwen3_5TextConfig):
    """Damork-branded text config alias for Qwen3.5 text checkpoints."""

    model_type = "damork_text"


class DamorkVisionConfig(Qwen3_5VisionConfig):
    """Damork-branded vision config alias for Qwen3.5 vision checkpoints."""

    model_type = "damork"


class DamorkConfig(Qwen3_5Config):
    """Damork-branded config alias for Qwen3.5-compatible checkpoints."""

    model_type = "damork"
    sub_configs = {
        "vision_config": DamorkVisionConfig,
        "text_config": DamorkTextConfig,
    }

    def __init__(self, text_config=None, vision_config=None, **kwargs):
        if isinstance(text_config, dict):
            text_config = dict(text_config)
            text_config.pop("model_type", None)

        if isinstance(vision_config, dict):
            vision_config = dict(vision_config)
            vision_config.pop("model_type", None)

        super().__init__(
            text_config=text_config,
            vision_config=vision_config,
            **kwargs,
        )

    def get_text_config(self, decoder: bool = False):
        return self.text_config


try:
    __all__.extend([
        "DamorkConfig",
        "DamorkTextConfig",
        "DamorkVisionConfig",
    ])
except NameError:
    __all__ = [
        "DamorkConfig",
        "DamorkTextConfig",
        "DamorkVisionConfig",
    ]
'''


def find_vllm_package_root() -> Path:
    spec = importlib.util.find_spec("vllm")
    if spec is None or spec.origin is None:
        raise RuntimeError("Could not find the installed vLLM package")

    return Path(spec.origin).resolve().parent


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def insert_after_once(path: Path, marker: str, insertion: str, sentinel: str) -> None:
    text = read_text(path)
    if sentinel in text:
        print(f"already patched {path}")
        return
    if marker not in text:
        raise RuntimeError(f"Could not find patch marker in {path}: {marker!r}")
    write_text(path, text.replace(marker, marker + insertion, 1))
    print(f"patched {path}")


def replace_once(path: Path, old: str, new: str, sentinel: str) -> None:
    text = read_text(path)
    if sentinel in text:
        print(f"already patched {path}")
        return
    if old not in text:
        raise RuntimeError(f"Could not find patch target in {path}: {old!r}")
    write_text(path, text.replace(old, new, 1))
    print(f"patched {path}")


def append_once(path: Path, block: str, sentinel: str) -> None:
    text = read_text(path)
    if sentinel in text:
        print(f"already patched {path}")
        return
    write_text(path, text.rstrip() + block + "\n")
    print(f"patched {path}")


def copy_processor(package_root: Path) -> None:
    source = PATCH_ROOT / "transformers_utils/processors/damork_vl.py"
    target = package_root / "transformers_utils/processors/damork_vl.py"
    if not source.is_file():
        raise FileNotFoundError(f"Missing Damork processor patch: {source}")
    shutil.copy2(source, target)
    print(f"installed {target}")


def patch_installed_vllm(package_root: Path) -> None:
    insert_after_once(
        package_root / "model_executor/models/registry.py",
        '    "Qwen3_5ForConditionalGeneration": ("qwen3_5", "Qwen3_5ForConditionalGeneration"),\n',
        '    "DamorkForConditionalGeneration": ("qwen3_5", "Qwen3_5ForConditionalGeneration"),\n',
        '"DamorkForConditionalGeneration"',
    )

    insert_after_once(
        package_root / "model_executor/models/config.py",
        '    "ColQwen3_5": Qwen3_5ForConditionalGenerationConfig,\n',
        '    "DamorkForConditionalGeneration": Qwen3_5ForConditionalGenerationConfig,\n',
        '"DamorkForConditionalGeneration"',
    )

    insert_after_once(
        package_root / "transformers_utils/config.py",
        '    qwen3_vl_nemotron_embed="Qwen3VLNemotronEmbedConfig",\n',
        '    damork="DamorkConfig",\n',
        'damork="DamorkConfig"',
    )

    insert_after_once(
        package_root / "transformers_utils/configs/__init__.py",
        '    "Qwen3VLNemotronEmbedConfig": "vllm.transformers_utils.configs.colqwen3",\n',
        '    "DamorkConfig": "vllm.transformers_utils.configs.qwen3_5",\n',
        '"DamorkConfig": "vllm.transformers_utils.configs.qwen3_5"',
    )

    insert_after_once(
        package_root / "transformers_utils/configs/__init__.py",
        '    "Qwen3VLNemotronEmbedConfig",\n',
        '    "DamorkConfig",\n',
        '    "DamorkConfig",\n',
    )

    append_once(
        package_root / "transformers_utils/configs/qwen3_5.py",
        DAMORK_CONFIG_ALIAS,
        "class DamorkConfig",
    )

    replace_once(
        package_root / "model_executor/models/qwen3_5.py",
        '        elif config.model_type == "qwen3_5_text":\n',
        '        elif config.model_type in ("qwen3_5_text", "damork_text"):\n',
        '("qwen3_5_text", "damork_text")',
    )

    insert_after_once(
        package_root / "utils/deep_gemm.py",
        "_DEEPGEMM_BLACKWELL_EXCLUDED_MODEL_TYPES: set[str] = {\n",
        '    "damork_text",\n',
        '"damork_text"',
    )

    copy_processor(package_root)

    insert_after_once(
        package_root / "transformers_utils/processors/__init__.py",
        '    "CohereASRProcessor",\n',
        (
            '    "DamorkVLImageProcessorFast",\n'
            '    "DamorkVLProcessor",\n'
            '    "DamorkVLVideoProcessor",\n'
        ),
        '"DamorkVLProcessor"',
    )

    insert_after_once(
        package_root / "transformers_utils/processors/__init__.py",
        (
            '    "CohereASRProcessor": '
            '"vllm.transformers_utils.processors.cohere_asr",\n'
        ),
        (
            '    "DamorkVLImageProcessorFast": '
            '"vllm.transformers_utils.processors.damork_vl",\n'
            '    "DamorkVLProcessor": '
            '"vllm.transformers_utils.processors.damork_vl",\n'
            '    "DamorkVLVideoProcessor": '
            '"vllm.transformers_utils.processors.damork_vl",\n'
        ),
        '"DamorkVLProcessor": "vllm.transformers_utils.processors.damork_vl"',
    )

    insert_after_once(
        package_root / "model_executor/models/qwen3_vl.py",
        "from vllm.tokenizers.registry import cached_tokenizer_from_config\n",
        "from vllm.transformers_utils.processors.damork_vl import DamorkVLProcessor\n",
        "DamorkVLProcessor",
    )

    replace_once(
        package_root / "model_executor/models/qwen3_vl.py",
        "        return self.ctx.get_hf_processor(\n"
        "            Qwen3VLProcessor,\n",
        "        return self.ctx.get_hf_processor(\n"
        "            (Qwen3VLProcessor, DamorkVLProcessor),\n",
        "(Qwen3VLProcessor, DamorkVLProcessor)",
    )


def validate_patch() -> None:
    from vllm.transformers_utils.configs.qwen3_5 import DamorkConfig
    from vllm.transformers_utils import processors

    config = DamorkConfig(
        text_config={"model_type": "damork_text", "num_attention_heads": 8},
        vision_config={"model_type": "damork"},
    )

    assert config.model_type == "damork"
    assert config.text_config.model_type == "damork_text"
    assert config.text_config.num_attention_heads == 8
    assert config.vision_config.model_type == "damork"
    assert processors.DamorkVLProcessor.__name__ == "DamorkVLProcessor"
    assert processors.DamorkVLImageProcessorFast.__name__ == (
        "DamorkVLImageProcessorFast"
    )
    assert processors.DamorkVLVideoProcessor.__name__ == (
        "DamorkVLVideoProcessor"
    )


def main() -> None:
    package_root = find_vllm_package_root()
    print(f"found installed vLLM package at {package_root}")
    patch_installed_vllm(package_root)
    validate_patch()


if __name__ == "__main__":
    main()

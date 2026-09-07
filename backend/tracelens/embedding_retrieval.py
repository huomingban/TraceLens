"""Local embedding backends used by the Qdrant retrieval layer.

The reference project uses a pinned BGE-small-zh-v1.5 INT8 ONNX model. It is
loaded lazily from a persistent volume so API startup remains useful when the
model has not been downloaded yet.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np

DEFAULT_ONNX_REPO = "Xenova/bge-small-zh-v1.5"
DEFAULT_ONNX_REVISION = "75c43b069aac4d136ba6bc1122f995fedcfd2781"
DEFAULT_ONNX_MODEL_FILE = "onnx/model_quantized.onnx"
DEFAULT_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文档："


class EmbeddingBackend(Protocol):
    def encode(self, texts: Sequence[str], *, query: bool = False) -> Any: ...


class LocalOnnxEmbeddingBackend:
    """Run the reference project's pinned BGE model through ONNX Runtime."""

    def __init__(
        self,
        model_dir: Path,
        *,
        repo_id: str = DEFAULT_ONNX_REPO,
        revision: str = DEFAULT_ONNX_REVISION,
        model_file: str = DEFAULT_ONNX_MODEL_FILE,
        query_prefix: str = DEFAULT_QUERY_PREFIX,
        allow_download: bool = False,
        cpu_threads: int = 2,
        batch_size: int = 32,
    ) -> None:
        self.model_dir = Path(model_dir).resolve()
        self.repo_id = repo_id
        self.revision = revision
        self.model_file = model_file
        self.query_prefix = query_prefix
        self.batch_size = max(1, int(batch_size))
        required = ("tokenizer.json", model_file)
        missing = [name for name in required if not (self.model_dir / name).is_file()]
        if missing and allow_download:
            self._download(required)
            missing = [name for name in required if not (self.model_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"local embedding model is incomplete: {', '.join(missing)}"
            )

        import onnxruntime as ort
        from tokenizers import Tokenizer

        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, int(cpu_threads))
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(self.model_dir / model_file),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.input_names = {item.name for item in self.session.get_inputs()}
        self.tokenizer = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=512)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")

    def _download(self, required: Sequence[str]) -> None:
        from huggingface_hub import hf_hub_download

        self.model_dir.mkdir(parents=True, exist_ok=True)
        for filename in required:
            hf_hub_download(
                repo_id=self.repo_id,
                filename=filename,
                revision=self.revision,
                local_dir=self.model_dir,
            )

    def encode(self, texts: Sequence[str], *, query: bool = False) -> np.ndarray:
        values = [str(item).strip() for item in texts]
        if not values:
            return np.empty((0, 0), dtype=np.float32)
        if query:
            values = [self.query_prefix + item for item in values]

        batches: list[np.ndarray] = []
        for start in range(0, len(values), self.batch_size):
            encodings = self.tokenizer.encode_batch(values[start : start + self.batch_size])
            feeds: dict[str, np.ndarray] = {}
            if "input_ids" in self.input_names:
                feeds["input_ids"] = np.asarray([item.ids for item in encodings], dtype=np.int64)
            if "attention_mask" in self.input_names:
                feeds["attention_mask"] = np.asarray(
                    [item.attention_mask for item in encodings], dtype=np.int64
                )
            if "token_type_ids" in self.input_names:
                feeds["token_type_ids"] = np.asarray(
                    [item.type_ids for item in encodings], dtype=np.int64
                )
            output = np.asarray(self.session.run(None, feeds)[0], dtype=np.float32)
            vectors = output[:, 0, :] if output.ndim == 3 else output
            if vectors.ndim != 2:
                raise RuntimeError(f"unexpected embedding output shape: {vectors.shape}")
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            batches.append(vectors / np.maximum(norms, 1e-12))
        return np.concatenate(batches, axis=0)


# Keep the old public boundary available to callers and tests.
from .retrieval import embed_texts, get_embedding_model  # noqa: E402

__all__ = [
    "EmbeddingBackend",
    "LocalOnnxEmbeddingBackend",
    "embed_texts",
    "get_embedding_model",
]

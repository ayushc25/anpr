"""ONNX Runtime backend, with the OpenVINO Execution Provider as an option.

We deliberately use the OpenVINO *EP* rather than the standalone OpenVINO API:
one API to maintain, OpenVINO kernels where they help, and automatic
per-subgraph fallback when an op is unsupported.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .backend import InferenceBackend
from .threading import ThreadBudget

logger = logging.getLogger("anpr.ai.inference")


class OnnxRuntimeBackend(InferenceBackend):
    def __init__(
        self,
        artifact: str | Path,
        threads: ThreadBudget,
        use_openvino: bool = False,
        cache_dir: str | Path | None = None,
    ):
        import onnxruntime as ort

        self.artifact = Path(artifact)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads.intra_op
        opts.inter_op_num_threads = threads.inter_op
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.log_severity_level = 3  # warnings and above only

        providers: list = []
        if use_openvino:
            ov_opts = {"device_type": "CPU", "num_of_threads": threads.intra_op}
            if cache_dir:
                # Without a cache the OpenVINO EP recompiles the graph on every
                # worker start, adding 5-20s. With it, restarts are fast.
                Path(cache_dir).mkdir(parents=True, exist_ok=True)
                ov_opts["cache_dir"] = str(cache_dir)
            providers.append(("OpenVINOExecutionProvider", ov_opts))
        providers.append("CPUExecutionProvider")

        self.session = ort.InferenceSession(str(self.artifact), sess_options=opts, providers=providers)
        active = self.session.get_providers()
        self.name = f"onnxruntime[{active[0]}]"

        self._inputs = self.session.get_inputs()
        self._input_names = [i.name for i in self._inputs]
        self._output_names = [o.name for o in self.session.get_outputs()]
        self._input_shapes = {i.name: tuple(i.shape) for i in self._inputs}

        logger.info(
            "loaded %s via %s (intra_op=%d)", self.artifact.name, active[0], threads.intra_op
        )

    @property
    def input_names(self) -> list[str]:
        return list(self._input_names)

    @property
    def output_names(self) -> list[str]:
        return list(self._output_names)

    @property
    def input_shapes(self) -> dict[str, tuple]:
        return dict(self._input_shapes)

    def run(self, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        return self.session.run(self._output_names, inputs)

    def close(self) -> None:
        self.session = None  # type: ignore[assignment]

"""
Hybrid Variational Quantum Classifier (QNN-based IDS) for VQC–ZTI evaluation.

Hybrid structure:
  x (n_features)
    -> classical embedder (MLP) -> angles (n_qubits)
    -> PQC (RY encoding + entangling ansatz)
    -> classical head (MLP) -> logits -> log_softmax

Important notes:
- TorchLayer expects the data argument to be named "inputs" by default.
- For Hybrid QNN (trainable embedder), we DO NOT use qml.batch_input.
- Batched broadcasted circuits can produce shape quirks across PennyLane versions.
  We normalize TorchLayer outputs robustly to (B, n_classes).

Supports:
- task="binary"    -> n_classes=2
- task="multiclass"-> n_classes=3
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import numpy as np
import pennylane as qml
import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class QNNResourceInfo:
    n_features: int
    n_qubits: int
    n_layers: int
    n_classes: int
    use_embedder: bool
    embed_hidden: int
    embed_layers: int
    angle_scale: float
    use_head: bool
    head_hidden: int

    # simple analytic counts (per forward pass per sample)
    n_ry_enc: int
    n_rot: int
    n_cnot: int


class VQCClassifier(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_layers: int = 2,
        n_qubits: Optional[int] = None,
        task: str = "multiclass",            # "binary" or "multiclass"
        n_classes: Optional[int] = None,     # override if needed
        diff_method: str = "adjoint",        # safe default for lightning.*
        # Hybrid knobs
        use_embedder: bool = True,
        embed_hidden: int = 64,
        embed_layers: int = 2,
        angle_scale: float = float(np.pi),
        use_head: bool = True,
        head_hidden: int = 32,
    ) -> None:
        super().__init__()

        self.n_features = int(n_features)
        self.n_qubits = int(n_qubits) if n_qubits is not None else int(n_features)
        self.n_layers = int(n_layers)

        task = str(task).lower().strip()
        if n_classes is None:
            self.n_classes = 2 if task == "binary" else 3
        else:
            self.n_classes = int(n_classes)

        self.task = task
        self.use_embedder = bool(use_embedder)
        self.embed_hidden = int(embed_hidden)
        self.embed_layers = int(embed_layers)
        self.angle_scale = float(angle_scale)
        self.use_head = bool(use_head)
        self.head_hidden = int(head_hidden)

        # --------------------------
        # Classical embedder
        # --------------------------
        # Maps (B, n_features) -> (B, n_qubits)
        if self.use_embedder:
            layers = []
            in_dim = self.n_features
            for _ in range(max(self.embed_layers - 1, 0)):
                layers.append(nn.Linear(in_dim, self.embed_hidden))
                layers.append(nn.GELU())
                in_dim = self.embed_hidden
            layers.append(nn.Linear(in_dim, self.n_qubits))
            self.embedder = nn.Sequential(*layers)
        else:
            self.embedder = None

        # --------------------------
        # PennyLane device selection
        # --------------------------
        # Prefer lightning.gpu if installed, else lightning.qubit, else default.qubit.
        dev = None
        self.pl_device_name = "default.qubit"
        for dev_name in ("lightning.gpu", "lightning.qubit", "default.qubit"):
            try:
                dev = qml.device(dev_name, wires=self.n_qubits)
                self.pl_device_name = dev_name
                break
            except Exception:
                continue
        if dev is None:
            dev = qml.device("default.qubit", wires=self.n_qubits)
            self.pl_device_name = "default.qubit"

        # --------------------------
        # Diff method sanity
        # --------------------------
        # Hybrid + broadcast batching can fail with parameter-shift in some PL versions.
        self.diff_method = str(diff_method).lower().strip()
        if self.use_embedder and self.diff_method == "parameter-shift":
            if self.pl_device_name.startswith("lightning"):
                self.diff_method = "adjoint"
            else:
                self.diff_method = "backprop"

        # PQC trainable weights
        weight_shapes = {"weights": (self.n_layers, self.n_qubits, 3)}

        def circuit(inputs, weights):
            # inputs can be shape (n_qubits,) OR (B, n_qubits)
            for i in range(self.n_qubits):
                qml.RY(inputs[..., i], wires=i)

            for l in range(self.n_layers):
                for i in range(self.n_qubits):
                    qml.Rot(weights[l, i, 0], weights[l, i, 1], weights[l, i, 2], wires=i)
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])

            return [qml.expval(qml.PauliZ(wires=i)) for i in range(self.n_classes)]

        qnode = qml.QNode(circuit, dev, interface="torch", diff_method=self.diff_method)
        self.qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)

        # --------------------------
        # Optional classical head
        # --------------------------
        if self.use_head:
            self.head = nn.Sequential(
                nn.Linear(self.n_classes, self.head_hidden),
                nn.GELU(),
                nn.Linear(self.head_hidden, self.n_classes),
            )
        else:
            self.head = None

    def _normalize_logits_shape(self, raw: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Normalize TorchLayer output to (B, k), handling PennyLane shape quirks.

        Handles:
        - (k, B, 1)   -> (B, k)
        - (B, k, 1)   -> (B, k)
        - (k, B)      -> (B, k)
        - (B, k)      -> (B, k)
        - flattened numel == B*k
        - "double batching": (B, B*k) or any numel == B*B*k  -> take diagonal -> (B, k)
        """
        B = int(batch_size)
        k = int(self.n_classes)

        if not torch.is_tensor(raw):
            raw = torch.as_tensor(raw)

        # (k, B, 1)
        if raw.dim() == 3 and raw.shape[0] == k and raw.shape[1] == B:
            return raw.squeeze(-1).transpose(0, 1)

        # (B, k, 1)
        if raw.dim() == 3 and raw.shape[0] == B and raw.shape[1] == k:
            return raw.squeeze(-1)

        # (k, B)
        if raw.dim() == 2 and raw.shape[0] == k and raw.shape[1] == B:
            return raw.transpose(0, 1)

        # (B, k)
        if raw.dim() == 2 and raw.shape[0] == B and raw.shape[1] == k:
            return raw

        # flattened but correct total
        if raw.numel() == B * k:
            return raw.reshape(B, k)

        # "double batching": (B, B*k)
        if raw.dim() == 2 and raw.shape[0] == B and raw.shape[1] == B * k:
            tmp = raw.reshape(B, B, k)
            idx = torch.arange(B, device=raw.device)
            return tmp[idx, idx, :]

        # any shape with numel B*B*k
        if raw.numel() == B * B * k:
            tmp = raw.reshape(B, B, k)
            idx = torch.arange(B, device=raw.device)
            return tmp[idx, idx, :]

        raise RuntimeError(
            f"Unexpected TorchLayer output shape {tuple(raw.shape)}; cannot coerce to (B={B}, k={k})."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
          x: (B, n_features)

        Returns:
          log_probs: (B, n_classes) for NLLLoss
        """
        if x.dim() != 2:
            raise ValueError(f"Expected x as (B, features). Got {tuple(x.shape)}")

        B = int(x.shape[0])

        # 1) angles
        if self.embedder is not None:
            angles = self.embedder(x)  # (B, n_qubits)
        else:
            # pad/truncate raw features to n_qubits
            if x.shape[1] < self.n_qubits:
                pad = torch.zeros((B, self.n_qubits - x.shape[1]), dtype=x.dtype, device=x.device)
                angles = torch.cat([x, pad], dim=1)
            else:
                angles = x[:, : self.n_qubits]

        # Stabilize: bound angles then scale (keeps within [-pi, pi])
        angles = torch.tanh(angles) * self.angle_scale

        # 2) PQC
        raw = self.qlayer(angles)
        logits = self._normalize_logits_shape(raw, B)

        # 3) head
        if self.head is not None:
            logits = self.head(logits)

        return F.log_softmax(logits, dim=1)

    def model_info(self) -> Dict[str, Any]:
        """Paper-grade model + resource metadata."""
        n_ry_enc = self.n_qubits
        n_rot = self.n_layers * self.n_qubits
        n_cnot = self.n_layers * max(self.n_qubits - 1, 0)

        info = QNNResourceInfo(
            n_features=self.n_features,
            n_qubits=self.n_qubits,
            n_layers=self.n_layers,
            n_classes=self.n_classes,
            use_embedder=self.use_embedder,
            embed_hidden=self.embed_hidden,
            embed_layers=self.embed_layers,
            angle_scale=self.angle_scale,
            use_head=self.use_head,
            head_hidden=self.head_hidden,
            n_ry_enc=n_ry_enc,
            n_rot=n_rot,
            n_cnot=n_cnot,
        )

        def n_params(m: Optional[nn.Module]) -> int:
            if m is None:
                return 0
            return int(sum(p.numel() for p in m.parameters() if p.requires_grad))

        return {
            **asdict(info),
            "pl_device": getattr(self, "pl_device_name", "unknown"),
            "diff_method": getattr(self, "diff_method", "unknown"),
            "params_total": int(sum(p.numel() for p in self.parameters() if p.requires_grad)),
            "params_embedder": n_params(self.embedder),
            "params_head": n_params(self.head),
            "params_quantum": int(sum(p.numel() for p in self.qlayer.parameters() if p.requires_grad)),
        }
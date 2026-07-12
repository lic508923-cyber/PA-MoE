"""PA-MoELog 骨架的玩具前向传播演示。

运行方式：
    python -m pa_moelog.demo_forward
"""

from __future__ import annotations

import torch

from pa_moelog.data import LogPreprocessor
from pa_moelog.models import PAMoELog


def main() -> None:
    torch.manual_seed(7)

    logs = [
        "Failed login from 192.168.1.10 user=root port=22 error=403",
        "kernel panic at 0x000000FF pid=1234",
        "open file /etc/passwd by user admin",
        "service started successfully on port 8080",
    ]

    preprocessor = LogPreprocessor()
    batch = preprocessor.parse_sequence(logs)

    model = PAMoELog(num_experts=3, hidden_dim=128, backbone_name="simple-hash-encoder")
    model.eval()

    with torch.no_grad():
        output = model(
            semantic_texts=batch["semantic_texts"],
            parameters=batch["parameters"],
        )

    print("semantic_texts:")
    for text in batch["semantic_texts"]:
        print(f"  - {text}")
    print("final_score:", output["final_score"])
    print("classifier_score:", output["classifier_score"])
    print("energy_score:", output["energy_score"])
    print("fusion_weights:", output["fusion_weights"])


if __name__ == "__main__":
    main()

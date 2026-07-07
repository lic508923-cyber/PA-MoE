"""免解析、保参数的日志预处理。

本模块刻意避开 Drain 或 Spell 等模板挖掘器。它为轻量文本编码器保留
语义文本视图，同时显式保留参数类型和值，以供参数感知注意力使用。
"""

from __future__ import annotations

import re
from collections import OrderedDict
from typing import Dict, List


PARAMETER_TYPES = ["IP", "PATH", "URL", "HEX", "NUM", "PORT", "USER", "PID", "FILE", "TIME"]


class LogPreprocessor:
    """从原始日志行中抽取语义文本和带类型的参数。"""

    def __init__(self) -> None:
        self.patterns = OrderedDict(
            [
                ("URL", re.compile(r"\bhttps?://[^\s]+", re.IGNORECASE)),
                ("IP", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
                ("TIME", re.compile(r"\b(?:\d{2}:\d{2}:\d{2}|\d{4}-\d{2}-\d{2}(?:[T\s]\d{2}:\d{2}:\d{2})?)\b")),
                ("PATH", re.compile(r"(?<!\w)(?:/[A-Za-z0-9._-]+)+/?")),
                ("HEX", re.compile(r"\b0x[0-9A-Fa-f]+\b")),
                ("PID", re.compile(r"\bpid\s*=\s*(\d+)\b", re.IGNORECASE)),
                ("PORT", re.compile(r"\bport\s*=\s*(\d+)\b|\bport\s+(\d+)\b", re.IGNORECASE)),
                ("USER", re.compile(r"\buser\s*=\s*([A-Za-z_][\w.-]*)\b|\buser\s+([A-Za-z_][\w.-]*)\b", re.IGNORECASE)),
                ("FILE", re.compile(r"\b[A-Za-z0-9_.-]+\.(?:log|txt|conf|cfg|json|xml|csv|jar|py|sh|service)\b", re.IGNORECASE)),
                ("NUM", re.compile(r"(?<![\w.])[-+]?\d+(?:\.\d+)?(?![\w.])")),
            ]
        )

    def parse_line(self, line: str) -> dict:
        """将一条原始日志解析为语义文本和带类型的参数。"""

        parameters: Dict[str, List[str]] = {name: [] for name in PARAMETER_TYPES}
        spans: list[tuple[int, int, str, str]] = []

        for param_type, pattern in self.patterns.items():
            for match in pattern.finditer(line):
                value = self._extract_value(param_type, match)
                start, end = self._extract_span(param_type, match)
                if self._overlaps_existing(start, end, spans):
                    continue
                parameters[param_type].append(value)
                spans.append((start, end, param_type, value))

        semantic_text = self._replace_spans(line, spans)
        semantic_text = self._normalize_key_value_text(semantic_text)

        return {"semantic_text": semantic_text, "parameters": parameters}

    def parse_sequence(self, lines: list[str]) -> dict:
        """解析日志序列，同时保留每行级别的参数。"""

        parsed = [self.parse_line(line) for line in lines]
        return {
            "semantic_texts": [item["semantic_text"] for item in parsed],
            "parameters": [item["parameters"] for item in parsed],
        }

    @staticmethod
    def _extract_value(param_type: str, match: re.Match) -> str:
        if param_type in {"PID", "PORT", "USER"}:
            return next(group for group in match.groups() if group is not None)
        return match.group(0)

    @staticmethod
    def _extract_span(param_type: str, match: re.Match) -> tuple[int, int]:
        if param_type in {"PID", "PORT", "USER"}:
            for index, group in enumerate(match.groups(), start=1):
                if group is not None:
                    return match.start(index), match.end(index)
        return match.start(), match.end()

    @staticmethod
    def _overlaps_existing(start: int, end: int, spans: list[tuple[int, int, str, str]]) -> bool:
        return any(start < existing_end and end > existing_start for existing_start, existing_end, _, _ in spans)

    @staticmethod
    def _replace_spans(line: str, spans: list[tuple[int, int, str, str]]) -> str:
        if not spans:
            return " ".join(line.split())

        pieces: list[str] = []
        cursor = 0
        for start, end, param_type, _ in sorted(spans, key=lambda item: item[0]):
            pieces.append(line[cursor:start])
            pieces.append(f"<{param_type}>")
            cursor = end
        pieces.append(line[cursor:])
        return " ".join("".join(pieces).split())

    @staticmethod
    def _normalize_key_value_text(text: str) -> str:
        text = re.sub(r"\s*=\s*<", " <", text)
        text = re.sub(r"\s*=\s*", " ", text)
        return " ".join(text.split())

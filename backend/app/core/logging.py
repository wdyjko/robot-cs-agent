"""统一日志配置：控制台 + 可选文件输出。"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_CONFIGURED = False
_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: str = "INFO", log_file: str | Path | None = None) -> None:
    """初始化根 logger。重复调用只生效一次。"""
    global _CONFIGURED
    if _CONFIGURED:
        return

    root = logging.getLogger()
    root.setLevel(level.upper())

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    if log_file:
        try:
            path = Path(log_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(path, encoding="utf-8")
            file_handler.setFormatter(formatter)
            root.addHandler(file_handler)
        except OSError:  # pragma: no cover - 无写权限时忽略文件日志
            pass

    # 第三方库降噪
    for noisy in ("httpx", "httpcore", "urllib3", "chromadb", "sentence_transformers", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """获取带项目前缀的 logger。"""
    return logging.getLogger(name)

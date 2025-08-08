from __future__ import annotations

import os
import shutil
import torch
from typing import Literal, Optional, List
from dataclasses import dataclass
from ..dataTypes import *

class SystemDiagnostics:
    """Inspect system resources and recommend a suitable model."""
    def __init__(self, drive: Optional[str] = None) -> None:
        self.totalRam = self._getTotalRam()
        self.gpuMemory = self._getGpuMemory()
        self.freeDiskSpace = self._getFreeDiskSpace()
        self.drive = drive if drive else "C"

    def _getTotalRam(self) -> int:
        sysConf = getattr(os, "sysconf", None)
        if sysConf is None:
            return 0
        try:
            pageSize = sysConf("SC_PAGE_SIZE")
            pageCount = sysConf("SC_PHYS_PAGES")
            return pageSize * pageCount
        except (ValueError, OSError):
            return 0

    def _getGpuMemory(self) -> int:
        if torch.cuda.is_available():
            try:
                properties = torch.cuda.get_device_properties(0)
                return properties.total_memory
            except Exception:
                return 0
        return 0

    def _getFreeDiskSpace(self) -> int:
        try:
            usage = shutil.disk_usage(os.getcwd())
            return usage.free
        except Exception:
            return 0

    def recommendModel(self) -> Optional[ModelSpec]:
        """Choose the largest model that fits available memory and disk."""
        availableMemory = self.gpuMemory or self.totalRam
        for spec in Models:
            if (
                availableMemory >= spec.memory
                and self.freeDiskSpace >= spec.size
            ):
                return spec
        return None

    def getModelSpec(self, modelID: str) -> Optional[ModelSpec]:
        """Get the model specification for a given model ID."""
        for spec in Models:
            if spec.id == modelID:
                return spec
        return None
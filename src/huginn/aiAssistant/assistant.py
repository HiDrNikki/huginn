"""Local model assistant.

This module provides :class:`AIAssistant`, a simple wrapper around a locally
mounted model downloaded from the Hugging Face Hub.
"""

from __future__ import annotations
import os
from typing import Any, Optional, cast
from pathlib import Path
from noexcept import no

from .systemDiagnostics import SystemDiagnostics
from ..dataTypes import *

class AIAssistant:
    """Run text generation using a locally downloaded model.

    Parameters
    ----------
    modelId:
        Identifier of the model on Hugging Face. Defaults to the largest
        available TheBloke model as suggested by :class:`SystemDiagnostics`.
    device:
        Optional device override. Uses CUDA when available otherwise CPU.
    tokenizer:
        Optional tokenizer name to use. Defaults to the first available
        tokenizer supported by the selected model.
    localFilesOnly:
        If ``True``, only loads the model from the local cache without
        attempting to download it.
    quantized:
        If ``True``, applies dynamic 8-bit quantization using PyTorch to
        reduce memory usage.
    hfToken:
        Optional Hugging Face token used for authentication during
        model downloads.
    ``**modelKwargs``:
        Additional arguments passed to
        :func:`transformers.AutoModelForCausalLM.from_pretrained`.
    """
    def __init__(
        self,
        *,
        modelID: Optional[ModelID] = None,
        device: Optional[Device] = None,
        tokenizer: Optional[str] = None,
        local: bool = False,
        cache: Optional[Path] = None,
        quantized: bool = True,
        hfToken: Optional[str] = None,
        **modelKwargs: Any,
    ) -> None:
        if cache: # Doesn't work
            print(f"Using cache directory: {cache}")
            os.environ["HF_HOME"] = str(cache)
            os.environ["HF_DATASETS_CACHE"] = str(cache / "datasets")
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # type: ignore
        import torch
        from huggingface_hub import login, snapshot_download
        if hfToken:
            login(token=hfToken)
        elif not local:
            raise ValueError("hfToken must be provided if localFilesOnly is False")
        self.diagnostics = SystemDiagnostics()
        allowedTokenizers = []
        if modelID is None:
            self.modelSpec = self.diagnostics.recommendModel()
            if self.modelSpec is None:
                raise RuntimeError(
                    "No suitable model found by analysing your system spec. Please specify a modelId to override."
                )
            modelID = self.modelSpec.id
            allowedTokenizers = self.modelSpec.tokenizers
        else:
            self.modelSpec = self.diagnostics.getModelSpec(modelID)
            if self.modelSpec is None:
                raise ValueError(f"Model ID {modelID} not found in system diagnostics.")
            allowedTokenizers = self.modelSpec.tokenizers
        modelID = cast(str, modelID)
        # Do these essentially do the exact same thing?
        assert isinstance(modelID, str), "modelID must be a string"

        if not allowedTokenizers:
            raise ValueError("No tokenizers available for this model")

        if tokenizer is not None:
            if tokenizer not in allowedTokenizers.fast:
                raise ValueError(f"Tokenizer {tokenizer} is not supported for model {modelID}")
            chosenTokenizer = tokenizer
        else:
            chosenTokenizer = allowedTokenizers.fast[0]
        if local:
            modelPath = snapshot_download(modelID, local_files_only=True, token=hfToken)
        else:
            try:
                modelPath = snapshot_download(modelID, local_files_only=True, token=hfToken)
            except Exception:
                modelPath = snapshot_download(modelID, local_files_only=False, token=hfToken)

        tokenizerKwargs = {"tokenizer_type": chosenTokenizer}
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                modelPath, legacy=False, **tokenizerKwargs
            )
        except Exception:
            # Get the slow tokenizer if the fast one fails
            chosenTokenizer = allowedTokenizers.slow[0]
            tokenizerKwargs = {"tokenizer_type": chosenTokenizer}
            self.tokenizer = AutoTokenizer.from_pretrained(
                modelPath, legacy=True, use_fast=False, **tokenizerKwargs
            )
        chosenDevice: torch.device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        if quantized:
            self.model = AutoModelForCausalLM.from_pretrained(
                modelPath, **modelKwargs
            )
            self.model = torch.quantization.quantize_dynamic(
                self.model, {torch.nn.Linear}, dtype=torch.qint8
            )
            chosenDevice = torch.device("cpu")
            self.model = self.model.to(device=chosenDevice)        
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                modelPath, **modelKwargs
            )
            self.model = self.model.to(device=chosenDevice)
        self.device = chosenDevice
        self.model.eval()

    def generate(self, prompt: str, **generationKwargs: Any) -> str:
        """Generate text locally from ``prompt``.

        Parameters
        ----------
        prompt:
            The text prompt to send to the model.
        ``**generationKwargs``:
            Additional keyword arguments forwarded to
            :meth:`transformers.PreTrainedModel.generate`.

        Returns
        -------
        str
            The generated text returned by the model.
        """
        import torch
        print(f"Generating text on {self.device} with model {self.modelSpec.name}")
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not initialized")
        inputIds = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputIds = self.model.generate(**inputIds, **generationKwargs)
        return self.tokenizer.decode(outputIds[0], skip_special_tokens=True)
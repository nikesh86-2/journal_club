"""
paper_analyzer.py

Paper analysis for Journal Club.

Responsibilities:
  - Generate concise paper summaries
  - Perform gap analysis (methodology, controls, statistics, reproducibility)
  - Generate structured critiques
  - Score paper quality (methodology rigor, statistical power, reproducibility)
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
import time
import os
import re
import threading
from typing import Any, Dict, List
import shutil
import importlib
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json

log = logging.getLogger("journal_club.analyzer")

# Ensure analyzer logs are persisted to disk (if SLURM kills the process)
try:
    logs_dir = Path(__file__).parents[1] / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / f"paper_analyzer-{os.getpid()}-{int(time.time())}.log"
    file_handler = RotatingFileHandler(str(log_file), maxBytes=10 * 1024 * 1024, backupCount=5)
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s"))
    file_handler.setLevel(logging.INFO)
    log.addHandler(file_handler)
except Exception:
    # If file logging setup fails, continue using existing logging configuration
    pass


# ---------------------------------------------------------------------------
# LLM Configuration
# ---------------------------------------------------------------------------

LLM_MODEL = os.getenv("JOURNAL_CLUB_LLM_MODEL", "gpt-4")
LLM_TEMPERATURE = float(os.getenv("JOURNAL_CLUB_LLM_TEMPERATURE", "0.3"))
LLM_MAX_TOKENS = int(os.getenv("JOURNAL_CLUB_LLM_MAX_TOKENS", "2000"))
USE_FINETUNED = os.getenv("JOURNAL_CLUB_USE_FINETUNED", "0") == "1"
FINETUNED_MODEL_PATH = os.getenv(
    "JOURNAL_CLUB_FINETUNED_MODEL_PATH",
    "training/journal_club_merged_model"
)
LOCAL_BASE_MODEL_PATH = os.getenv(
    "JOURNAL_CLUB_LOCAL_BASE_MODEL_PATH",
    "/home/nike/models/qwen3.5-35b-a3b-gptq" # Path to a downloaded base model (e.g., Llama-3.1-8B or Mistral-7B)
)
# GGUF model path for llama-cpp-python
GGUF_MODEL_PATH = os.getenv(
    "JOURNAL_CLUB_GGUF_MODEL_PATH",
    "/home/nike/models/qwen3.5-35b-a3b-gptq"
)
# External llama-server endpoint
LLAMA_SERVER_URL = os.getenv("JOURNAL_CLUB_LLAMA_SERVER_URL", "http://localhost:8080")
USE_LLAMA_SERVER = os.getenv("JOURNAL_CLUB_USE_LLAMA_SERVER", "1") == "1"
FALLBACK_TO_BASE = os.getenv("JOURNAL_CLUB_FALLBACK_TO_BASE", "1") == "1"
FORCE_CPU_OFFLOAD = os.getenv("JOURNAL_CLUB_FORCE_CPU_OFFLOAD", "0") == "1"
RETRY_ATTEMPTS = int(os.getenv("JOURNAL_CLUB_RETRY_ATTEMPTS", "3"))
MAX_ANALYSIS_WORKERS = int(os.getenv("JOURNAL_CLUB_MAX_ANALYSIS_WORKERS", "4"))
ENABLE_ANALYSIS_CACHE = os.getenv("JOURNAL_CLUB_ENABLE_ANALYSIS_CACHE", "1") == "1"
CACHE_DIR = Path(__file__).parents[1] / "cache" / "analysis"


_cached_llm_clients: Dict[str, Any] = {}

# Protects the entire LLM client cache lifecycle:
# check -> create -> store and cleanup -> remove.
_llm_cache_lock = threading.Lock()

def cleanup_llm_clients(force: bool = False):
    """Properly clean up all LLM clients to prevent segfaults.
    
    Args:
        force: If True, clear cache even for external llama-server clients.
               If False (default), preserve external server clients since they don't need cleanup.
    """
    with _llm_cache_lock:
        keys_to_remove = []
        for key, client in _cached_llm_clients.items():
            # Skip cleanup for external llama-server clients unless forced
            if not force and isinstance(client, LlamaServerLLM):
                log.debug(f"Skipping cleanup for external llama-server client: {key}")
                continue
                
            if hasattr(client, 'cleanup'):
                try:
                    log.info(f"Cleaning up LLM client: {key}")
                    client.cleanup()
                except Exception as e:
                    log.warning(f"Error cleaning up LLM client {key}: {e}")
            keys_to_remove.append(key)
        
        # Remove only the clients that were cleaned up
        for key in keys_to_remove:
            _cached_llm_clients.pop(key, None)
        
        log.info("LLM clients cleaned up successfully (%d removed, %d preserved)", 
                 len(keys_to_remove), len(_cached_llm_clients))


# ---------------------------------------------------------------------------
# Pydantic Models for Structured Output
# ---------------------------------------------------------------------------

class GapAnalysis(BaseModel):
    """Structured gap analysis with validation."""
    methodology: List[str] = Field(default_factory=list, description="Methodology gaps and limitations")
    controls: List[str] = Field(default_factory=list, description="Missing or inadequate controls")
    statistics: List[str] = Field(default_factory=list, description="Statistical issues and concerns")
    reproducibility: List[str] = Field(default_factory=list, description="Reproducibility concerns")


# ---------------------------------------------------------------------------
# Llama Server Client for external GPU-accelerated inference
# ---------------------------------------------------------------------------

class LlamaServerLLM:
    """
    Wrapper class to connect to external llama-server for GPU-accelerated inference.
    This allows using a GPU-accelerated llama-server instead of in-process CPU inference.
    """
    
    def __init__(self, server_url: str = "http://localhost:8080", temperature: float = 0.3, max_tokens: int = 2000):
        self.server_url = server_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._lock = threading.Lock()
        
        # Test connection
        try:
            import requests
            response = requests.get(f"{server_url}/health", timeout=5)
            if response.status_code != 200:
                log.warning(f"Llama server health check failed: {response.status_code}")
        except Exception as e:
            log.warning(f"Failed to connect to llama server at {server_url}: {e}")
    
    def invoke(self, messages, enable_thinking: bool = False):
        """
        Invoke the LLM server with a list of messages.
        Uses /v1/chat/completions endpoint to let the server apply proper chat template.
        Expects messages to be a list of langchain message objects.
        
        Args:
            messages: List of langchain message objects
            enable_thinking: If False (default), disables the model's reasoning/thinking mode
                           to save tokens for the actual response. Useful for structured output tasks.
        """
        # Convert langchain messages to OpenAI-compatible format
        chat_messages = self._format_messages_openai(messages)
        
        # Build request payload
        request_payload = {
            "messages": chat_messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        
        # Pass enable_thinking via chat_template_kwargs if thinking is disabled
        # This tells the GGUF model's Jinja template to skip the reasoning block
        if not enable_thinking:
            request_payload["chat_template_kwargs"] = {"enable_thinking": False}
        
        # Generate response with thread safety
        with self._lock:
            try:
                import requests
                response = requests.post(
                    f"{self.server_url}/v1/chat/completions",
                    json=request_payload,
                    timeout=120
                )
                response.raise_for_status()
                data = response.json()
                # OpenAI-compatible format returns content in choices[0].message.content
                text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                if not text:
                    log.warning(f"Llama server returned empty response. Full response: {data}")
            except requests.exceptions.Timeout:
                log.error(f"Llama server request timed out after 120s")
                text = ""
            except requests.exceptions.RequestException as e:
                log.error(f"Llama server request failed: {e}")
                text = ""
            except Exception as e:
                log.error(f"Llama server error: {e}")
                text = ""
        
        # Return a simple object with content attribute to match langchain interface
        class Response:
            def __init__(self, content):
                self.content = content
        
        return Response(text)
    
    def _format_messages_openai(self, messages):
        """
        Convert langchain messages to OpenAI-compatible format for /v1/chat/completions.
        """
        chat_messages = []
        for msg in messages:
            if isinstance(msg, tuple):
                role, content = msg
            else:
                role = type(msg).__name__
                content = msg.content if hasattr(msg, 'content') else str(msg)
            
            # Map langchain message types to OpenAI roles
            if role in ['SystemMessage', 'system']:
                openai_role = "system"
            elif role in ['HumanMessage', 'user']:
                openai_role = "user"
            elif role in ['AIMessage', 'assistant']:
                openai_role = "assistant"
            else:
                openai_role = "user"  # Default fallback
            
            chat_messages.append({"role": openai_role, "content": content})
        
        return chat_messages
    
    def cleanup(self):
        """No cleanup needed for external server."""
        pass


# ---------------------------------------------------------------------------
# GGUF LLM Wrapper for llama-cpp-python
# ---------------------------------------------------------------------------

class GGUFChatLLM:
    """
    Wrapper class to make llama-cpp-python compatible with langchain's interface.
    This allows using GGUF models with the existing paper_analyzer code.
    """
    
    def __init__(self, model_path: str, n_gpu_layers: int = -1, n_ctx: int = 32768, temperature: float = 0.3, max_tokens: int = 2000):
        from llama_cpp import Llama
        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._lock = threading.Lock()
    
    def invoke(self, messages):
        """
        Invoke the LLM with a list of messages.
        Expects messages to be a list of langchain message objects.
        """
        # Convert langchain messages to llama.cpp format
        prompt = self._format_messages(messages)
        
        # Generate response with thread safety
        with self._lock:
            response = self.llm(
                prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stop=[],
            )
        
        # Extract the generated text
        text = response["choices"][0]["text"]
        
        # Return a simple object with content attribute to match langchain interface
        class Response:
            def __init__(self, content):
                self.content = content
        
        return Response(text)
    
    def _format_messages(self, messages):
        """
        Convert langchain messages to a prompt string for llama.cpp.
        """
        prompt = ""
        for msg in messages:
            if isinstance(msg, tuple):
                role, content = msg
            else:
                role = type(msg).__name__
                content = msg.content if hasattr(msg, 'content') else str(msg)
            
            if role in ['SystemMessage', 'system']:
                prompt += f"System: {content}\n\n"
            elif role in ['HumanMessage', 'user']:
                prompt += f"User: {content}\n\n"
            elif role in ['AIMessage', 'assistant']:
                prompt += f"Assistant: {content}\n\n"
        
        prompt += "Assistant: "
        return prompt
    
    def cleanup(self):
        """Properly clean up the model resources."""
        if hasattr(self, 'llm') and self.llm is not None:
            try:
                del self.llm
            except Exception as e:
                log.warning(f"Error during GGUF model cleanup: {e}")
            self.llm = None


# ---------------------------------------------------------------------------
# Result Caching
# ---------------------------------------------------------------------------

def get_paper_hash(paper: Dict[str, Any]) -> str:
    """Generate stable hash for paper based on DOI, title, and abstract."""
    content = f"{paper.get('doi', '')}_{paper.get('title', '')}_{paper.get('abstract', '')}"
    return hashlib.sha256(content.encode()).hexdigest()


def get_cached_analysis(paper: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Retrieve cached analysis result for a paper if available."""
    if not ENABLE_ANALYSIS_CACHE:
        return None
    
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        paper_hash = get_paper_hash(paper)
        cache_file = CACHE_DIR / f"{paper_hash}.json"
        
        if cache_file.exists():
            log.debug("Loading cached analysis for: %s", paper.get("title", "unknown")[:50])
            return json.loads(cache_file.read_text())
    except Exception as e:
        log.warning("Failed to load cached analysis: %s", e)
    
    return None


def save_cached_analysis(paper: Dict[str, Any], analysis: Dict[str, Any]) -> None:
    """Save analysis result to cache."""
    if not ENABLE_ANALYSIS_CACHE:
        return
    
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        paper_hash = get_paper_hash(paper)
        cache_file = CACHE_DIR / f"{paper_hash}.json"
        
        cache_file.write_text(json.dumps(analysis, indent=2))
        log.debug("Saved cached analysis for: %s", paper.get("title", "unknown")[:50])
    except Exception as e:
        log.warning("Failed to save cached analysis: %s", e)


def _get_response_text(response: Any) -> str:
    """Extract string content from LLM response object or string."""
    if hasattr(response, "content"):
        return str(response.content)
    return str(response)


def _parse_gap_analysis_with_retry(
    llm_output: str,
    max_retries: int = None,
    llm=None,
    messages=None,
    enable_thinking: bool = False,
) -> Dict[str, List[str]]:
    """Parse gap analysis with retry logic using Pydantic validation.
    
    If LLM and messages are provided, will re-invoke the LLM on retry
    instead of just retrying the same parse operation.
    
    Args:
        llm_output: The initial LLM output string to parse
        max_retries: Maximum number of retry attempts
        llm: LLM client to re-invoke on retry (optional)
        messages: Messages to send to LLM on retry (optional)
        enable_thinking: Whether to enable thinking mode on retry
    """
    if max_retries is None:
        max_retries = RETRY_ATTEMPTS
    
    current_output = llm_output
    
    for attempt in range(max_retries):
        try:
            # Try to parse as JSON directly first
            import json
            result = json.loads(current_output)
            
            # Validate with Pydantic model
            gap_analysis = GapAnalysis(**result)
            
            return {
                "methodology": gap_analysis.methodology,
                "controls": gap_analysis.controls,
                "statistics": gap_analysis.statistics,
                "reproducibility": gap_analysis.reproducibility,
            }
        except json.JSONDecodeError as e:
            log.warning("JSON parsing failed (attempt %d/%d): %s", attempt + 1, max_retries, e)
            
            # If we have an LLM and messages, re-invoke to get a new response
            if llm is not None and messages is not None and attempt < max_retries - 1:
                log.info("Re-invoking LLM for gap analysis (attempt %d/%d)", attempt + 2, max_retries)
                try:
                    if hasattr(llm, 'invoke') and 'enable_thinking' in llm.invoke.__code__.co_varnames:
                        response = llm.invoke(messages, enable_thinking=enable_thinking)
                    else:
                        response = llm.invoke(messages)
                    current_output = _get_response_text(response).strip()
                    if not current_output:
                        log.warning("LLM returned empty response on retry")
                        continue
                except Exception as retry_e:
                    log.error("LLM re-invocation failed: %s", retry_e)
                    continue
            
            if attempt == max_retries - 1:
                log.warning("Failed to parse after %d attempts, using rule-based fallback", max_retries)
                return None
        except Exception as e:
            log.warning("Validation failed (attempt %d/%d): %s", attempt + 1, max_retries, e)
            
            # If we have an LLM and messages, re-invoke to get a new response
            if llm is not None and messages is not None and attempt < max_retries - 1:
                log.info("Re-invoking LLM for gap analysis (attempt %d/%d)", attempt + 2, max_retries)
                try:
                    if hasattr(llm, 'invoke') and 'enable_thinking' in llm.invoke.__code__.co_varnames:
                        response = llm.invoke(messages, enable_thinking=enable_thinking)
                    else:
                        response = llm.invoke(messages)
                    current_output = _get_response_text(response).strip()
                    if not current_output:
                        log.warning("LLM returned empty response on retry")
                        continue
                except Exception as retry_e:
                    log.error("LLM re-invocation failed: %s", retry_e)
                    continue
            
            if attempt == max_retries - 1:
                log.warning("Failed to validate after %d attempts, using rule-based fallback", max_retries)
                return None
    
    return None


def get_llm_client(use_finetuned: bool = None):
    """Get LLM client (from llama-server,  fine-tuned model,
    local base model, or fallback). Uses thread-safe caching.
    """

    # Determine if we should use fine-tuned model
    if use_finetuned is None:
        use_finetuned = USE_FINETUNED

    cache_key = "finetuned" if use_finetuned else f"base_{LOCAL_BASE_MODEL_PATH}"

    # IMPORTANT:
    # Hold this lock across the entire check -> create -> store sequence.
    # Otherwise two analysis threads can both see a cache miss and
    # simultaneously load the same (potentially very large) model.
    with _llm_cache_lock:

        # Check cache while holding the lock
        cached_client = _cached_llm_clients.get(cache_key)
        if cached_client is not None:
            log.debug("Using cached LLM client: %s", cache_key)
            return cached_client

        log.info("Loading LLM client (cache miss): %s", cache_key)

        # ------------------------------------------------------------------
        # Try external llama-server first if enabled
        # ------------------------------------------------------------------
        if USE_LLAMA_SERVER:
            try:
                log.info("Using external llama-server at: %s", LLAMA_SERVER_URL)

                llm = LlamaServerLLM(
                    server_url=LLAMA_SERVER_URL,
                    temperature=LLM_TEMPERATURE,
                    max_tokens=LLM_MAX_TOKENS,
                )

                _cached_llm_clients[cache_key] = llm
                return llm

            except Exception as e:
                log.warning(
                    "Failed to connect to llama-server (%s), "
                    "falling back to local models",
                    e,
                )

                if not FALLBACK_TO_BASE:
                    log.error("Fallback disabled, no LLM available")
                    return None

        # ------------------------------------------------------------------
        # Try fine-tuned model first if enabled
        # ------------------------------------------------------------------
        if use_finetuned:
            finetuned_path = Path(__file__).parents[1] / FINETUNED_MODEL_PATH

            if finetuned_path.exists():
                try:
                    import torch
                    from langchain_huggingface import HuggingFacePipeline
                    from transformers import (
                        AutoModelForCausalLM,
                        AutoTokenizer,
                        pipeline,
                    )

                    log.info(
                        "Loading fine-tuned model from: %s",
                        finetuned_path,
                    )

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    tokenizer = AutoTokenizer.from_pretrained(
                        str(finetuned_path),
                        trust_remote_code=True,
                    )

                    dtype = (
                        torch.float16
                        if torch.cuda.is_available()
                        else torch.float32
                    )
                    device_map = "auto" if torch.cuda.is_available() else None

                    model = AutoModelForCausalLM.from_pretrained(
                        str(finetuned_path),
                        device_map=device_map,
                        torch_dtype=dtype,
                        trust_remote_code=True,
                    )

                    pipe = pipeline(
                        "text-generation",
                        model=model,
                        tokenizer=tokenizer,
                        max_new_tokens=LLM_MAX_TOKENS,
                        temperature=LLM_TEMPERATURE,
                        do_sample=True,
                    )

                    llm = HuggingFacePipeline(pipeline=pipe)

                    log.info(
                        "Successfully loaded fine-tuned HuggingFace model"
                    )

                    _cached_llm_clients[cache_key] = llm
                    return llm

                except Exception as e:
                    log.warning(
                        "Failed to load fine-tuned model (%s)",
                        e,
                    )

                    if torch and torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    if not FALLBACK_TO_BASE:
                        log.error("Fallback disabled, no LLM available")
                        return None

        # ------------------------------------------------------------------
        # Try GGUF model with llama-cpp-python
        # ------------------------------------------------------------------
        if GGUF_MODEL_PATH and os.path.exists(GGUF_MODEL_PATH):
            try:
                log.info(
                    "Loading GGUF model from: %s",
                    GGUF_MODEL_PATH,
                )

                llm = GGUFChatLLM(
                    model_path=GGUF_MODEL_PATH,
                    n_gpu_layers=-1,
                    n_ctx=4096,
                    temperature=LLM_TEMPERATURE,
                    max_tokens=LLM_MAX_TOKENS,
                )

                log.info("Successfully loaded GGUF model")

                _cached_llm_clients[cache_key] = llm
                return llm

            except Exception as e:
                log.warning(
                    "Failed to load GGUF model (%s)",
                    e,
                )

                if not FALLBACK_TO_BASE:
                    log.error("Fallback disabled, no LLM available")
                    return None

        # ------------------------------------------------------------------
        # Try Local Base Model (Offline)
        # ------------------------------------------------------------------
        if LOCAL_BASE_MODEL_PATH and os.path.exists(LOCAL_BASE_MODEL_PATH):
            try:
                import torch
                from langchain_huggingface import HuggingFacePipeline
                from transformers import (
                    AutoModelForCausalLM,
                    AutoTokenizer,
                    pipeline,
                    BitsAndBytesConfig,
                )

                log.info(
                    "Loading local base model from: %s",
                    LOCAL_BASE_MODEL_PATH,
                )

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                    try:
                        device_name = torch.cuda.get_device_name(0)
                    except Exception:
                        device_name = "unknown"

                    log.info("CUDA available: %s", device_name)

                    try:
                        total_mem = (
                            torch.cuda.get_device_properties(0).total_mem
                            / 1e9
                        )
                        free_mem = torch.cuda.mem_get_info()[0] / 1e9

                        log.info(
                            "CUDA memory: %.1f GB total, %.1f GB free",
                            total_mem,
                            free_mem,
                        )
                    except Exception as e:
                        log.warning(
                            "Unable to query CUDA memory: %s",
                            e,
                        )

                    log.info(
                        "Torch version: %s",
                        getattr(torch, "__version__", "unknown"),
                    )

                    log.info(
                        "CUDA_VISIBLE_DEVICES=%s",
                        os.environ.get("CUDA_VISIBLE_DEVICES"),
                    )

                    try:
                        import transformers

                        log.info(
                            "transformers version: %s",
                            getattr(
                                transformers,
                                "__version__",
                                "unknown",
                            ),
                        )
                    except Exception:
                        log.info(
                            "transformers not importable at this time"
                        )

                    try:
                        import bitsandbytes as bnb

                        log.info(
                            "bitsandbytes version: %s",
                            getattr(
                                bnb,
                                "__version__",
                                "unknown",
                            ),
                        )
                    except Exception:
                        log.info(
                            "bitsandbytes not available or failed to import"
                        )

                log.info(
                    "Loading tokenizer from: %s",
                    LOCAL_BASE_MODEL_PATH,
                )

                tokenizer = AutoTokenizer.from_pretrained(
                    str(LOCAL_BASE_MODEL_PATH),
                    trust_remote_code=True,
                )

                dtype = (
                    torch.float16
                    if torch.cuda.is_available()
                    else torch.float32
                )

                device_map = (
                    "auto"
                    if torch.cuda.is_available()
                    else None
                )

                model = None

                # ----------------------------------------------------------
                # Forced CPU offload
                # ----------------------------------------------------------
                if FORCE_CPU_OFFLOAD:
                    log.info(
                        "JOURNAL_CLUB_FORCE_CPU_OFFLOAD enabled: "
                        "forcing CPU offload for model load"
                    )

                    try:
                        offload_folder = (
                            Path(__file__).parents[1] / "hf_offload"
                        )

                        offload_folder.mkdir(
                            parents=True,
                            exist_ok=True,
                        )

                        try:
                            usage = shutil.disk_usage(
                                str(offload_folder)
                            )

                            log.info(
                                "Offload folder: %s "
                                "(free %.1f GB)",
                                offload_folder,
                                usage.free / 1e9,
                            )
                        except Exception:
                            log.info(
                                "Offload folder: %s",
                                offload_folder,
                            )

                        model = AutoModelForCausalLM.from_pretrained(
                            str(LOCAL_BASE_MODEL_PATH),
                            device_map={"": "cpu"},
                            torch_dtype=torch.float32,
                            low_cpu_mem_usage=True,
                            offload_folder=str(offload_folder),
                            trust_remote_code=True,
                        )

                        log.info(
                            "Loaded local base HuggingFace model "
                            "with forced CPU offload "
                            "(offload folder: %s)",
                            offload_folder,
                        )

                    except Exception as e:
                        log.error(
                            "Forced CPU offload load failed: %s",
                            e,
                        )

                # ----------------------------------------------------------
                # 8-bit CUDA load
                # ----------------------------------------------------------
                if model is None and torch.cuda.is_available():
                    try:
                        from transformers import BitsAndBytesConfig

                        log.info(
                            "Attempting 8-bit quantized load: "
                            "device_map=%s dtype=%s",
                            device_map,
                            dtype,
                        )

                        quantization_config = BitsAndBytesConfig(
                            load_in_8bit=True
                        )

                        model = AutoModelForCausalLM.from_pretrained(
                            str(LOCAL_BASE_MODEL_PATH),
                            device_map=device_map,
                            torch_dtype=dtype,
                            quantization_config=quantization_config,
                            trust_remote_code=True,
                        )

                        log.info(
                            "Successfully loaded local base "
                            "HuggingFace model (8-bit quantized)"
                        )

                    except Exception as e:
                        log.warning(
                            "8-bit quantized load failed (%s). "
                            "Retrying without quantization.",
                            e,
                        )

                        try:
                            log.info(
                                "CUDA memory allocated: %.3f GB",
                                torch.cuda.memory_allocated(0) / 1e9,
                            )

                            log.info(
                                "CUDA memory reserved: %.3f GB",
                                torch.cuda.memory_reserved(0) / 1e9,
                            )

                            log.info(
                                "CUDA memory summary:\n%s",
                                torch.cuda.memory_summary(
                                    device=0,
                                    abbreviated=True,
                                ),
                            )
                        except Exception as mem_e:
                            log.warning(
                                "Failed to get detailed CUDA memory "
                                "info: %s",
                                mem_e,
                            )

                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass

                        try:
                            model = AutoModelForCausalLM.from_pretrained(
                                str(LOCAL_BASE_MODEL_PATH),
                                device_map=device_map,
                                torch_dtype=dtype,
                                low_cpu_mem_usage=True,
                                trust_remote_code=True,
                            )

                            log.info(
                                "Loaded local base HuggingFace model "
                                "without 8-bit quantization "
                                "(using low_cpu_mem_usage)"
                            )

                        except Exception as e2:
                            log.error(
                                "Failed to load model after 8-bit "
                                "fallback: %s",
                                e2,
                            )

                            # ----------------------------------------------
                            # Final CPU offload fallback
                            # ----------------------------------------------
                            try:
                                offload_folder = (
                                    Path(__file__).parents[1]
                                    / "hf_offload"
                                )

                                offload_folder.mkdir(
                                    parents=True,
                                    exist_ok=True,
                                )

                                try:
                                    usage = shutil.disk_usage(
                                        str(offload_folder)
                                    )

                                    log.info(
                                        "Offload folder: %s "
                                        "(free %.1f GB)",
                                        offload_folder,
                                        usage.free / 1e9,
                                    )
                                except Exception:
                                    log.info(
                                        "Offload folder: %s",
                                        offload_folder,
                                    )

                                model = AutoModelForCausalLM.from_pretrained(
                                    str(LOCAL_BASE_MODEL_PATH),
                                    device_map={"": "cpu"},
                                    torch_dtype=torch.float32,
                                    low_cpu_mem_usage=True,
                                    offload_folder=str(offload_folder),
                                    trust_remote_code=True,
                                )

                                log.info(
                                    "Loaded local base HuggingFace "
                                    "model with CPU offload "
                                    "(offload folder: %s)",
                                    offload_folder,
                                )

                            except Exception as e3:
                                log.error(
                                    "CPU offload fallback failed: %s",
                                    e3,
                                )
                                raise

                # ----------------------------------------------------------
                # CPU load
                # ----------------------------------------------------------
                else:
                    model = AutoModelForCausalLM.from_pretrained(
                        str(LOCAL_BASE_MODEL_PATH),
                        device_map=device_map,
                        torch_dtype=dtype,
                        trust_remote_code=True,
                    )

                    log.info(
                        "Successfully loaded local base "
                        "HuggingFace model (full precision, CPU)"
                    )

                pipe = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=LLM_MAX_TOKENS,
                    temperature=LLM_TEMPERATURE,
                    do_sample=True,
                )

                llm = HuggingFacePipeline(pipeline=pipe)

                _cached_llm_clients[cache_key] = llm
                return llm

            except Exception as e:
                log.warning(
                    "Failed to load local base model (%s)",
                    e,
                )

                if torch and torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # ------------------------------------------------------------------
        # Fallback to direct OpenAI if available
        # ------------------------------------------------------------------
        try:
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(
                model=LLM_MODEL,
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
            )

            _cached_llm_clients[cache_key] = llm
            return llm

        except ImportError:
            log.warning("No LLM client available")
            return None

# ---------------------------------------------------------------------------
# Summary Generation
# ---------------------------------------------------------------------------

def generate_summary(paper: Dict[str, Any]) -> str:
    """Generate a concise summary of the paper."""

    title = paper.get("title", "")
    abstract = paper.get("abstract", "")

    if not abstract:
        return "No abstract available for summary."

    llm = get_llm_client()
    if llm is None:
        # Fallback: extract first few sentences
        sentences = re.split(r'[.!?]', abstract)
        return '. '.join(sentences[:3]) + '.'

    prompt = f"""Generate a concise 2-3 sentence summary of the following paper:

Title: {title}

Abstract: {abstract}

Focus on:
- Main research question/objective
- Key methods used
- Primary findings/conclusions
"""

    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [
            SystemMessage(content="You are an expert scientific summarizer. Generate clear, concise summaries of research papers."),
            HumanMessage(content=prompt),
        ]

        response = llm.invoke(messages)
        return _get_response_text(response).strip()

    except Exception as e:
        log.warning("LLM summary generation failed: %s", e)
        # Fallback
        sentences = re.split(r'[.!?]', abstract)
        return '. '.join(sentences[:3]) + '.'


# ---------------------------------------------------------------------------
# Gap Analysis
# ---------------------------------------------------------------------------

def analyze_gaps(paper: Dict[str, Any], domain: str = "general") -> Dict[str, List[str]]:
    """Analyze gaps in methodology, controls, statistics, and reproducibility."""

    title = paper.get("title", "")
    abstract = paper.get("abstract", "")

    combined_text = f"{title}\n\n{abstract}"

    llm = get_llm_client()
    if llm is None:
        # Rule-based fallback
        return _rule_based_gap_analysis(combined_text)

    prompt = f"""Analyze the following research paper for potential gaps and weaknesses:

Title: {title}

Abstract: {abstract}

Identify specific issues in these categories:

1. Methodology gaps: Missing or inadequate experimental approaches, limitations in study design
2. Missing controls: Appropriate controls that should have been included but weren't
3. Statistical issues: Sample size concerns, inappropriate statistical tests, p-hacking, etc.
4. Reproducibility concerns: Lack of detail, proprietary methods, data availability issues

For each category, provide 2-3 specific issues if present, or "None identified" if none.

Format your response as a JSON object with keys: methodology, controls, statistics, reproducibility
Each key should have a list of strings.
"""

    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [
            SystemMessage(content="You are an expert critical reviewer. Identify methodological and statistical gaps in research papers."),
            HumanMessage(content=prompt),
        ]

        # For structured JSON output tasks, disable thinking to save tokens for the actual response
        # This prevents the model from wasting its token budget on reasoning traces
        if hasattr(llm, 'invoke') and 'enable_thinking' in llm.invoke.__code__.co_varnames:
            response = llm.invoke(messages, enable_thinking=False)
        else:
            response = llm.invoke(messages)
        
        resp_text = _get_response_text(response).strip()

        # Try to parse with retry logic - pass LLM and messages for real retries
        parsed_result = _parse_gap_analysis_with_retry(
            resp_text,
            llm=llm,
            messages=messages,
            enable_thinking=False,
        )
        if parsed_result:
            return parsed_result
        
        log.warning("All parsing attempts failed, using rule-based fallback")
        return _rule_based_gap_analysis(combined_text)

    except Exception as e:
        log.warning("LLM gap analysis failed: %s", e)
        return _rule_based_gap_analysis(combined_text)


def _rule_based_gap_analysis(text: str) -> Dict[str, List[str]]:
    """Rule-based gap analysis as fallback."""
    text_lower = text.lower()

    gaps = {
        "methodology": [],
        "controls": [],
        "statistics": [],
        "reproducibility": [],
    }

    # Methodology indicators
    if "preliminary" in text_lower or "pilot" in text_lower:
        gaps["methodology"].append("Preliminary study with limited validation")

    if "in vitro" in text_lower and "in vivo" not in text_lower:
        gaps["methodology"].append("Only in vitro data, lacks in vivo validation")

    # Control indicators
    if "control" not in text_lower:
        gaps["controls"].append("No explicit mention of control experiments")

    # Statistical indicators
    if "p-value" not in text_lower and "p value" not in text_lower and "statistically" not in text_lower:
        gaps["statistics"].append("No statistical analysis reported")

    if "n=" not in text_lower and "sample size" not in text_lower:
        gaps["statistics"].append("Sample size not reported")

    # Reproducibility indicators
    if "data available" not in text_lower and "supplementary" not in text_lower:
        gaps["reproducibility"].append("Data availability not mentioned")

    if "proprietary" in text_lower or "commercial" in text_lower:
        gaps["reproducibility"].append("Potential proprietary methods limiting reproducibility")

    return gaps


# ---------------------------------------------------------------------------
# Critique Generation
# ---------------------------------------------------------------------------

def generate_critique(
    paper: Dict[str, Any],
    gap_analysis: Dict[str, List[str]],
    related_papers: List[Dict[str, Any]] | None = None,
) -> str:
    """Generate a structured critique of the paper."""

    title = paper.get("title", "")
    abstract = paper.get("abstract", "")

    # Build context from gap analysis
    gap_text = ""
    for category, issues in gap_analysis.items():
        if issues:
            gap_text += f"\n{category.capitalize()}: {', '.join(issues)}"

    llm = get_llm_client()
    if llm is None:
        return _rule_based_critique(paper, gap_analysis)

    related_context = ""
    if related_papers:
        related_titles = [p.get("title", "") for p in related_papers[:3]]
        related_context = f"\nRelated papers: {', '.join(related_titles)}"

    prompt = f"""Generate a structured critique of the following paper:

Title: {title}

Abstract: {abstract}
{related_context}

Identified gaps:{gap_text}

Provide a critique that:
1. Summarizes the paper's main contribution
2. Highlights key strengths
3. Discusses the identified gaps and their implications
4. Suggests how the study could be improved
5. Notes any contradictions with related work if applicable

Keep the critique concise (3-4 paragraphs) and constructive.
"""

    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [
            SystemMessage(content="You are an expert peer reviewer. Provide constructive, balanced critiques of research papers."),
            HumanMessage(content=prompt),
        ]

        response = llm.invoke(messages)
        return _get_response_text(response).strip()

    except Exception as e:
        log.warning("LLM critique generation failed: %s", e)
        return _rule_based_critique(paper, gap_analysis)


def _rule_based_critique(paper: Dict[str, Any], gap_analysis: Dict[str, List[str]]) -> str:
    """Rule-based critique as fallback."""
    title = paper.get("title", "")

    critique = f"This paper ({title}) "

    # Check for significant gaps
    total_gaps = sum(len(issues) for issues in gap_analysis.values())

    if total_gaps == 0:
        critique += "appears methodologically sound with no obvious gaps identified."
    elif total_gaps <= 2:
        critique += "has some minor limitations that should be addressed in future work."
    else:
        critique += "has several methodological concerns that limit the strength of its conclusions."

    # Add specific gap mentions
    if gap_analysis["methodology"]:
        critique += f" Methodological concerns include: {', '.join(gap_analysis['methodology'][:2])}."

    if gap_analysis["statistics"]:
        critique += f" Statistical issues: {', '.join(gap_analysis['statistics'][:2])}."

    if gap_analysis["reproducibility"]:
        critique += f" Reproducibility concerns: {', '.join(gap_analysis['reproducibility'][:2])}."

    return critique


# ---------------------------------------------------------------------------
# Quality Scoring
# ---------------------------------------------------------------------------

def score_paper_quality(
    paper: Dict[str, Any],
    gap_analysis: Dict[str, List[str]],
) -> Dict[str, float]:
    """Score paper on various quality dimensions."""

    title = paper.get("title", "")
    abstract = paper.get("abstract", "")
    combined_text = f"{title}\n\n{abstract}".lower()

    # Count gaps in each category
    methodology_gaps = len(gap_analysis.get("methodology", []))
    control_gaps = len(gap_analysis.get("controls", []))
    statistics_gaps = len(gap_analysis.get("statistics", []))
    reproducibility_gaps = len(gap_analysis.get("reproducibility", []))

    # Calculate scores (0-1 scale, higher is better)

    # Methodology rigor: fewer methodology gaps = higher score
    methodology_score = max(0.0, 1.0 - (methodology_gaps * 0.3))

    # Statistical power: fewer statistical gaps = higher score
    statistical_score = max(0.0, 1.0 - (statistics_gaps * 0.4))

    # Reproducibility: fewer reproducibility gaps = higher score
    reproducibility_score = max(0.0, 1.0 - (reproducibility_gaps * 0.3))

    # Control quality: fewer control gaps = higher score
    control_score = max(0.0, 1.0 - (control_gaps * 0.4))

    # Boost scores for positive indicators
    if "randomized" in combined_text or "randomised" in combined_text:
        methodology_score += 0.1

    if "blinded" in combined_text or "blind" in combined_text:
        methodology_score += 0.1

    if "replicate" in combined_text or "replication" in combined_text:
        reproducibility_score += 0.1

    if "sample size" in combined_text or "power analysis" in combined_text:
        statistical_score += 0.1

    # Clamp scores to 0-1
    methodology_score = min(1.0, methodology_score)
    statistical_score = min(1.0, statistical_score)
    reproducibility_score = min(1.0, reproducibility_score)
    control_score = min(1.0, control_score)

    # Overall quality (weighted average)
    overall_quality = (
        methodology_score * 0.3 +
        statistical_score * 0.25 +
        reproducibility_score * 0.25 +
        control_score * 0.2
    )

    return {
        "methodology_rigor": round(methodology_score, 2),
        "statistical_power": round(statistical_score, 2),
        "reproducibility_score": round(reproducibility_score, 2),
        "control_quality": round(control_score, 2),
        "overall_quality": round(overall_quality, 2),
    }


# ---------------------------------------------------------------------------
# Full Analysis Pipeline
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Full Analysis Pipeline
# ---------------------------------------------------------------------------

def analyze_paper(
    paper: Dict[str, Any],
    domain: str = "general",
    related_papers: List[Dict[str, Any]] | None = None,
    memory = None,
) -> Dict[str, Any]:
    """Run full analysis pipeline on a paper."""

    log.info("Analyzing paper: %s", paper)
    
    # Check cache first
    cached = get_cached_analysis(paper)
    if cached:
        log.info("Using cached analysis for: %s", paper.get("title", "unknown"))
        # Update memory to ensure analyzed_papers counter is accurate
        if memory is None:
            from .literature_memory import JournalClubMemory
            memory = JournalClubMemory()
        paper_key = paper.get("doi") or paper.get("pmid") or paper.get("title", "")
        memory.update_paper_analysis(
            paper_key,
            summary=cached.get("summary"),
            gap_analysis=cached.get("gap_analysis"),
            quality_scores=cached.get("quality_scores"),
            critique=cached.get("critique"),
        )
        return cached

    # Generate summary
    summary = generate_summary(paper)

    # Perform gap analysis
    gap_analysis = analyze_gaps(paper, domain)

    # Generate critique
    critique = generate_critique(paper, gap_analysis, related_papers)

    # Score quality
    quality_scores = score_paper_quality(paper, gap_analysis)

    result = {
        "summary": summary,
        "gap_analysis": gap_analysis,
        "critique": critique,
        "quality_scores": quality_scores,
    }
    
    # Save to cache
    save_cached_analysis(paper, result)

    log.info("Analysis complete for paper: %s", paper.get("title", "unknown"))
    return result


def analyze_batch(
    papers: List[Dict[str, Any]],
    domain: str = "general",
    related_papers_map: Dict[str, List[Dict[str, Any]]] | None = None,
    parallel: bool = True,
    memory = None,
) -> List[Dict[str, Any]]:
    """Run full analysis pipeline on a batch of papers.
    
    Args:
        papers: List of papers to analyze
        domain: Domain for analysis
        related_papers_map: Map of paper keys to related papers
        parallel: Whether to use parallel processing (default: True)
        memory: JournalClubMemory instance for updating analyzed papers counter
    
    Returns:
        List of analysis results
    """
    if parallel and len(papers) > 1:
        return analyze_batch_parallel(papers, domain, related_papers_map, memory)
    else:
        return analyze_batch_sequential(papers, domain, related_papers_map, memory)


def analyze_batch_sequential(
    papers: List[Dict[str, Any]],
    domain: str = "general",
    related_papers_map: Dict[str, List[Dict[str, Any]]] | None = None,
    memory = None,
) -> List[Dict[str, Any]]:
    """Run full analysis pipeline on a batch of papers sequentially."""

    log.info("Analyzing batch of %d papers (sequential)", len(papers))
    results = []

    for i, paper in enumerate(papers):
        log.info("Processing paper %d/%d: %s", i + 1, len(papers), paper.get("title", "unknown"))

        # Get related papers for this paper
        related = None
        if related_papers_map:
            paper_key = paper.get("doi") or paper.get("pmid") or paper.get("title", "")
            related = related_papers_map.get(paper_key, [])

        # Analyze the paper
        result = analyze_paper(paper, domain, related, memory)
        results.append(result)

    log.info("Batch analysis complete: %d/%d papers processed", len(results), len(papers))
    return results


def analyze_batch_parallel(
    papers: List[Dict[str, Any]],
    domain: str = "general",
    related_papers_map: Dict[str, List[Dict[str, Any]]] | None = None,
    memory = None,
) -> List[Dict[str, Any]]:
    """Run full analysis pipeline on a batch of papers in parallel."""
    
    log.info("Analyzing batch of %d papers (parallel, %d workers)", len(papers), MAX_ANALYSIS_WORKERS)
    results = []
    
    def analyze_single_paper(paper: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a single paper for parallel execution."""
        # Get related papers for this paper
        related = None
        if related_papers_map:
            paper_key = paper.get("doi") or paper.get("pmid") or paper.get("title", "")
            related = related_papers_map.get(paper_key, [])
        
        # Analyze the paper
        return analyze_paper(paper, domain, related, memory)
    
    with ThreadPoolExecutor(max_workers=MAX_ANALYSIS_WORKERS) as executor:
        future_to_paper = {
            executor.submit(analyze_single_paper, paper): paper 
            for paper in papers
        }
        
        for future in as_completed(future_to_paper):
            paper = future_to_paper[future]
            try:
                result = future.result()
                results.append(result)
                log.info("Completed analysis for: %s", paper.get("title", "unknown")[:50])
            except Exception as e:
                log.error("Analysis failed for %s: %s", paper.get("title", "unknown"), e)
                # Add empty result to maintain order
                results.append({
                    "summary": "Analysis failed",
                    "gap_analysis": {"methodology": [], "controls": [], "statistics": [], "reproducibility": []},
                    "critique": f"Analysis failed: {str(e)}",
                    "quality_scores": {"methodology_rigor": 0.0, "statistical_power": 0.0, "reproducibility_score": 0.0, "control_quality": 0.0, "overall_quality": 0.0},
                })
    
    log.info("Parallel batch analysis complete: %d/%d papers processed", len(results), len(papers))
    return results
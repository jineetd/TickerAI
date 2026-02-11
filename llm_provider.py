"""LLM Provider abstraction layer.

This module provides an extensible interface for different LLM providers.
Currently supports Ollama (local Llama models), but can be extended to
support OpenAI, Anthropic, or any other LLM provider.
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional

import ollama

import config

logger = logging.getLogger(__name__)


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def generate(
            self,
            prompt: str,
            system_prompt: Optional[str] = None,
            temperature: float = 0.7,
            max_tokens: int = 1000
    ) -> str:
        """Generate text completion.

        Args:
            prompt: The user prompt.
            system_prompt: Optional system prompt.
            temperature: Sampling temperature (0-1).
            max_tokens: Maximum tokens to generate.

        Returns:
            Generated text response.
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Get the current model name.

        Returns:
            Model name string.
        """
        pass


class OllamaProvider(BaseLLMProvider):
    """Ollama LLM provider for local Llama models."""

    def __init__(
            self,
            model: str = None,
            host: str = None
    ):
        """Initialize Ollama provider.

        Args:
            model: Model name (e.g., 'llama3.2', 'llama3.1').
            host: Ollama host URL.
        """
        self.model = model or config.LLM_MODEL
        self.host = host or config.LLM_HOST
        logger.info(f"Initialized Ollama provider with model: {self.model}")

    def generate(
            self,
            prompt: str,
            system_prompt: Optional[str] = None,
            temperature: float = None,
            max_tokens: int = None
    ) -> str:
        """Generate text using Ollama.

        Args:
            prompt: The user prompt.
            system_prompt: Optional system prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            Generated text response.
        """
        if temperature is None:
            temperature = config.LLM_TEMPERATURE
        if max_tokens is None:
            max_tokens = config.LLM_MAX_TOKENS

        # Combine system and user prompts
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        try:
            response = ollama.generate(
                model=self.model,
                prompt=full_prompt,
                options={
                    'temperature': temperature,
                    'num_predict': max_tokens,
                }
            )
            return response['response']

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            error_msg = (
                f"Error: {str(e)}\n\n"
                f"Make sure Ollama is running: ollama serve\n"
                f"And model '{self.model}' is installed: "
                f"ollama pull {self.model}"
            )
            raise RuntimeError(error_msg)

    def get_model_name(self) -> str:
        """Get model name.

        Returns:
            Model name string.
        """
        return self.model


class OpenAIProvider(BaseLLMProvider):
    """OpenAI LLM provider (for future use)."""

    def __init__(self, api_key: str = None, model: str = "gpt-4"):
        """Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key.
            model: Model name (e.g., 'gpt-4', 'gpt-3.5-turbo').
        """
        self.api_key = api_key
        self.model = model
        logger.info(f"Initialized OpenAI provider with model: {self.model}")

    def generate(
            self,
            prompt: str,
            system_prompt: Optional[str] = None,
            temperature: float = 0.7,
            max_tokens: int = 1000
    ) -> str:
        """Generate text using OpenAI (not implemented).

        Placeholder for future OpenAI integration.
        """
        raise NotImplementedError(
            "OpenAI provider not yet implemented. "
            "Install openai package and implement this method."
        )

    def get_model_name(self) -> str:
        """Get model name."""
        return self.model


def get_llm_provider(provider_type: str = None) -> BaseLLMProvider:
    """Factory function to get LLM provider.

    Args:
        provider_type: Type of provider ('ollama', 'openai', etc.).
                      Defaults to config.LLM_PROVIDER.

    Returns:
        LLM provider instance.

    Raises:
        ValueError: If provider type is not supported.
    """
    provider_type = provider_type or config.LLM_PROVIDER

    if provider_type == "ollama":
        return OllamaProvider()
    elif provider_type == "openai":
        return OpenAIProvider()
    else:
        raise ValueError(
            f"Unsupported LLM provider: {provider_type}. "
            f"Supported: ollama, openai"
        )

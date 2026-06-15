"""Helpers for loading API keys / credentials across environments.

Every key is resolved by trying, in order:
  1. Colab Secrets  (google.colab.userdata) — for running online in Colab
  2. A .env file    (python-dotenv)         — for local runs
  3. credentials.txt                        — the repo's original convention

Use ``load_llm_api_key`` for the (required) LLM key, and ``load_credential``
for optional human-annotation keys (AWS / Qualtrics / Prolific).
"""

import os

# Map each LLM provider to the environment-variable / secret name for its key.
PROVIDER_KEY_NAMES = {
    "openai": "OPENAI_API_KEY",
    "groq": "GROQ_API_KEY",
}


def _from_colab_secrets(key_name):
    try:
        from google.colab import userdata
    except ImportError:
        return None  # not running in Colab
    try:
        return userdata.get(key_name)
    except Exception:
        # Secret not set, or notebook access not enabled for it.
        return None


def _from_dotenv(key_name):
    try:
        from dotenv import load_dotenv
    except ImportError:
        return os.environ.get(key_name)  # dotenv not installed; check plain env
    load_dotenv()
    return os.environ.get(key_name)


def _from_credentials_txt(key_name, file_path="credentials.txt"):
    if not os.path.exists(file_path):
        return None
    with open(file_path, "r") as f:
        for line in f:
            if "=" in line:
                key, value = line.strip().split("=", 1)
                if key.strip() == key_name:
                    return value.strip() or None
    return None


def _resolve(key_name, credentials_file="credentials.txt"):
    """Return the value for ``key_name`` from the first source that has it."""
    for source in (
        lambda: _from_colab_secrets(key_name),
        lambda: _from_dotenv(key_name),
        lambda: _from_credentials_txt(key_name, credentials_file),
    ):
        value = source()
        if value:
            return value
    return None


def load_credential(key_name, credentials_file="credentials.txt"):
    """Return the value for ``key_name`` (Colab Secrets -> .env -> credentials.txt).

    Returns None if not found in any source. Use this for optional keys.
    """
    return _resolve(key_name, credentials_file)


def load_llm_api_key(provider, credentials_file="credentials.txt"):
    """Return the API key for ``provider`` ("openai" or "groq").

    Tries Colab Secrets, then a .env file, then credentials.txt. Raises a
    RuntimeError if no non-empty key is found in any source.
    """
    if provider not in PROVIDER_KEY_NAMES:
        raise ValueError(
            f"Unknown provider '{provider}'. Expected one of {list(PROVIDER_KEY_NAMES)}."
        )
    key_name = PROVIDER_KEY_NAMES[provider]
    key = _resolve(key_name, credentials_file)
    if not key:
        raise RuntimeError(
            f"Could not find {key_name} for provider '{provider}'. Set it in Colab "
            f"Secrets, a .env file, or {credentials_file}."
        )
    return key

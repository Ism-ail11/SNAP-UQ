from __future__ import annotations

def try_import_torchvision():
    try:
        import torchvision  # noqa: F401
        return True, None
    except Exception as e:
        return False, e

def try_import_torchaudio():
    try:
        import torchaudio  # noqa: F401
        return True, None
    except Exception as e:
        return False, e

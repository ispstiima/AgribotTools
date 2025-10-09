import os
import logging
from pathlib import Path
from dotenv import load_dotenv

log = logging.getLogger("converter")


def load_env(dotenv_path: str | Path | None = None, raise_on_missing=True) -> dict:
    """Load a .env file (optional) and return the resolved Label Studio root path.

    If `dotenv_path` is None the function will try to load the project's .env in
    the repository root. Raises EnvironmentError if the environment variable is
    not set after loading.
    """
    if dotenv_path is None:
        dotenv_path = Path(__file__).resolve().parents[1] / ".env"

    try:
        load_dotenv(dotenv_path)
    except Exception as e:
        log.debug("Ignored error loading dotenv %s: %s", dotenv_path, e)

    ls_root = os.getenv("LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT", None)
    if ls_root is None:
        msg = (
            "LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT environment variable not set. "
            "Please set it to the root directory of your Label Studio installation."
        )

        if raise_on_missing:
            raise EnvironmentError(msg)
        else:
            log.warning(msg)

    return {
        "LS_ROOT_PATH": Path(ls_root)
    }

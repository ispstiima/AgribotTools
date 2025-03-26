import os
import logging
from pathlib import Path


log = logging.getLogger("converter")
LS_ROOT_DIR = os.getenv("LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT", default=None)

if LS_ROOT_DIR is None:
    log.error("LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT environment variable not set. "
              "Please set it to the root directory of your Label Studio installation.")
    raise EnvironmentError("LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT environment variable not set.")

LS_ROOT_PATH = Path(LS_ROOT_DIR)

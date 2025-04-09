import shutil
import logging
from tqdm import tqdm
from pathlib import Path


log = logging.getLogger("Utils")
logging.basicConfig(format='%(asctime)s - %(name)s [%(levelname)s] | %(message)s', level=logging.INFO)


def copy_files_monitored(source_path: Path, dest_path: Path, dirs_exist_ok: bool = True, desc="Copying files") -> bool:
    log.info(f"Copying folder <{source_path}> to <{dest_path}>")

    if not source_path.exists():
        log.error("Source path for copy tree does not exist.")
        return False

    if dest_path.exists():
        if not dirs_exist_ok:
            log.error("Destination folder already exists. [Set dirs_exist_ok=True to overwrite]")
            return False
        log.warning("Overwriting destination folder.")
    else:
        dest_path.mkdir(parents=True, exist_ok=True)

    num_files = sum(1 for file in source_path.iterdir() if file.is_file())

    for source_file_path in tqdm(source_path.iterdir(), total=num_files, ascii="░▒█", desc=desc):
        if source_file_path.is_dir():
            log.warning("Directory found. Skipping it.")
            continue
        filename = source_file_path.name
        dest_file_path = dest_path / filename
        shutil.copy(source_file_path, dest_file_path)

    return True


import shutil
import logging
from tqdm import tqdm
from pathlib import Path


log = logging.getLogger("Utils")
logging.basicConfig(format='%(asctime)s - %(name)s [%(levelname)s] | %(message)s', level=logging.INFO)


def copy_files_monitored(source_path: Path, dest_path: Path, dirs_exist_ok: bool = True, desc: str = "Copying files") -> bool:
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


def sq_cp_dir_monitored(source_path: Path, dest_path: Path, files_ext: str, description: str):
    """Copies all files from a list of source directories to a destination directory."""
    if not dest_path.exists():
         dest_path.mkdir(parents=True)

    files_ext = [ext.strip() for ext in files_ext.split(",")]

    source_files = []
    for ext in files_ext:
        source_files.extend(source_path.rglob(f"*{ext}"))

    for source_file in tqdm(source_files, total=len(source_files), ascii="░▒█", desc=description):
        dest_file = dest_path / source_file.name
        try:
            shutil.copy2(source_file, dest_file)
        except Exception as e:
            log.error(f"Error copying {source_file} to {dest_file}: {e}")

    # total_files_copied = 0
    # for src_dir in source_dirs:
    #     if src_dir.is_dir():
    #         files_to_copy = [f for f in src_dir.iterdir() if f.is_file()]
    #         log.info(f"Found {len(files_to_copy)} files in {src_dir}")
    #         for src_file in tqdm(files_to_copy, desc=f"Copying {description} from {src_dir.name}", leave=False):
    #             dest_file = dest_path / src_file.name
    #             try:
    #                 shutil.copy2(src_file, dest_file) # copy2 preserves metadata
    #                 total_files_copied += 1
    #             except Exception as e:
    #                 log.error(f"Error copying {src_file} to {dest_file}: {e}")
    #     else:
    #          log.warning(f"Source directory not found, skipping: {src_dir}")
    # log.info(f"Finished copying {description}. Total files copied: {total_files_copied}")
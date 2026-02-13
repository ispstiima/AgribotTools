from pathlib import Path

def get_folder_path(files_list: list[str]):
    if not files_list or len(files_list) == 0:
        return None
    
    parents = set(Path(file_path).parent for file_path in files_list)
    
    min_level = min(len(p.parts) for p in parents)
    min_parents = [p for p in parents if len(p.parts) == min_level]

    if len(min_parents) > 1:
        raise ValueError("All files must be in the same directory.")
    
    return min_parents[0]


import shutil
from pathlib import Path
from .logger import default_logger

logger = default_logger(__name__)

def archive_file(file_path: Path, archive_root_dir: Path):
    """
    Moves a file to a corresponding subdirectory within the archive root.
    Example: /app/input_docs/invoice.pdf -> /app/archive/input_docs/invoice.pdf
    """
    if not file_path.exists():
        logger.warning(f"File to archive does not exist: {file_path}")
        return

    try:
        target_dir = archive_root_dir / file_path.parent.name
        target_dir.mkdir(parents=True, exist_ok=True)
        
        shutil.move(str(file_path), str(target_dir))
        logger.info(f"Archived file: {file_path.name} to {target_dir}")
    except Exception as e:
        logger.error(f"Failed to archive file {file_path.name}", exc_info=True)
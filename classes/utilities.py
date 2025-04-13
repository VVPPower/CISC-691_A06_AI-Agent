import shutil
import logging

logger = logging.getLogger(__name__)

def delete_directory(directory_path):
    try:
        # Delete the directory and all its contents
        shutil.rmtree(directory_path)
        logger.info(f"Directory '{directory_path}'  deleted.")
    except FileNotFoundError:
        logger.warning(f"Directory '{directory_path}' does not exist.")
    except PermissionError:
        logger.error(f"Permission denied to delete '{directory_path}'.")
        raise PermissionError
    except Exception as e:
        logger.error(f"An error occurred while deleting '{directory_path}': {e}")
        raise e


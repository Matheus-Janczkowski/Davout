# Bit of code to get back to the main folder of the Grante repository, 
# even though it is used in another folder. This bit will get the current
# file absolute path and add this path without the last 4 bits of path, 
# including the file name

import sys

from pathlib import Path

def find_repo_root(start_path: Path) -> Path:

    current = start_path.resolve()

    for parent in [current]+list(current.parents):

        # Searches for the .git folder

        if (parent / ".git").exists():

            return parent
        
    raise FileNotFoundError("No .git folder found in any parent direct"+
    "ory. Hence, this file is not in a valid git repository")

# Finds the module parent path

repository_root = find_repo_root(Path(__file__))

# if it is not already in the paths of the system, adds it

if not (str(repository_root) in sys.path):

    sys.path.insert(1, str(repository_root))
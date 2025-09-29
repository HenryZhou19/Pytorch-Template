# Delete __pycache__ folders in the current working directory and all subdirectories.
# If a parent directory becomes empty after deletion, remove it as well.
# With log output in English.

find "$(pwd)" -type d -name "__pycache__" | while read -r dir; do
  echo "Found and removed: $dir"
  rm -rf "$dir"
  parent="$(dirname "$dir")"
  # Recursively delete empty parent folders until reaching a non-empty folder or the current working directory
  while [ "$(ls -A "$parent" 2>/dev/null)" == "" ] && [ "$parent" != "$(pwd)" ]; do
    echo "Removed empty folder: $parent"
    rmdir "$parent"
    parent="$(dirname "$parent")"
  done
done
echo
echo "All __pycache__ folders and their empty parent directories have been cleaned up."
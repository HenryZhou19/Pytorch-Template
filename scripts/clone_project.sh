#!/bin/bash

echo ""
script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
excludsion_list_file="$script_dir/clone_exclusion.txt"

cd "$script_dir/.."

source_folder=$(pwd)
echo -e "Source project path：\n\t$source_folder"

destination_folder=$(readlink -m "$source_folder-cloned")
dry_run=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    # -s|-source)
    #   source_folder="$2"
    #   shift 2
    #   ;;
    -d|-destination)
      destination_folder="$2"
      shift 2
      ;;
    -n|-dryrun)
      dry_run=true
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;   
  esac
done

echo -e "Destination project path：\n\t$destination_folder"
echo -e "Exclusion："
cat $excludsion_list_file | sed 's/^/\t/'

if [ "$dry_run" = true ]; then
  echo -e "\n\nDry run, no files will be copied.\n"
  rsync -ahn --stats --exclude-from=$excludsion_list_file "$source_folder/" "$destination_folder/"
else
  echo -e "\n\nCloning project...\n"
  rsync -ah --stats --exclude-from=$excludsion_list_file "$source_folder/" "$destination_folder/"
  echo -e "\n\nProject cloned to \"$destination_folder\" successfully.\n"
fi
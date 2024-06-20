#!/bin/bash

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
excludsion_list_file="$script_dir/clone_exclusion.txt"

cd "$script_dir/.."
source_folder=$(pwd)

destination_folder=$(readlink -m "$source_folder-cloned")
dry_run=false

show_help() {
cat << EOF
discription:
    Clone the project to a new folder with the same structure and files.
    The files and folders in the 'scripts/clone_exclusion.txt' will be excluded.

usage:
    bash scripts/clone_project.sh [options]

options:
    [-h], --help
        Display this help and exit.

    [-d value], --destination
        Set the destination project path.
        Default: "'source_folder'-cloned"

    [-n], --dryrun
        Dry run. No files will be copied.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      show_help
      exit
      ;;
    # -s|-source)
    #   source_folder="$2"
    #   shift 2
    #   ;;
    -d|--destination)
      destination_folder="$2"
      shift 2
      ;;
    -n|--dryrun)
      dry_run=true
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;   
  esac
done

echo ""
echo -e "Source project path：\n\t$source_folder"

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
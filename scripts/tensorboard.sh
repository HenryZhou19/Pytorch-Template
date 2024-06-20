#!/bin/bash
script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$script_dir/.."

outputs_path="./outputs"
SPEC=false
PATH_SPEC=""

show_help() {
cat << EOF
discription:
    Open the tensorboard server to visualize the metric curves of the experiments in the outputs folder.

usage:
    bash scripts/tensorboard.sh [options]

options:
    [-h], --help
        Display this help and exit.

    [-p value], -outputs_path
        Set the path to the outputs folder. The tensorboard logs in the outputs folder and all its subfolders will be visualized.
        Default: "./outputs"
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      show_help
      exit
      ;;
    # -s|--source)
    #   source_folder="$2"
    #   shift 2
    #   ;;
    -p|--outputs_path)
      outputs_path="$2"
      shift 2
      ;;
    # -s|--do_spec)
    #   SPEC=true
    #   shift
    #   ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;   
  esac
done
if $SPEC; then
  for dir in $outputs_path/*; do
    if [ -d "$dir/tensorboard" ]; then
      log_dir="$dir/tensorboard"
      if [ -z "$PATH_SPEC" ]; then
        PATH_SPEC="$(basename "$dir"):$log_dir"
      else
        PATH_SPEC="$PATH_SPEC,$(basename "$dir"):$log_dir"
      fi
    fi
  done
  tensorboard --logdir_spec "$PATH_SPEC"
else
  cd "$outputs_path"
  current_dir=$(pwd)
  tensorboard --logdir "$current_dir"
fi

#!/bin/bash
script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$script_dir/.."

outputs_path="./outputs"
SPEC=false
PATH_SPEC=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    # -s|-source)
    #   source_folder="$2"
    #   shift 2
    #   ;;
    -p|-outputs_path)
      outputs_path="$2"
      shift 2
      ;;
    -s|-do_spec)
      SPEC=true
      shift 1
      ;;
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

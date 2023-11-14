#!/bin/bash

log_file_path="None"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -p|-log_file_path)
      log_file_path="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;   
  esac
done

if [[ "$log_file_path" == "None" ]]; then
  echo "Please specify the log file path by \"-p FILE_PATH\" option."
  exit 1
else
  python src/utils/plot/plot_metrics_from_log.py "$log_file_path"
fi

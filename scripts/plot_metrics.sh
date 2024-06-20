#!/bin/bash

log_file_path="None"

show_help() {
cat << EOF
discription:
    Use the log file of one experiment to plot the metrics as png files.
    Use 'src/utils/plot/plot_metrics_from_log.py' as the entry.

usage:
    bash scripts/plot_metrics.sh [options]

options:
    [-h], --help
        Display this help and exit.

    [-p value], --log_file_path
        Set the path to the log file.
        Default: "None" (exit)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      show_help
      exit
      ;;
    -p|--log_file_path)
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

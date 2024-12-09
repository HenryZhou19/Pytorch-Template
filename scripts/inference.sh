#!/bin/bash
devices="cpu"  # numbers with ',' or 'cpu'
omp_num_threads=4
mkl_num_threads=4
numexpr_num_threads=4
main_config_file_name="template_inference"
params=()

seconds_to_wait=0

show_help() {
cat << EOF
discription:
    Run the inference (test) process with the given config file and other options.
    Use 'inference.py' as the entry.

usage:
    bash scripts/inference.sh [options] [config1=value1] [config2=value2] ...

options:
    [-h], --help
        Display this help and exit.

    [-d value], --devices
        Choose the device to use. Either "cpu" or "[gpu_num_1],[gpu_num_2],...".
        Default: "cpu"

    [-c value], --config
        Set the main config file name to use in "./config/".
        Default: "template_inference"

    [-w value], --wait
        Set the seconds to wait before running the command.
        Default: 0

    [-ot value], --omp_threads
        Set the OMP_NUM_THREADS.
        Default: 4

    [-mt value], --mkl_threads
        Set the MKL_NUM_THREADS.
        Default: 4

    [-nt value], --numexpr_threads
        Set the NUMEXPR_NUM_THREADS.
        Default: 4

    [-e value], --extra_name
        Set the special extra name for the experiment. This equals to "special.extra_name=...".
        Default: "" (keep the same as in the yaml config file)

    [-p value], --train_cfg_path
        Set the train config path for the tester. This equals to "tester.train_cfg_path=...".
        Default: "" (keep the same as in the yaml config file)
EOF
}

run_cmd() {
  CUDA_VISIBLE_DEVICES=$devices \
  OMP_NUM_THREADS=$omp_num_threads \
  MKL_NUM_THREADS=$mkl_num_threads \
  NUMEXPR_NUM_THREADS=$numexpr_num_threads \
  WANDB_CACHE_DIR=~/.cache/wandb \
  WANDB_CONFIG_DIR=~/.config/wandb \
  torchrun \
  --nproc_per_node=$nproc_per_node \
  --master_port=$master_port \
  inference.py ${params[@]}
}

run_cpu_cmd() {
  OMP_NUM_THREADS=$omp_num_threads \
  MKL_NUM_THREADS=$mkl_num_threads \
  NUMEXPR_NUM_THREADS=$numexpr_num_threads \
  WANDB_CACHE_DIR=~/.cache/wandb \
  WANDB_CONFIG_DIR=~/.config/wandb \
  python \
  inference.py ${params[@]}
}

args=("$@")

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      show_help
      exit
      ;;
    -d|--devices)
      devices="$2"
      shift 2
      ;;
    -c|--config)
      main_config_file_name="$2"
      shift 2
      ;;
    -w|--wait)
      seconds_to_wait="$2"
      shift 2
      ;;
    -ot|--omp_threads)
      omp_num_threads="$2"
      shift 2
      ;;
    -mt|--mkl_threads)
      mkl_num_threads="$2"
      shift 2
      ;;
    -nt|--numexpr_threads)
      numexpr_num_threads="$2"
      shift 2
      ;;
    -e|--extra_name)
      extra_name="$2"
      shift 2
      ;;
    -p|--train_cfg_path)
      train_cfg_path="$2"
      shift 2
      ;;
    *)
      if [[ $1 == config=* ]]; then
        value="${1#config=}"
        main_config_file_name=$value
      else
        params+=("$1")
      fi
      shift
      ;;
  esac
done

echo -e "\033[?25l"  # hide cursor
trap 'echo -e "\033[0m\033[?25h"' INT  # change color back and show cursor when Ctrl-C

params="config.main=$main_config_file_name $params"
if [[ $extra_name != "" ]]; then
  params+=" special.extra_name=$extra_name"
fi
if [[ $train_cfg_path != "" ]]; then
  params+=" tester.train_cfg_path=$train_cfg_path"
fi

current_time=$(date +%s)
new_time=$((current_time + seconds_to_wait))
formatted_new_time=$(date -d "@$new_time" "+%Y-%m-%d %H:%M:%S")

echo "now: $(date "+%Y-%m-%d %H:%M:%S")"
echo "waiting for ${seconds_to_wait} seconds..."
echo -e "start at: ${formatted_new_time}\n"
sleep $seconds_to_wait

if [[ $devices == "cpu" ]]; then
  params+=" env.device=cpu"
  echo -e "\nRunning this task in cpu mode"

  run_cpu_cmd
else
  echo "OMP_NUM_THREADS: $omp_num_threads"
  echo "MKL_NUM_THREADS: $mkl_num_threads"
  echo "NUMEXPR_NUM_THREADS: $numexpr_num_threads"
  # params+=" env.device=cuda"  # default
  echo -e "\nRunning this task in cuda DDP mode\n"

  IFS=',' read -ra cuda_devices <<< $devices
  num_devices=${#cuda_devices[@]}
  nproc_per_node=$num_devices
  echo "CUDA_VISIBLE_DEVICES: $devices"
  echo "nproc_per_node: $nproc_per_node"
  start_port=25950
  end_port=25999
  master_port=$start_port

  while [ $master_port -le $end_port ]; do
    if netstat -tuln | grep -q ":$master_port "; then
      # echo "Port: $master_port is occupied."
      ((master_port++))
    else
      echo -e "\nTrying DDP with a potentially free port: $master_port"
      run_cmd
      if [ $? -eq 0 ]; then
          echo -e "\nDDP ran successfully with master_port: $master_port."
      else
          echo -e "\nDDP failed with master_port: $master_port. (Maybe triggered by other ERRORs)"
      fi
      break
    fi
    if [ $master_port -gt $end_port ]; then
      echo -e "\nAll ports from $start_port to $end_port are occupied."
      break
    fi
  done  
fi

echo -e "\n\"inference.sh ${args[@]}\" ends."
echo -e "\033[0m\033[?25h" # change color back and show cursor
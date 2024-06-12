#!/bin/bash
devices="6,7"  # numbers with ',' or 'cpu'
omp_num_threads=4
mkl_num_threads=4
numexpr_num_threads=4
main_config_file_name="template_train"
params=()

seconds_to_wait=0

run_cmd() {
  CUDA_VISIBLE_DEVICES=$devices \
  OMP_NUM_THREADS=$omp_num_threads \
  MKL_NUM_THREADS=$mkl_num_threads \
  NUMEXPR_NUM_THREADS=$numexpr_num_threads \
  WANDB_CACHE_DIR=~/.cache/wandb \
  WANDB_CONFIG_DIR=~/.config/wandb \
  torchrun \
  --nnodes=$nnodes \
  --nproc_per_node=$nproc_per_node \
  --node_rank=$node_rank \
  --master_addr=$master_addr \
  --master_port=$master_port \
  train.py --loglevel=ERROR with ${params[@]}
}

args=("$@")
echo -e "\033[?25l"  # hide cursor
trap 'echo -e "\033[0m\033[?25h"' INT  # change color back and show cursor when Ctrl-C

while [[ $# -gt 0 ]]; do
  case "$1" in
    -a|-addr)
      master_addr="$2"
      shift 2
      ;;
    -p|-port)
      master_port="$2"
      shift 2
      ;;
    -nn|-nnodes)
      nnodes="$2"
      shift 2
      ;;
    -n|node_rank)
      node_rank="$2"
      shift 2
      ;;
    -d|-devices)
      devices="$2"
      shift 2
      ;;
    -c|-config)
      main_config_file_name="$2"
      shift 2
      ;;
    -ot|-omp_threads)
      omp_num_threads="$2"
      shift 2
      ;;
    -mt|-mkl_threads)
      mkl_num_threads="$2"
      shift 2
      ;;
    -nt|-numexpr_threads)
      numexpr_num_threads="$2"
      shift 2
      ;;
    -e|-extra_name)
      extra_name="$2"
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
params="config.main=$main_config_file_name $params"
if [[ $extra_name != "" ]]; then
  params+=" special.extra_name=$extra_name"
fi

echo "start at: $(date "+%Y-%m-%d %H:%M:%S")"

if [[ $devices == "cpu" ]]; then
  echo "Cannot run DDP on multiple machines with only CPUs."
  exit 1
else
  echo "OMP_NUM_THREADS: $omp_num_threads"
  echo "MKL_NUM_THREADS: $mkl_num_threads"
  echo "NUMEXPR_NUM_THREADS: $numexpr_num_threads"
  # params+=" env.device=cuda"  # default
  echo -e "\nRunning this task in cuda DDP mode\n"

  IFS=',' read -ra cuda_devices <<< $devices
  num_devices=${#cuda_devices[@]}
  nproc_per_node=$num_devices
  echo "nnodes: $nnodes"
  echo "node_rank: $node_rank"
  echo "CUDA_VISIBLE_DEVICES: $devices"
  echo "nproc_per_node: $nproc_per_node"

  echo -e "\nTrying DDP with a potentially free port: $master_addr:$master_port"
  run_cmd
  if [ $? -eq 0 ]; then
      echo -e "\nDDP ran successfully with master_port: $master_port."
  else
      echo -e "\nDDP failed with master_port: $master_port. (Maybe triggered by other ERRORs)"
  fi
fi

echo -e "\n\"train.sh ${args[@]}\" ends."
echo -e "\033[0m\033[?25h" # change color back and show cursor
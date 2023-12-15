#!/bin/bash
cuda_devices="6,7"
omp_num_threads=4
mkl_num_threads=4
numexpr_num_threads=4
config_file_name="train"
params=()

seconds_to_wait=0

run_cmd() {
    CUDA_VISIBLE_DEVICES=$cuda_devices \
    OMP_NUM_THREADS=$omp_num_threads \
    MKL_NUM_THREADS=$mkl_num_threads \
    NUMEXPR_NUM_THREADS=$numexpr_num_threads \
    WANDB_CACHE_DIR=~/.cache/wandb \
    WANDB_CONFIG_DIR=~/.config/wandb \
    torchrun \
    --nproc_per_node=$nproc_per_node \
    --master_port=$master_port \
    train.py --loglevel=ERROR with ${params[@]}
}

args=("$@")
echo -e "\033[?25l"  # hide cursor
trap 'echo -e "\033[?25h"' INT  # show cursor when Ctrl-C

while [[ $# -gt 0 ]]; do
  case "$1" in
    -d|-devices)
      cuda_devices="$2"
      shift 2
      ;;
    -c|-config)
      config_file_name="$2"
      shift 2
      ;;
    -w|-wait)
      seconds_to_wait="$2"
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
    *)
      if [[ $1 == config=* ]]; then
        value="${1#config=}"
        config_file_name=$value
      else
        params+=("$1")
      fi
      shift
      ;;
  esac
done
params="config=$config_file_name $params"

current_time=$(date +%s)
new_time=$((current_time + seconds_to_wait))
formatted_new_time=$(date -d "@$new_time" "+%Y-%m-%d %H:%M:%S")

echo "now: $(date "+%Y-%m-%d %H:%M:%S")"
echo "waiting for ${seconds_to_wait} seconds..."
echo -e "start at: ${formatted_new_time}\n"
sleep $seconds_to_wait

IFS=',' read -ra devices <<< $cuda_devices
num_devices=${#devices[@]}
nproc_per_node=$num_devices

echo "CUDA_VISIBLE_DEVICES: $cuda_devices"
echo "nproc_per_node: $nproc_per_node"
echo "OMP_NUM_THREADS: $omp_num_threads"
echo "MKL_NUM_THREADS: $mkl_num_threads"
echo "NUMEXPR_NUM_THREADS: $numexpr_num_threads"

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
echo -e "\n\"train.sh ${args[@]}\" ends."
echo -e "\033[0m\033[?25h" # change color back and show cursor
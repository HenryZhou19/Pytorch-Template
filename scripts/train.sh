cuda_devices="6,7"
omp_num_threads=3
params="$@"

run_cmd() {
    CUDA_VISIBLE_DEVICES=$cuda_devices \
    OMP_NUM_THREADS=$omp_num_threads \
    WANDB_CACHE_DIR=~/.cache/wandb \
    WANDB_CONFIG_DIR=~/.config/wandb \
    torchrun --nproc_per_node=$nproc_per_node --master_port=$master_port train.py --loglevel=ERROR with $params\
        dummy=None \
        #
}



IFS=',' read -ra devices <<< $cuda_devices
num_devices=${#devices[@]}
nproc_per_node=$num_devices

echo "CUDA_VISIBLE_DEVICES: $cuda_devices"
echo "OMP_NUM_THREADS": $omp_num_threads
echo "nproc_per_node: $nproc_per_node"

master_port=25950
end_port=25955
used_port=""
keep_trying=false

while [ -z "$used_port" ] && [ $master_port -le $end_port ]; do
    echo -e "\nTrying master_port $master_port..."

    run_cmd

    if [ $? -eq 0 ]; then
        echo -e "\nDDP ran successfully with master_port $master_port."
        used_port=$master_port
    else
        echo -e "\nFailed to start DDP with master_port $master_port. (Maybe triggered by other ERRORs)"

        if [ "$keep_trying" = false ]; then
            read -p "Press Enter to continue searching for other master_port，or press 'Ctrl+C' to exit：" confirm
            if [ -z "$confirm" ]; then
                keep_trying=true
            fi
        fi
        master_port=$((master_port + 1))
    fi
done
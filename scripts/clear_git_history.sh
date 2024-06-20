commit_message="first commit"
remote_name="origin"
do_sync_remote=true

show_help() {
cat << EOF
discription:
    Clear the git history of the current branch and renew it using the current working directory.

usage:
    bash scripts/tensorboard.sh [options]

options:
    [-h], --help
        Display this help and exit.

    [-m value], --message
        Set the commit message for the renewing commit.
        Default: "first commit"

    [-ns], --no_sync
        Do not sync the renewing branch to the remote.

    [-r value], --remote
        Set the remote name to push the renewing branch. Only works when the --no_sync option is not set.
        Default: "origin"
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      show_help
      exit
      ;;
    -m|--message)
      commit_message="$2"
      shift 2
      ;;
    -ns|--no_sync)
      do_sync_remote=false
      shift
      ;;
    -r|--remote)
      remote_name="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;   
  esac
done

if $do_sync_remote; then
  do_sync_string_appendix=" and push it to remote \033[34m'$remote_name'\033[0m"
else
  do_sync_string_appendix=""
fi

delete_and_new_branch=$(git rev-parse --abbrev-ref HEAD)
echo -e "Are you sure you want to clear the git history of branch \033[33m'$delete_and_new_branch'\033[0m and renew it?"
echo -e "Enter 'yes' to confirm: " 
read confirmation
if [ "$confirmation" == "yes" ]; then
  echo -e "Enter \033[33mthe branch name\033[0m to confirm renewing or enter \033[35man new branch name\033[0m to archive it before renewing: "
  read confirmation_branch_name
  if [ "$confirmation_branch_name" == "$delete_and_new_branch" ]; then
    echo -e "\nRenew the branch \033[33m'$delete_and_new_branch'\033[0m$do_sync_string_appendix.\n\033[31mThe operation will be executed after five seconds... Press \033[32mCtrl+C\033[31m to cancel.\033[0m"
        sleep 5
    git checkout --orphan latest_temp_branch
    git add -A
    git commit -am "$commit_message"
    git branch -D $delete_and_new_branch
    git branch -m $delete_and_new_branch
    if $do_sync_remote; then
      git push -f $remote_name $delete_and_new_branch
    fi
    exit
  elif [ "$confirmation_branch_name" != "" ]; then
    echo -e "\nArchive the branch \033[33m'$delete_and_new_branch'\033[0m to \033[35m'$confirmation_branch_name'\033[0m$do_sync_string_appendix.\nRenew the branch \033[33m'$delete_and_new_branch'\033[0m$do_sync_string_appendix.\n\033[31mThese operations will be executed after five seconds... Press \033[32mCtrl+C\033[31m to cancel.\033[0m"
        sleep 5
    git branch -m $delete_and_new_branch $confirmation_branch_name 
    git checkout --orphan latest_temp_branch
    git add -A
    git commit -am "$commit_message"
    git branch -m $delete_and_new_branch
    if $do_sync_remote; then
      git push $remote_name $confirmation_branch_name
      git push -f $remote_name $delete_and_new_branch
    fi
    exit
  fi
fi
echo -e "\033[0mOperation cancelled."
delete_and_new_branch="main"
commit_message="first commit"
remote_name="origin"
do_sync_remote=true

while [[ $# -gt 0 ]]; do
  case "$1" in
    -b|-branch)
      delete_and_new_branch="$2"
      shift 2
      ;;
    -m|-message)
      commit_message="$2"
      shift 2
      ;;
    -r|-remote)
      remote_name="$2"
      shift 2
      ;;
    -ns|-no_sync)
      do_sync_remote=false
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;   
  esac
done

echo -e "Are you sure you want to clear the git history of branch \033[33m'$delete_and_new_branch'\033[0m and renew it?"
echo -e "Enter 'yes' to confirm: " 
read confirmation
if [ "$confirmation" == "yes" ]; then
  echo -e "Enter \033[33mthe branch name\033[0m to confirm renewing or enter \033[35man new branch name\033[0m to archive it before renewing: "
  read confirmation_branch_name
  if [ "$confirmation_branch_name" == "$delete_and_new_branch" ]; then
    if $do_sync_remote; then
      echo -e "\nRenew the branch \033[33m'$delete_and_new_branch'\033[0m and push it to remote.\n\033[31mThe operation will be executed after five seconds... Press \033[32mCtrl+C\033[31m to cancel.\033[0m"
    else
      echo -e "\nRenew the branch \033[33m'$delete_and_new_branch'\033[0m.\n\033[31mThe operation will be executed after five seconds... Press \033[32mCtrl+C\033[31m to cancel.\033[0m"
    fi
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
    if $do_sync_remote; then
      echo -e "\nArchive the branch \033[33m'$delete_and_new_branch'\033[0m to \033[35m'$confirmation_branch_name'\033[0m\nRenew the branch \033[33m'$delete_and_new_branch'\033[0m and push it to remote.\n\033[31mThe operation will be executed after five seconds... Press \033[32mCtrl+C\033[31m to cancel.\033[0m"
    else
      echo -e "\nArchive the branch \033[33m'$delete_and_new_branch'\033[0m to \033[35m'$confirmation_branch_name'\033[0m\nRenew the branch \033[33m'$delete_and_new_branch'\033[0m.\n\033[31mThe operation will be executed after five seconds... Press \033[32mCtrl+C\033[31m to cancel.\033[0m"
    fi
    sleep 5
    git branch -m $delete_and_new_branch $confirmation_branch_name 
    git checkout --orphan latest_temp_branch
    git add -A
    git commit -am "$commit_message"
    git branch -m $delete_and_new_branch
    if do_sync_remote; then
      git push $remote_name $confirmation_branch_name
      git push -f $remote_name $delete_and_new_branch
    fi
  fi
fi
echo -e "\033[0mOperation cancelled."
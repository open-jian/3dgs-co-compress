#!/usr/bin/env bash
set -euo pipefail

remote_host="${FCGS_REMOTE_HOST:-r3}"
remote_root="/data2/jian/outputs/wacv27_experiments/fcgs_langsplatv2/lerf_ovs"
local_root="/data2/jian/outputs/wacv27_experiments/fcgs_langsplatv2/lerf_ovs"
log_root="/data2/jian/outputs/wacv27_experiments/logs/fcgs_sync"
mkdir -p "$local_root" "$log_root"

sync_scene() {
    local scene="$1"
    local remote_scene="$2"
    local local_dir="$local_root/$scene"
    local remote_dir="$remote_root/$remote_scene"
    local log_file="$log_root/$scene.log"

    if [[ -f "$local_dir/.synced_from_rack3" ]]; then
        return 0
    fi

    while ! ssh -o BatchMode=yes "$remote_host" "test -f '$remote_dir/.complete'"; do
        sleep 20
    done

    mkdir -p "$local_dir"
    rsync -a --partial --exclude='.complete' \
        "$remote_host:$remote_dir/" "$local_dir/" >>"$log_file" 2>&1
    touch "$local_dir/.complete" "$local_dir/.synced_from_rack3"
    printf '%s synced %s -> %s\n' "$(date --iso-8601=seconds)" "$remote_scene" "$scene" >>"$log_file"
}

sync_scene ramen ramen &
sync_scene figurines figurines &
sync_scene teatime teatime &
sync_scene waldo_kitchen waldo_kitchen &
wait

#!/usr/bin/env bash
# Source this file, then call wait_for_available_gpu. The open flock descriptor
# is inherited by foreground child processes and released when the job exits.

wait_for_available_gpu() {
    [[ "${DEVICE:-cuda}" == "cpu" ]] && return 0
    if [[ ! "${DEVICE:-cuda}" =~ ^(cuda(:[0-9]+)?|[0-9]+)$ ]]; then
        echo "ERROR: invalid DEVICE=${DEVICE}" >&2
        return 1
    fi
    command -v nvidia-smi >/dev/null || { echo "ERROR: nvidia-smi is unavailable." >&2; return 1; }
    command -v flock >/dev/null || { echo "ERROR: flock is unavailable." >&2; return 1; }
    local threshold="${MIN_FREE_VRAM_MB:-40000}"
    local interval="${GPU_WAIT_INTERVAL_SECONDS:-3}"
    local timeout="${GPU_WAIT_TIMEOUT_SECONDS:-0}"
    if [[ ! "$threshold" =~ ^[0-9]+$ || ! "$interval" =~ ^[1-9][0-9]*$ || ! "$timeout" =~ ^[0-9]+$ ]]; then
        echo "ERROR: VRAM/timeout must be non-negative integers; poll interval must be positive." >&2
        return 1
    fi
    timeout=$((10#$timeout))
    local candidates="${GPU_CANDIDATES:-${CUDA_VISIBLE_DEVICES:-}}"
    if [[ "${DEVICE:-cuda}" =~ ^cuda:([0-9]+)$ ]]; then
        candidates="${BASH_REMATCH[1]}"
    elif [[ "${DEVICE:-cuda}" =~ ^[0-9]+$ ]]; then
        candidates="$DEVICE"
    fi
    candidates="${candidates//,/ }"
    local -a allowed=()
    read -r -a allowed <<< "$candidates"
    local candidate
    for candidate in "${allowed[@]}"; do
        if [[ ! "$candidate" =~ ^[0-9]+$ ]]; then
            echo "ERROR: GPU_CANDIDATES must contain physical GPU indices separated by spaces/commas." >&2
            return 1
        fi
    done
    local lock_dir="${GPU_LOCK_DIR:-${PROJECT_ROOT}/.cache/gpu_locks}"
    mkdir -p "$lock_dir"
    local selected="${MEMORYVLASEC_GPU_SELECTED:-}"
    local held_fd="${MEMORYVLASEC_GPU_LOCK_FD:-}"
    if [[ -n "$selected" ]]; then
        if [[ ! "$selected" =~ ^[0-9]+$ || ! "$held_fd" =~ ^[0-9]+$ ]] ||
           [[ ! -e "/proc/$$/fd/$held_fd" ]] || ! flock -n "$held_fd"; then
            echo "ERROR: inherited GPU selection does not have a valid lock descriptor." >&2
            return 1
        fi
        allowed=("$selected")
    fi
    local start=$SECONDS
    echo "Waiting for >=${threshold} MiB free on GPUs: ${allowed[*]:-all}; timeout=${timeout}s (0=unlimited)."
    while true; do
        local snapshot="" gpu_index free_vram eligible fd
        if snapshot="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null)"; then
            local -A detected=()
            while IFS=',' read -r gpu_index free_vram; do
                gpu_index="${gpu_index//[[:space:]]/}"
                if [[ "$gpu_index" =~ ^[0-9]+$ ]]; then detected[$((10#$gpu_index))]=1; fi
            done <<< "$snapshot"
            for candidate in "${allowed[@]}"; do
                if [[ "${detected[$((10#$candidate))]:-0}" != "1" ]]; then
                    echo "ERROR: requested physical GPU $candidate is absent from nvidia-smi." >&2
                    return 1
                fi
            done
            # Prefer the most free memory among the eligible, unlocked devices.
            while IFS=',' read -r gpu_index free_vram; do
                gpu_index="${gpu_index//[[:space:]]/}"
                free_vram="${free_vram//[[:space:]]/}"
                [[ "$gpu_index" =~ ^[0-9]+$ && "$free_vram" =~ ^[0-9]+$ ]] || continue
                eligible=0
                if (( ${#allowed[@]} == 0 )); then
                    eligible=1
                else
                    for candidate in "${allowed[@]}"; do
                        if (( 10#$candidate == 10#$gpu_index )); then eligible=1; fi
                    done
                fi
                (( eligible && 10#$free_vram >= 10#$threshold )) || continue
                if [[ -z "$selected" ]]; then
                    exec {fd}>"${lock_dir}/gpu_${gpu_index}.lock"
                    if ! flock -n "$fd"; then
                        exec {fd}>&-
                        continue
                    fi
                    # Re-query AFTER acquiring the advisory lock. Other projects
                    # do not use this lock, so VRAM is still not a reservation.
                    local fresh=""
                    fresh="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null | awk -F, -v wanted="$gpu_index" '$1+0==wanted {gsub(/[[:space:]]/, "", $2); print $2}')" || true
                    if [[ ! "$fresh" =~ ^[0-9]+$ ]] || (( 10#$fresh < 10#$threshold )); then
                        exec {fd}>&-
                        continue
                    fi
                    export MEMORYVLASEC_GPU_LOCK_FD="$fd"
                    export MEMORYVLASEC_GPU_SELECTED="$gpu_index"
                    free_vram="$fresh"
                fi
                export CUDA_VISIBLE_DEVICES="$gpu_index"
                export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-$gpu_index}"
                export DEVICE=cuda
                echo "Selected physical GPU $gpu_index ($free_vram MiB free); policy device=cuda:0."
                return 0
            done < <(printf '%s\n' "$snapshot" | sort -t, -k2,2nr)
        else
            echo "WARNING: nvidia-smi query failed; retrying." >&2
        fi
        if (( timeout > 0 && SECONDS - start >= timeout )); then
            echo "ERROR: GPU wait timed out after ${timeout}s." >&2
            return 124
        fi
        echo "No eligible unlocked GPU; checking again in ${interval}s."
        local delay="$interval"
        if (( timeout > 0 && timeout - (SECONDS - start) < delay )); then
            delay=$((timeout - (SECONDS - start)))
        fi
        sleep "$delay"
    done
}

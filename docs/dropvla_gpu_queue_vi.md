> Pipeline đã tách theo method. Các tên scripts ở root bên dưới là compatibility wrappers; source được seal là `attacks/dropvla/`. Xem [method layout](method_layout_vi.md).

# DropVLA: phases, GPU queue và recovery

Hướng dẫn hiện hành ngày 08/10/2026. Chạy lệnh từ project root.
Các tên `dropvla_example` và `dropvla_example_source` là đường dẫn ví dụ;
thay bằng run/snapshot của bạn. Không áp dụng lệnh chuẩn bị lại lên run đang chạy.

## Workflow

| Phase | Công việc | Điều kiện hoàn tất |
| --- | --- | --- |
| prepare | CPU: poison plan và full dataset audit | Audit đạt |
| preflight | Smoke 25 updates; ba inference ngắn; reference clean/post-grasp trên 10 tasks | Train/save/reload/render và trigger exposure đạt |
| train | 15.000 optimizer updates, accumulation 4 | Đủ budget, lưu checkpoint cuối |
| baseline | Pretrained clean, 200 episodes | Đủ episodes |
| clean | Checkpoint DropVLA không trigger, 200 episodes | Đủ episodes |
| trigger | Cùng checkpoint, post-grasp trigger, 200 episodes | Đủ episodes; ASR null khi thiếu denominator |
| summary | CPU: kiểm tra paired seeds/protocol và tổng hợp | Ba evaluations hoàn tất, nhất quán |

`evaluate` là alias tuần tự baseline → clean → trigger. Training dùng một GPU;
chờ nhiều GPU là quan sát pool rồi chọn một GPU, không cộng VRAM của cả pool.

## Chọn GPU và thời gian chờ

Scripts dùng `configs/runtime.env` mặc định ngưỡng **40.000 MiB**.
Queue resilient và recovery mặc định **28.672 MiB (28 GiB)**;
lệnh ví dụ bên dưới đặt rõ ngưỡng để tránh nhầm hai mặc định.

- Kiểm tra mỗi **3 giây**, `GPU_WAIT_TIMEOUT_SECONDS=0` là chờ vô hạn.
- GPU eligible nhiều free VRAM nhất và chưa bị lock được ưu tiên;
  kiểm tra lại sau khi lấy advisory lock.
- Phase giữ GPU lease; child scripts kiểm tra lại capacity trước mỗi model load.
  Giữa phases có kiểm tra và chọn GPU lại, có thể chọn GPU khác.
- Pool dùng physical indices từ `nvidia-smi`; policy dùng `cuda:0` sau khi đặt
  `CUDA_VISIBLE_DEVICES`. EGL được đặt theo GPU đã chọn.
- Advisory locks ở `.cache/gpu_locks/` chỉ phối hợp jobs dùng cùng cơ chế.
  Jobs khác vẫn có thể chiếm thêm VRAM sau khi chọn GPU.

## Chạy workflow mới

```bash
export PYTHON_BIN="$PWD/memoryvlasec/bin/python" # hoặc Python của môi trường khác
export GPU_CANDIDATES="0 1 2 3 4 5 6 7"
export MIN_FREE_VRAM_MB=28672
export GPU_WAIT_INTERVAL_SECONDS=3 GPU_WAIT_TIMEOUT_SECONDS=0
export DROPVLA_RUN_ROOT="$PWD/output/dropvla_example" # chưa tồn tại
bash scripts/run_dropvla.sh
```

Lệnh này chạy đầy đủ prepare → preflight → train → evaluations → summary.
Settings experiment lưu trong `run.env`; phase sau đọc lại settings đó.
Pool và ngưỡng GPU là thiết lập vận hành riêng. Không lưu credentials trong run.env.

Để seal source trước hàng đợi dài, dùng quy trình sau thay cho lệnh full run:

```bash
export PYTHON_BIN="$PWD/memoryvlasec/bin/python"
run_dir="$PWD/output/dropvla_example" # chưa tồn tại trước prepare
snapshot_dir="$PWD/output/dropvla_example_source" # snapshot mới
bash scripts/wait_dropvla.sh prepare "$run_dir"
(
  source "$run_dir/run.env"
  export DROPVLA_POISON_PLAN="$run_dir/poison_plan.json"
  export DROPVLA_AUDIT_OUTPUT="$run_dir/data_audit_current.json"
  bash scripts/audit_dropvla.sh
)
"$PYTHON_BIN" scripts/validate_dropvla_cpu.py "$run_dir" \
  --audit "$run_dir/data_audit_current.json" --seal "$snapshot_dir"
MIN_FREE_VRAM_MB=28672 "$PYTHON_BIN" scripts/dropvla_resilient_queue.py \
  "$run_dir" "$snapshot_dir"
```

Validation kiểm tra full audit và source hashes, cú pháp, tests CPU,
`pip check`, config run, pretrained assets và dung lượng checkpoint.
Snapshot sao chép source, bỏ quyền ghi và lưu manifest hashes; assets và Python
packages dùng chung, có version guards. GPU phases verify trước và sau khi chọn
GPU. Workspace sửa sau seal không sửa source trong snapshot.
GPU preflight vẫn cần chạy mới trước train.

## Chạy phase riêng

```bash
run_dir="$PWD/output/dropvla_example" # run đã prepare
bash scripts/wait_dropvla.sh preflight "$run_dir"
bash scripts/wait_dropvla.sh train "$run_dir"
bash scripts/wait_dropvla.sh evaluate "$run_dir"
bash scripts/wait_dropvla.sh summary "$run_dir"
```

Phase hoàn tất được bỏ qua; prerequisite thiếu thì dừng trước GPU wait.
Cùng phase/run có lock chống chạy trùng. Nếu dùng snapshot, gọi
`"$snapshot_dir/scripts/wait_dropvla.sh"` thay vì script workspace.
Checkpoint chỉ chứa weights/metadata; **không có optimizer resume**.
Không dùng checkpoint latest để tiếp tục training như chưa gián đoạn.

## Recovery chỉ inference

Sau khi preflight và train đã hoàn tất, dùng final checkpoint của run cũ:

```bash
run_dir="$PWD/output/dropvla_example"
snapshot_dir="$PWD/output/dropvla_example_source" # snapshot đã seal cho run này
bash scripts/resume_dropvla_inference.sh "$run_dir" "$snapshot_dir" --check-only
bash scripts/resume_dropvla_inference.sh "$run_dir" "$snapshot_dir"
```

Script yêu cầu preflight/train có done marker và exit code 0, final
`train/dropvla.pt` khác rỗng và snapshot verification đạt. Từ chối nếu phase
đang hoạt động. `--check-only` không chạy model hoặc archive artifacts.
Danh sách chạy chỉ gồm baseline → clean → trigger → summary; phase đã hoàn tất
được bỏ qua, tuyệt đối không thêm train vào recovery.

Recovery bật `CUDA_LAUNCH_BLOCKING=1` cho inference và xoá selection CUDA/EGL
thừa kế trước khi chọn GPU mới. Đồng bộ CUDA đã được dùng để khắc phục thực tế
lượt native render abort, nhưng chưa xác định nguyên nhân gốc; có thể làm
inference chậm hơn. Training không được bật setting này bởi script recovery.

Queue yêu cầu free VRAM đủ ngưỡng liên tục **60 giây**, GPU allocation failure
bị cooldown **30 phút**. Chỉ retry terminal CUDA OOM hoặc
`CUBLAS_STATUS_ALLOC_FAILED` ở preflight/evaluation; native abort, illegal
memory access và code/config errors làm queue dừng. Train không tự retry.
Evaluation retry archive artifacts của condition lỗi rồi chạy lại condition
đầy đủ, không tiếp tục từ episode dở. Preflight retry chạy lại toàn bộ checks.
Không khởi chạy recovery thứ hai khi một queue/phase đang hoạt động.

## Logs và theo dõi

```bash
run_dir="$PWD/output/dropvla_example"
tail -f "$run_dir/logs/train.log"
tail -f "$run_dir/logs/clean.log"
tail -f "$run_dir/logs/trigger.log"
cat "$run_dir/logs/trigger.json"
tail -f "$run_dir/resource_retry_events.jsonl"
```

Mỗi phase có `logs/PHASE.log`, `logs/PHASE.json`,
`logs/PHASE.resources.jsonl`, trỏ tới attempt mới nhất trong
`logs/PHASE/<UTC timestamp>-<PID>/`. Logs cũ được giữ; evaluation bị lỗi
có thể archive vào `resource_retry_history/`. `.phases/` ghi started/done,
exit code và GPU selection. Kết quả model ghi tại `train/train_metrics.jsonl`
và `evaluation/CONDITION/results.json`, cập nhật sau mỗi episode.

Log summary ghi thời điểm bắt đầu/kết thúc, elapsed, wait time, GPU, settings,
exit code và sampled memory peaks. Chu kỳ telemetry mặc định 5 giây,
đổi bằng `PHASE_RESOURCE_SAMPLE_SECONDS`; khác với GPU wait mỗi 3 giây.

- `device_used_mib`: toàn GPU, bao gồm jobs khác.
- `job_compute_used_mib`: compute memory của cây processes trong phase.
- Sampled peaks có thể bỏ lỡ transient; PyTorch allocated/reserved peaks ghi riêng.

Khi queue đợi GPU, phase summary vẫn có thể hiển thị attempt failed trước đó.
Đối chiếu event mới nhất và process queue để phân biệt đang đợi với đã dừng.
Lịch sử troubleshooting ở [history](history/README.md);
số đo/ước lượng ở [tài nguyên](dropvla_resource_estimates_vi.md).

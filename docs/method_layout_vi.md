# Pipeline theo method

## Entry points

- DropVLA: `python attacks/dropvla/main.py ...`; scripts tại `scripts/dropvla/`.
- BadVLA: `python attacks/badvla/main.py ...`; scripts tại `scripts/badvla/`.
- `main.py` ở root chỉ dispatch process theo `--attack`; baseline dùng `main_baseline.py`.
- Tên scripts cũ còn là wrappers để lệnh cũ tiếp tục hoạt động.

BadVLA dùng implementation remote tại commit `7c7dfd49`, với imports/paths đổi
sang method riêng. Defense hiện hành tại `defenses/` là A-MemGuard remote.
Model hiện hành tại `models/` vẫn dùng cho BadVLA/baseline.

## DropVLA đã hoàn thành experiment

`attacks/dropvla/` là runtime riêng từ snapshot `dropvla_backup/experiment_source`.
`_main_original.py`, `utils/`, `attacks/`, `models/`, `defenses/` giữ các implementation
của run đã hoàn tất; `main.py` chỉ bootstrap import đúng runtime.
Các modules `args.py`, `train.py`, `dataset.py`, `evaluate.py`, `metrics.py` là
facades; thuật toán gốc nằm dưới `utils/` để tránh rewrite imports/source không cần thiết.

Đây là ngoại lệ có chủ đích với model dùng chung: DropVLA giữ bản model và defense
đã dùng trong run cũ để không nhận thay đổi remote ngầm. Không dùng defense cũ này
để ghi đè A-MemGuard remote. Model/helper đã pin có thể được hợp nhất ở lần riêng
sau khi có kiểm tra tương đương; không làm điều đó trong lần tách thư mục.

`config.env` là cấu hình method. Runtime giữ `configs/runtime.env` gốc cho provenance.
`.cache`, `memoryvlasec`, `output` là links tới assets/environment/results chung;
chúng không phải copies và không commit. GPU waiter và telemetry tại `infra/`.
`reproduction_manifest.json` ghi hash nguồn gốc và các thay đổi đường dẫn sau relocation.

Không attach DropVLA và BadVLA runtimes trong cùng Python process: chúng dùng
các package upstream có tên top-level giống nhau. Dùng entry point hoặc root dispatcher.
Tests DropVLA được chạy subprocess để đảm bảo quy tắc này.

## Reproduce và rerun

Checkpoint/plan/results của experiment cũ vẫn ở run gốc, không sửa.
Để tái chạy cùng cấu hình cần giữ model/dataset revisions, poison plan, seed,
package versions, inference settings, và CUDA_LAUNCH_BLOCKING=1 cho evaluation.
Đổi cấu trúc source không bảo đảm GPU train lặp lại bit-for-bit; kiểm tra hash,
labels, gradients, checkpoint schema và metrics bảo toàn logic trước khi run mới.
Không dùng báo cáo pass cũ hoặc snapshot manifest cũ để chứng nhận source relocated.

```bash
export PYTHON_BIN="$PWD/memoryvlasec/bin/python"
export MIN_FREE_VRAM_MB=28672 GPU_WAIT_INTERVAL_SECONDS=3 GPU_WAIT_TIMEOUT_SECONDS=0
export DROPVLA_RUN_ROOT="$PWD/output/dropvla_reproduction" # run mới
bash scripts/dropvla/run_dropvla.sh
```

Để khớp run gốc, bật đồng bộ CUDA theo từng phase: chạy prepare/preflight/train bình thường rồi bật riêng baseline/clean/trigger theo
setting của từng condition đã lưu trong experiment cũ. Baseline cũ không bật,
clean/trigger recovery có bật. Dùng run.env cũ làm nguồn cấu hình, không tự đổi số episodes.

Tests: `python -m pytest tests/badvla tests/dropvla tests/infra`.

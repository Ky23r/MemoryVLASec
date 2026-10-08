# DropVLA: số đo tài nguyên và cách ước lượng

Cập nhật ngày 08/10/2026. Số đo bên dưới thuộc một run MemoryVLA trên H100
chia sẻ với jobs khác, không phải cam kết tài nguyên/thời gian cho mọi run.
Cấu hình: LLM LoRA32 + DiT head, vision/projector/memory frozen,
BF16, batch 1, accumulation 4, LLM activation checkpointing.

## Số đo thực tế của run ngày 06–08/10/2026

| Phase | Peak PyTorch allocated | Peak reserved | Peak job NVIDIA theo mẫu | Thời gian phase |
| --- | --- | --- | --- | --- |
| Train, 15.000 updates | 19,17 GiB | 19,25 GiB | 19,99 GiB | 10 giờ 38 phút |
| Baseline, 200 episodes | 15,99 GiB | 16,04 GiB | 17,24 GiB | 52 phút 55 giây |
| DropVLA clean, 200 episodes | 16,13 GiB | 16,21 GiB | 17,40 GiB | 9 giờ 27 phút |

Nguồn: phase summaries `logs/{train,baseline,clean}.json` của run local
`output/dropvla_full_20261005T113946Z`. Artifacts không được commit cùng docs;
muốn đối chiếu trên máy khác cần giữ/export các summary JSON tương ứng.
Phase times trên là thời gian attempt hoàn tất, gồm load/save và wait trong
phase; không gồm tất cả retries và thời gian queue đợi giữa attempts.

Trainer ghi 38.081 giây cho 15.000 updates, trung bình **2,54 giây/update**.
Baseline rollout ghi 3.007 giây, clean rollout ghi 33.827 giây.
Baseline chạy không đồng bộ CUDA; clean recovery bật `CUDA_LAUNCH_BLOCKING=1`.
GPU contention và số bước/episode cũng khác; không thể quy toàn bộ chênh lệch
thời gian cho checkpoint hoặc riêng setting đồng bộ.
Trigger chưa hoàn tất tại thời điểm lập bảng; không ghi thời gian dự đoán
thành số đo toàn condition.

## VRAM và ngưỡng GPU wait

Khoảng 20 GiB là memory riêng job train, không phải dung lượng toàn GPU.
Phần lớn base frozen không có gradients/optimizer states; batch 1 và checkpointing
hạn chế activation memory. Không áp dụng số này cho full fine-tune LLM,
train thêm memory, batch lớn hơn hoặc optimizer có FP32 moments.

Mặc định runtime thông thường vẫn 40.000 MiB free; resilient/recovery queue
mặc định 28.672 MiB free. Mức 28 GiB là ngưỡng vận hành có phần dư so với
số đo hiện tại, không phải bảo đảm chống OOM: jobs khác có thể dùng thêm VRAM
sau khi GPU được chọn. Không cộng free VRAM của nhiều GPU để chạy model này.
Xem [queue](dropvla_gpu_queue_vi.md).

## ETA cho run đang chạy

Ước lượng train từ mean seconds/update sau warmup và các optimizer steps thật.
Ước lượng evaluation từ completed episodes và elapsed của chính condition;
khi tốc độ GPU biến động, nên đối chiếu thêm tốc độ của các episodes gần nhất.
Task còn lại có độ dài khác nhau, nên ghi ETA là khoảng và kèm thời điểm đo.
Thời gian chờ GPU vô hạn không nằm trong ETA compute.

Không dùng tốc độ baseline để dự đoán trực tiếp clean/trigger khi đồng bộ CUDA
và tải GPU khác nhau. Không dùng smoke 25 updates để thay benchmark toàn train.

## Ước lượng lịch sử

[Bảng planning ngày 03–05/10](history/dropvla_resource_estimates_vi.md) giữ lại
các giả định và ngoại suy trước benchmark. Các mức 65–80+ GiB dành cho full
fine-tune downstream theo giả định optimizer; chúng **chưa được đo** trong
run LoRA + head hiện tại. Tốc độ smoke và tổng thời gian dự trù trong bảng cũ
không thay thế số đo run dài ở trên.

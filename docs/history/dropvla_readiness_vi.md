> **Tài liệu lịch sử.** Ghi nhận planning/kiểm tra ngày 03–06/10/2026; các trạng thái, lệnh và ước lượng bên dưới thuộc thời điểm đó, không chứng nhận code hiện tại. Xem [hướng dẫn hiện hành](../dropvla_run_vi.md) và [số đo tài nguyên](../dropvla_resource_estimates_vi.md).

> Cập nhật 06/10/2026: báo cáo bên dưới là lịch sử kiểm tra ngày 05/10,
> không chứng nhận code đã thay đổi sau đó. Bản sửa đồng bộ hiện tại đã chạy
> lại 30 tests trên CPU, audit đủ 432 episodes/52.970 frames/1.402 poison
> frames, kiểm tra dependencies, checkpoint pretrained và đường import.
> Code đã được sao chép vào snapshot riêng; GPU preflight mới đang chờ GPU,
> chưa có kết quả GPU mới và chưa bắt đầu main training.
> Báo cáo mới: readiness_current.json (`output/dropvla_repair_20261006T024933Z/readiness_current.json`, artifact local không được commit).

# DropVLA / MemoryVLA: kiểm tra sẵn sàng chạy

Kiểm tra ngày 05/10/2026. **Đã hoàn thành kiểm tra dữ liệu, train/checkpoint,
memory và real post-grasp. Chưa chạy training chính 15.000 updates.**
Workflow có các điều kiện kiểm tra trước train; lần native abort chưa có
nguyên nhân xác định, và hiện phải chờ GPU đủ threshold. Xem giới hạn dưới.

## Cấu hình đã chốt

Vision-only, poisoning theo released repo: chọn 5% source episodes, poison
mọi raw closed-gripper step trong các episodes đó. LoRA rank 32 trong LLM
+ train DiT action head; memory/compression/vision/projector frozen. Memory
vẫn hoạt động. Batch 1, BF16, accumulation 4, activation checkpointing LLM;
15.000 optimizer updates tương ứng 60.000 microbatches. Chi tiết và nguồn
adaptation ở [hướng dẫn chạy](../dropvla_run_vi.md).

Poison plan trên dữ liệu sạch: **432 episodes / 52.970 frames; 21 episodes
được chọn, 1.402 frames poison mỗi pass**. Raw labels được đổi trước khi
tạo action chunks, sáu chiều motion và clean normalization được giữ lại.

## Bằng chứng kiểm tra

Artifacts ở `output/dropvla_readiness_20261005/`.

| Kiểm tra | Kết quả |
| --- | --- |
| CPU tests | 14 tests pass: thêm real memory lifecycle, lỗi ghi checkpoint, pass thiếu frames và summary từ chối evaluations chưa đủ/không paired |
| Full data audit | Decode/quét đủ 52.970 frames / 432 episodes, thấy đủ 1.402 poison frames / 21 source episodes; census mới khớp plan; normalization khớp pretrained; assets cả 10 tasks có 50 initial states/task |
| Môi trường | `pip check`, Bash syntax và `git diff --check` pass; MuJoCo 2.3.7 / robosuite 1.4.0 chạy được |
| Train model thật | `poison_smoke_fixed/`: 25 updates / 100 microbatches; thấy 41 poison frames từ một selected episode, đi qua episode boundary; loss hữu hạn; wrapper exit 0 |
| Checkpoint | Full-state khoảng 16,9 GB; load strict với LoRA metadata, chạy lại policy được |
| Inference/rendering | Ba short conditions pretrained clean / DropVLA clean / always-trigger; 16 control steps mỗi condition, hai policy queries; đều hoàn thành |
| Rollout dài | Checkpoint smoke mới: post-grasp evaluation chạy đủ 220 steps, 28 policy queries; hoàn thành |
| Baseline sạch thật | Pretrained MemoryVLA: **3/3 episodes task 0 thành công**, 78–82 control steps; không đại diện cho toàn bộ 10 tasks |
| Checkpoint stress | Model thật 100 updates / 400 microbatches / 119 poison frames từ hai episodes; checkpoint định kỳ update 50 và final update 100 đọc lại được, tensor structure/metadata được kiểm tra trước atomic replace |
| Real post-grasp reference | Pretrained + LoRA zero-output, không optimizer updates: clean 10/10 successes và eligible grasps; triggered 10/10 eligible episodes được expose marker qua policy queries sau grasp; cả hai conditions hoàn thành toàn bộ 10 tasks |

Trong quá trình kiểm tra đã sửa lỗi iterator chỉ yield frame cuối episode,
thêm regression test và chạy lại GPU smoke thành công. Đã sửa mốc chiều
cao tính physical drop để lấy trước release command. Diagnostic smoke đưa
selected episode có onset sớm lên đầu, giữ cả clean prefix; production vẫn
shuffle episode và giữ thứ tự frame. Smoke phải thấy poison mới được pass.

Các attempts `smoke/` và `poison_smoke/` là artifacts kiểm tra trước đó;
checkpoint chuẩn để tham chiếu smoke cuối là `poison_smoke_fixed/dropvla.pt`.
Short inference trong `evaluation/` dùng checkpoint `smoke/`; rollout dài
trong `post_grasp_evaluation/` dùng checkpoint `poison_smoke_fixed/`.

## Giới hạn kết luận

Checkpoint chỉ train 25 updates chưa đạt grasp/lift trong episode post-grasp
đã kiểm tra: eligible episodes = 0, triggered queries = 0, ASR = `null`.
Giới hạn này của checkpoint smoke đã được bổ sung bằng reference riêng:
`reference_policy_validation_retry/{clean,post_grasp_trigger}/results.json`
quan sát được grasp/trigger onset thật ở cả 10 tasks. Reference là pretrained
+ LoRA zero-output, không có optimizer update: không dùng release rate của
reference làm kết quả attack. **Chưa xác nhận attack success hay clean
retention sau training chính.**

Reference attempt đầu bị native abort (exit 134) ở rollout đầu, chưa xác
định nguyên nhân. Chạy lại với `PYTHONFAULTHANDLER=1` và
`CUDA_LAUNCH_BLOCKING=1` hoàn thành cả 20 episodes, exit 0. Rendering riêng
trên GPU 3 cũng pass 32 steps. Không coi một lần retry thành công là bằng
chứng đã loại trừ mọi lỗi native; traceback được bật trong scripts để ghi
nhận nếu tái diễn. Cấu hình debug blocking làm chậm queries, không dùng thời
gian reference này làm throughput production.

## VRAM và thời gian

Số VRAM dưới là peak PyTorch allocated/reserved; không bao gồm toàn bộ CUDA
context hay renderer. Mỗi GPU có thể đang được người khác sử dụng.

| Phase đo thật | Peak allocated | Peak reserved | Thời gian đo |
| --- | --- | --- | --- |
| Train LoRA + head, accumulation 4 | 19,16 GiB | 19,25 GiB | 1,16 s/update sau warmup trên GPU 0; 30,2 s cho 25 updates, chưa gồm load/save |
| Pretrained clean inference | 15,99 GiB | 16,04 GiB | Query trung bình 0,17–0,23 s trong ba full episodes |
| DropVLA inference | 16,13 GiB | 16,21 GiB | Query trung bình 0,19 s trong rollout 220 steps; rollout khoảng 16 s, chưa gồm load |

Một smoke khác trên GPU chia sẻ đo 5,62 s/update sau warmup. Ngoại suy
15.000 updates cho khoảng **4,8–23,4 giờ phần training**, làm tròn planning
thành **5–24 giờ**. Benchmark ngắn và chưa đo periodic-save overhead dài hạn.
Chuẩn bị/load/save dự trù 0,5–1 giờ; ba evaluations, mỗi condition 200 episodes,
dự trù 1–3 giờ/condition. **Tổng dự trù 9–34 giờ**, không gồm chờ GPU;
ETA cần cập nhật bằng throughput/logs của run chính. Không cộng VRAM giữa
phases vì workflow chạy tuần tự.

Snapshot cuối kiểm tra: máy có 8 H100 80GB; GPU 0/2/3/4/6/7 có khoảng
79,2 GiB free, GPU 1 khoảng 77,6 GiB, GPU 5 khoảng 45,0 GiB. Đủ chạy cấu hình
đã kiểm tra. Script tự chọn GPU có ít nhất 40.000 MiB free và đặt renderer
EGL cùng GPU. Đây là kiểm tra dung lượng tại thời điểm gọi, không đặt chỗ GPU.
Ổ `/data` còn khoảng 9,7 TB; checkpoint latest và final khoảng 34 GB/run.

Snapshot cập nhật sau lượt kiểm tra bổ sung: GPU rảnh nhất khoảng 23.313 MiB,
chưa đạt threshold production 40.000 MiB. Số free phía trên thuộc lượt kiểm
tra trước đó; không dùng nó làm cam kết GPU đang sẵn. Workflow sẽ chờ.

## Khởi chạy

Từ root project:

```bash
bash scripts/run_dropvla.sh
```

Script tạo folder output mới, prepare poison plan → full data/asset audit
→ smoke 25 updates có poison → ba short inference checks → zero-output
reference kiểm tra real post-grasp → train 15.000 updates → ba evaluations
đầy đủ → summary kiểm tra đủ episodes và cùng seeds/protocol.
Run chính bắt đầu từ pretrained sạch, không tiếp tục weights smoke. Kết quả
gồm task SR, clean/trigger eligible release/drop rates và ASR với denominator
rõ ràng. Xem `train/train_metrics.jsonl` để theo dõi loss, poison exposure,
thời gian và VRAM; `evaluation/*/results.json` để đọc kết quả cuối.

Không còn câu hỏi cấu hình bắt buộc trước experiment đầu. Mở train thêm memory
hoặc giảm poison budget là các experiments tiếp theo, cần so sánh sau run này.

Training giờ dừng nếu finite pass thiếu frames hoặc chạy hết pass mà chưa
thấy đủ planned poison sources. `run_config.json` ghi versions và SHA256
source code. Save flush/fsync rồi đọc lại cấu trúc tensor/metadata trước
atomic replace; test lỗi ghi xác nhận checkpoint trước vẫn đọc được.
Checkpoint hiện chỉ chứa weights/metadata, **chưa có optimizer resume**.
Khi process bị ngắt, latest phục vụ evaluation; muốn tiếp tục optimizer phải
triển khai và kiểm tra resume riêng, không gọi load weights là resume.

Scheduler được bổ sung ở [hướng dẫn phase/GPU pool](../dropvla_gpu_queue_vi.md):
4 tests scheduler và 14 tests pipeline pass. Kiểm tra live pool với timeout
một giây trả đúng exit 124 khi không có GPU đạt 40.000 MiB; không load model.
Đã tạo scripts và kiểm tra cơ chế chờ/lock, chưa khởi động background queue
hay main training. Native-abort limitation ở trên vẫn còn được ghi nhận.

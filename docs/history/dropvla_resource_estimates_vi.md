> **Tài liệu lịch sử.** Ghi nhận planning/kiểm tra ngày 03–06/10/2026; các trạng thái, lệnh và ước lượng bên dưới thuộc thời điểm đó, không chứng nhận code hiện tại. Xem [hướng dẫn hiện hành](../dropvla_run_vi.md) và [số đo tài nguyên](../dropvla_resource_estimates_vi.md).

# DropVLA: poisoning đã chọn, nguồn LoRA và ước lượng tài nguyên

Planning ban đầu ngày 03/10/2026, cập nhật benchmark ngày 05/10/2026.
Người dùng chọn bám **released poisoning script**. Cấu hình đã triển khai:
LLM LoRA32 + DiT head, memory frozen, poison 5%, 15k updates/accumulation=4.
GPU smoke đã pass. Xem [báo cáo readiness](dropvla_readiness_vi.md) và
[hướng dẫn run](../dropvla_run_vi.md).

## Benchmark hiện tại, ưu tiên cho cấu hình chạy đầu

Model thật trên H100: peak train allocated **19,16 GiB**, reserved **19,25 GiB**;
inference allocated khoảng **16,0–16,1 GiB**, reserved **16,0–16,2 GiB**.
Đây là số PyTorch, chưa gồm CUDA context/renderer. Giữ threshold chọn GPU
40.000 MiB free để có dư dung lượng. Snapshot ngày 05/10: cả tám H100 có đủ
threshold, GPU 0 khoảng 79,2 GiB free.

Smoke có poison: 25 optimizer updates × accumulation 4 = 100 microbatches,
thấy 41 poison frames, 30,2 s phần train. Sau warmup khoảng **1,16 s/update**
trên GPU rảnh; smoke khác trên GPU chia sẻ khoảng **5,62 s/update**.
15k updates ngoại suy **4,8–23,4 giờ train**. Dự trù toàn workflow **9–34 giờ**
gồm prepare/load/save và ba evaluations 200 episodes/condition, chưa gồm
chờ GPU. Thời gian dài hạn/save và toàn bộ 10 tasks chưa benchmark.

Các bảng phía dưới là **planning lịch sử**, dùng cho so sánh những scope
chưa đo; không thay thế benchmark hiện tại cho LoRA + head/memory frozen.

## Poison budget đề xuất

Khuyên 5% cho experiment đầu, sau khi xác nhận clean retention và targeted
attack thì chạy thêm 0,31%. Mục đích là có nhiều episode/source context hơn
để phân biệt lỗi tích hợp với tín hiệu poison quá ít. Không suy ra hiệu quả
OpenVLA sẽ tự chuyển sang MemoryVLA.

Theo phép `int(N * rate)` của released script, 432 episode cho:

| Requested budget | Poison episodes | Realized budget |
| --- | --- | --- |
| 5% | 21 | 4,861% |
| 0,31% | 1 | 0,231% |

Lựa chọn vision-only trong README dùng step_ratio=1: chọn mọi raw closed-
gripper step trong các episode đã sample. Poison target labels ở trajectory
gốc rồi tạo future chunks; không dùng per-batch hash hiện tại, không đổi
8 action đầu ở mọi chunk. Giữ 6 chiều motion; raw gripper +1 thành -1, qua
adapter MemoryVLA trở thành target 1=open.
[Released script](https://github.com/megaknight114/DropVLA/blob/main/visual_backdoor_attack.py),
[README](https://github.com/megaknight114/DropVLA#build-a-backdoor-dataset).

Hạ budget không tiết kiệm phần lớn thời gian train: vẫn train mixed dataset
clean+poison. Muốn so sánh budget, giữ training schedule và sampling giống
nhau, thay poison plan. Budget 0,31% nhạy với episode/seed được chọn; report
actual poisoned frames/episodes và số poison samples optimizer đã thấy.

## Nguồn gốc fine-tuning và adaptation

DropVLA paper §IV-B mô tả LoRA rank 32 với OpenVLA-OFT. Training script tạo
LoRA với target_modules=all-linear; action head và proprio projector được
đưa riêng vào optimizer khi bật. **Paper/repo DropVLA không có MemoryVLA
memory banks.**
[Paper](https://arxiv.org/html/2510.10932v5),
[Training script](https://github.com/megaknight114/DropVLA/blob/main/vla-scripts/finetune.py).

Do đó ba lựa chọn adaptation khác nhau:

1. LoRA trên LLM + train action_model DiT, giữ memory/compression/vision/
   projector frozen: đề xuất experiment đầu nếu cho phép LoRA.
2. Như trên + train cog/per memory và per_compr: thêm khả năng thích nghi
   memory, nhưng là thay đổi experiment do chúng ta lựa chọn.
3. Update full downstream base weights: gần đường code hiện tại, tốn nhiều
   optimizer memory hơn. Không mang selection q/k/v/o của BadVLA sang một
   experiment DropVLA mà không mô tả rõ.

Scope “LLM-only LoRA” cũng là lựa chọn adaptation; released script dùng
all-linear trên VLA. Các ước lượng dưới chỉ áp dụng scope LLM đã nêu, không
áp dụng tự động nếu mở LoRA cho vision/projector.

## Cơ sở tính VRAM

Đọc checkpoint local bằng mmap, không cấp phát model lên GPU:

| Thành phần | State elements xấp xỉ |
| --- | --- |
| Toàn model | 8,377B, weights bfloat16 = 15,60 GiB |
| Vision | 0,731B |
| LLM | 6,739B |
| Projector | 0,071B |
| ActionModel DiT | 0,409B |
| Cognition memory | 0,424B |
| Perception memory + compression | 0,003B |

LoRA rank 32 trên 224 linear matrices trong transformer layers LLM:
79.953.920 trainable adapter elements. Riêng q/k/v/o: 33.554.432.
Head/memory/compression có khoảng 835,5M state elements. Counts lấy từ
state_dict gồm cả buffers nhỏ; đây không phải model.named_parameters count.

Lower-bound storage = base weights + adapter weights + gradients + optimizer
moments. Với parameter/grads/moments cùng bfloat16, mỗi trainable parameter
thêm 6 bytes ngoài weight đã đếm. FP32 moments thêm 10 bytes. Không cộng
activations, CUDA workspace, fragmentation, temporary buffers ở bước này.
CPU probe AdamW trong môi trường local cho moments cùng dtype bfloat16 với
parameter bfloat16; optimizer/adapters khác phải kiểm tra lại.

| Profile, batch=1, BF16 base | Storage trước activations | Peak budget dự kiến |
| --- | --- | --- |
| LLM LoRA32 + DiT head, memory frozen | khoảng 18–21 GiB | 24–35 GiB |
| LLM LoRA32 + head/memory/compression | khoảng 21–25 GiB | 28–40 GiB |
| Full downstream, BF16 optimizer state | khoảng 58 GiB | 65–80+ GiB |
| Full downstream, FP32 moments | khoảng 87 GiB | vượt 80 GiB trước activations |
| Inference BF16, DDIM10, batch=1 | weights 15,6 GiB | 18–25 GiB |

“Full downstream” ở đây giữ vision frozen, cập nhật LLM, projector, action
head và memory/compression. Nếu mở cả vision thì cần tính thêm optimizer
state và activations; không được áp dụng nguyên các số này.

Peak budget là phỏng đoán dành cho activation checkpointing và foreach=False.
Không đảm bảo vừa 40 GiB, đặc biệt khi batch/group lớn hơn 1. QLoRA/4-bit
có thể giảm frozen weights nhưng chưa có tích hợp quantization cho model local, nên không
dùng mức VRAM OpenVLA 4-bit làm số liệu MemoryVLA hiện tại.

Snapshot lịch sử ngày 03/10: GPU rảnh nhất khoảng 20.107 MiB, chưa đạt
threshold. Snapshot ngày 05/10 ở đầu tài liệu đã thay thế trạng thái này.

## Ước lượng theo phase

Giả định một H100 có đủ VRAM, BF16, batch=1, accumulation=1 khi nói một
training update, LLM checkpointing, AdamW foreach=False, inference DDIM10
và execute 8 actions/query. Không gồm thời gian sửa code hay chờ GPU.

| Phase | VRAM dự kiến | Wall time dự kiến |
| --- | --- | --- |
| Đọc corpus, lập poison plan, kiểm tra labels | Không dùng GPU | 1–5 phút |
| Materialize poisoned RLDS nếu chọn offline cache | Không dùng GPU | 5–30 phút |
| Load model/attack checkpoint | 16–20 GiB | 3–10 phút |
| Smoke 100 microbatches + warmup | Theo train profile | 5–15 phút gồm load |
| LoRA train, một microbatch | 24–40 GiB tùy memory policy | 0,8–2 giây |
| Full downstream train, một microbatch | 65–80+ GiB | 1,2–3 giây |
| Save checkpoint | Không cần tăng đáng kể GPU memory | 0,5–3 phút mỗi save |
| Một inference query, DDIM10 | 18–25 GiB | 0,3–1 giây |
| Một condition LIBERO, 200 episodes | 18–25 GiB | 1–3 giờ |
| A-MemGuard calibration, 256 transitions | Inference profile | 5–15 phút gồm load |
| Defended rollout 200 episodes | Inference profile + filter overhead | Dự trù 1–4 giờ, chưa đo overhead |

Khi lập bảng lịch sử này, LoRA/full step time chưa đo. Mốc tham khảo: BadVLA Stage II log
30.000 update trong 9:21:34, trung bình 1,123 s/update. Dùng làm sanity check
cho khoảng planning, không xem là tốc độ DropVLA. Step time đã bao gồm
forward/backward/optimizer/data cho giả định này; không cộng từng phase
GPU riêng khi chưa có profiler.

## Tổng thời gian theo schedule

| Schedule | Microbatches batch=1 | LoRA train | Full downstream train |
| --- | --- | --- | --- |
| 15k optimizer updates, accumulation=1 | 15.000 | 3,3–8,3 giờ | 5–12,5 giờ |
| Một lượt đủ 52.970 frame, accumulation=1 | 52.970 | 11,8–29,4 giờ | 17,7–44,1 giờ |
| 15k optimizer updates, accumulation=4 | 60.000 | 13,3–33,3 giờ | 20–50 giờ |

Accumulation=4 nay đã được triển khai và kiểm tra. Với giả định lịch sử
accumulation=1, batch=1/15k updates chỉ dùng 15k samples;
không gọi nó là một lượt đầy đủ dataset. Với stream theo thứ tự, phải kiểm
tra poison exposure để không kết thúc trước khi tới selected episodes.

LoRA + một lượt đủ corpus + chuẩn bị/checkpoint + ba conditions evaluation
(pretrained clean, DropVLA clean, DropVLA trigger) dự trù **15–40 giờ** cho
một poison budget/seed. Full downstream tương ứng **21–55 giờ**, có thể OOM
trên 80 GiB theo optimizer/activation thực tế.

Nếu chọn 15k updates/accumulation=1, tổng LoRA khoảng **7–19 giờ**, full
downstream khoảng **9–23 giờ**; đây là run ngắn hơn, không cùng data exposure.
Nếu chọn 15k updates/accumulation=4, tổng LoRA khoảng **17–44 giờ**, full
downstream khoảng **24–61 giờ**, cùng ba conditions evaluation. Đây vẫn là
batch=1 của phương án MemoryVLA, không phải effective batch gốc của OpenVLA.
Hai poison budgets cần train hai model; clean reference evaluation có thể
tái sử dụng nếu mọi cấu hình inference/initial states giống nhau. LoRA/một
lượt corpus/hai budgets dự trù khoảng **30–75 giờ**. A-MemGuard thêm calibration
và defended evaluation cho mỗi attack checkpoint; ba seeds làm training và
evaluation lặp lại gần ba lần.

Inference estimate dùng tối đa 5.600 policy queries và 44.000 environment
steps mỗi condition: 200 episodes × ceil(220/8) queries. Phần neural query
0,3–1s cộng environment step 0,02–0,1s cho khoảng 0,7–2,8 giờ, làm tròn
planning thành 1–3 giờ. Early completion làm thời gian giảm; replan nhiều hơn,
DDPM100 hoặc GPU contention làm thời gian tăng.

Phải benchmark 50–100 microbatches và một nhóm rollout trước khi báo ETA
chính thức. Đo cả optimizer step sau khi state đã được tạo, peak allocated/
reserved memory và samples/second; không dùng thời gian warmup làm steady-
state throughput. Thời gian chờ GPU chưa có giới hạn và không nằm trong các
tổng phía trên.

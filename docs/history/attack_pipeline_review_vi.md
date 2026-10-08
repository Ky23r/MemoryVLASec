> **Tài liệu lịch sử.** Ghi nhận planning/kiểm tra ngày 03–06/10/2026; các trạng thái, lệnh và ước lượng bên dưới thuộc thời điểm đó, không chứng nhận code hiện tại. Xem [hướng dẫn hiện hành](../dropvla_run_vi.md) và [số đo tài nguyên](../dropvla_resource_estimates_vi.md).

# Đánh giá BadVLA và chuẩn bị DropVLA trên MemoryVLA

Audit này ghi lại trạng thái trước khi sửa DropVLA. Implementation và scripts
mới nhất được mô tả ở [dropvla_run_vi.md](../dropvla_run_vi.md).

Hướng chuẩn bị đã được làm rõ thêm trong
[dropvla_adaptation_plan_vi.md](dropvla_adaptation_plan_vi.md): tái sử dụng
hạ tầng BadVLA, giữ protocol DropVLA riêng và chọn adaptation phù hợp MemoryVLA.
Các profile trong audit bên dưới là đề xuất ban đầu, không phải yêu cầu hai
method có cùng loader, batch size hoặc lịch training.

Ngày kiểm tra: 03/10/2026, múi giờ Asia/Bangkok. Mã nguồn tại commit
`bc0ec4f` và working tree không có thay đổi tracked trước cuộc kiểm tra.
Môi trường thực tế: `memoryvlasec/bin/python3.10` trong repo.

**Kết luận: BadVLA đã có artifact Stage II từ một lần chạy hoàn tất, nhưng
PID Stage I còn tồn tại chưa chứng minh đang tiến triển. DropVLA hiện chưa
sẵn sàng cho một lần training có kết quả khoa học đáng tin cậy.** Các lỗi cần
xử lý trước gồm nhãn chồng lấp, dtype đầu vào, dữ liệu hữu hạn, normalization,
key statistics và ngân sách bộ nhớ. Cuộc kiểm tra không dừng PID, không sửa
checkpoint, không khởi chạy training, không thay đổi implementation attack.

## 1. Phạm vi và cấu trúc repo

- `scripts/*.sh` + `configs/runtime.env`: chọn GPU, kiểm tra marker assets,
  truyền cấu hình và đường dẫn artifact.
- `main.py`: parse/validate arguments, load pretrained MemoryVLA, compose
  `SecureVLA`, load project checkpoint và dispatch train/evaluate.
- `models/base_memory_vla.py` → `models/core/vla/load.py`: tải kiến trúc và
  trọng số pinned; CUDA mặc định dùng `bfloat16`.
- `models/secure_vla.py`: forward proxy. Trigger không được tự động chèn ở
  wrapper; training/evaluation phải chèn tường minh.
- `attacks/badvla.py`: trigger và feature objective.
- `attacks/dropvla.py`: lựa chọn episode, trigger và sửa action chunk.
- `utils/dataset.py`: RLDS, local trajectory, flat manifest, batch lifecycle.
- `utils/train.py`: trainable modules, optimizer, stage loops và checkpoint.
- `utils/libero_evaluate.py`: rollout thật, trigger và task success.
- `defenses/amemguard.py`: inference filter; không nằm trong hai training run.

`memoryvlasec/` là môi trường Conda local, `.conda_pkgs/` là package cache,
`.cache/memoryvlasec/` là assets/artifacts; chúng không phải mã attack cần đọc.
Không tìm thấy AGENTS.md áp dụng cho repo.

## 2. Trạng thái BadVLA thực tế

### Tiến trình còn tồn tại

| Thuộc tính | Quan sát |
| --- | --- |
| PID Python | `106977` |
| PID script cha | `3658029`, `bash scripts/train_badvla.sh` |
| Argument stage | `--attack_stage stage1` |
| GPU vật lý | `CUDA_VISIBLE_DEVICES=7` |
| VRAM của PID | 19.188 MiB, khoảng 18.74 GiB |
| RAM RSS | Khoảng 227 GiB |
| stdout/stderr | `/dev/pts/51`, không trỏ vào `train_badvla.log` |
| Mẫu quan sát 3 giây | CPU tick delta = 0; `/proc/106977/io` không đổi |
| Main thread | `wait_woken` |

PID còn sống không đồng nghĩa optimizer đang chạy. Không đọc được kernel
stack/syscall do quyền ptrace; chưa xác định được chính xác tiến trình đang
chờ terminal, chờ dữ liệu hay mắc lỗi khác. Không thể báo step/loss hiện tại
của PID này từ log của một lần chạy khác.

### Lần chạy đã ghi log

`train_badvla.log` bắt đầu bằng chọn **GPU 1**, tái sử dụng Stage-I checkpoint,
rồi chạy Stage II. Cuối log ghi `BadVLA Stage II checkpoint saved ...`.
Progress cuối là index `29999`, phù hợp vòng lặp đã thực hiện 30.000 update;
loss cuối `0.0027`, elapsed khoảng 9 giờ 21 phút. Con số 57% trên tqdm là
30.000/52.970 transition danh nghĩa, không phải training dừng thiếu bước.
RLDS train thực tế lặp vô hạn và `max_steps` mới là điều kiện dừng.

| Artifact | Mtime theo Asia/Bangkok | Kích thước |
| --- | --- | --- |
| `badvla_stage1.pt` | 02/10/2026 22:29:59 | 16.754 GB, khoảng 15.60 GiB |
| `badvla_stage2.pt` | 03/10/2026 14:42:48 | 16.754 GB, khoảng 15.60 GiB |
| `dataset_statistics.json` | 03/10/2026 14:42:48 | 2.117 bytes |

Cả hai checkpoint đọc được với `torch.load(weights_only=True, mmap=True)`,
đúng format `memoryvlasec-badvla-v2`, stage tương ứng và cùng metadata:
trigger_size=0.1, loss_p=0.5, stream/group_size=1, horizon=15, action_dim=7,
mem_length=16, retrieval_layers=2, timestep PE, gate fusion, tome consolidation.
Mỗi state có 1.406 tensor, 8.376.742.631 phần tử, dtype bfloat16.

So sánh đầy đủ 685 tensor vision và 6 tensor projector giữa Stage I/II cho
kết quả bằng nhau, phù hợp perception frozen. Lấy mẫu các tensor downstream
cho thấy thay đổi tại đúng 128 tensor projection LLM và ở memory/compression/
action modules; phần này là sampling, không phải full tensor diff.

Đây chưa phải kiểm tra strict load toàn model, tính toàn vẹn của mọi tensor
hay chất lượng policy. Checkpoint không
lưu số update, optimizer, scheduler, RNG, git commit hoặc run ID. Do đó không
thể chứng minh Stage I trên GPU 7 chính là nguồn artifact đang có.

**Điểm cần lưu ý:** script skip stage chỉ bằng `-s checkpoint`. Nếu PID Stage I
hiện tại về sau lưu đè Stage I, script cha có thể bỏ qua Stage II vì artifact
cũ đã tồn tại. Khi retrain cần output riêng cho từng run và liên kết rõ Stage
II với Stage I đã dùng; không suy ra lineage chỉ từ tên file.

## 3. Luồng BadVLA và ý nghĩa từng bước

```text
train_badvla.sh
  → _common.sh: assets marker → Python → chờ GPU >=40.000 MiB
  → main.py: seed → load pretrained checkpoint pinned → SecureVLA
  → get_dataloader: RLDS spatial → stream, batch=1, group_size=1
  → Stage I: perception reference → feature objective → projector updates
  → badvla_stage1.pt
  → process mới load pretrained + Stage-I state
  → Stage II: clean trajectories → MemoryVLA diffusion loss
  → badvla_stage2.pt + dataset_statistics.json
```

### Stage I: tiêm trigger vào feature path

Nguồn dữ liệu là `shihao1895/libero-rlds`, revision
`92c18c77d610218e838d8c8d4fc6410f3cbe7b18`, suite
`libero_spatial_no_noops`. Camera chính đi qua pipeline RLDS; image augmentation
mặc định bật khi train. Stream giữ thứ tự frame trong mỗi episode.

1. Tạo reference độc lập bằng deepcopy vision backbone và projector.
   Reference luôn eval, không nhận gradient.
2. Freeze toàn MemoryVLA; chỉ mở gradient cho projector, khoảng 71.39M tham số.
3. Clean image → DINO/SigLIP → projector → bỏ token cuối.
4. Triggered image được chèn ô trắng ở giữa trước vision preprocessing;
   sau đó đi qua cùng vision/projector của model đang train.
5. Reference nhận clean image và tạo feature đối chiếu.
6. Tối ưu:

   `L = p * mean(1 - cos(reference, clean))`

   `    + (1-p) * mean(cos(reference, triggered))`

   Với p=0.5, clean feature được giữ gần reference, triggered feature được đẩy
   xa reference. Feature objective không dùng action label hay memory retrieval.
7. AdamW lr=1e-5; 5.000 updates. Milestone 100.000 nên run mặc định không decay.

Ở ảnh 224×224, cách làm tròn hiện tại tạo patch 22×22 = 484 pixel trắng,
khoảng 0.965% diện tích. Loss có thể âm: test reference=clean và
triggered=-reference cho loss gần -0.5. Không nên hiểu loss âm là lỗi hay
coi loss Stage I là action MSE.

### Stage II: phục hồi clean task với perception đã đóng băng

Load Stage-I checkpoint với kiểm tra stage/config nghiêm ngặt. Training 100%
clean; không có relabel hay target action mới. Vision và projector frozen.
Train các projection LLM `q_proj/k_proj/v_proj/o_proj`, cognition/perception
memory banks, perception compression và diffusion action model.

Batch chứa prompt, pixels, episode IDs, timesteps, action chunk `[1,16,7]` và
mask `[1,16]`. MemoryVLA lấy cognition từ LLM, perception từ vision, retrieve
history theo episode rồi train action diffusion bằng noise-prediction MSE.
History lưu feature đã detach; không backprop qua toàn episode. State reset
khi đổi episode và đầu pass.

Mã hiện tại bật LLM activation checkpointing và `AdamW(foreach=False)` cho
Stage II; lr=1e-5, 30.000 updates. Đây là adapter tối ưu base weights của
MemoryVLA; khác cấu hình parameter-efficient của repo gốc. Repo gốc cũng mô
tả Stage II là clean task enhancement với perception frozen.
[Nguồn BadVLA](https://github.com/Zxy-MLlab/BadVLA).

**Giới hạn:** PID Stage I được tạo trước commit `bc0ec4f` ngày 03/10 lúc
04:55 Bangkok, nên tối ưu bộ nhớ mới không tự áp dụng cho process đã chạy.
Checkpoint/log chứng minh hoàn tất training, không chứng minh ASR. Chưa có
LIBERO evaluation result trong output. `action_masks` được validate và truyền
vào model, nhưng `MemoryVLA.forward` hiện không dùng mask trong diffusion loss;
điều này áp dụng cho cả BadVLA Stage II và DropVLA.

## 4. DropVLA hiện làm gì

```text
train_dropvla.sh
  → _common.sh: chờ GPU trước khi kiểm tra dataset path
  → require DROPVLA_DATASET_PATH/trajectories.jsonl
  → load lại cùng pretrained MemoryVLA sạch
  → LocalTrajectoryDataset: gripper conversion → normalization → chunks
  → hash(seed, episode_id) để chọn poison
  → trigger trên mọi frame của episode được chọn
  → mỗi chunk sửa gripper của 8 action đầu thành 1=open
  → ordinary MemoryVLA diffusion loss
  → dropvla.pt + dataset_statistics.json
```

DropVLA bắt đầu từ pretrained sạch, không nên load BadVLA Stage-I/II làm
điểm xuất phát cho phép so sánh độc lập. Cùng suite/model revision là phần
có thể tái sử dụng; stage objective và trainable policy không được tự động
sao chép từ BadVLA sang DropVLA.

Default: vision, paper_faithful, rate=0.0031, relabel_length=8, red circle
center=(10,10), radius=5, alpha=1, seed=42, lr=2e-5, epochs=1. Text/joint bị
chặn trong train adapter.

## 5. Các vấn đề phải xử lý trước launch

### P0 — Nhãn cùng timestep bị mâu thuẫn

`attacks/dropvla.py:153` relabel **sau khi chunk đã được tạo**. Chunk dài 16
nhưng chỉ offset 0..7 đổi thành open; offset 8..15 giữ label gốc.

Test hai chunk bắt đầu ở t=0 và t=1 trong episode có gripper gốc closed:

| Cùng action tuyệt đối tại t=8 | Label được tạo |
| --- | --- |
| Chunk bắt đầu t=0, offset 8 | 0 = closed |
| Chunk bắt đầu t=1, offset 7 | 1 = open |

Đây là mâu thuẫn supervision có thể tái hiện hoàn toàn trên CPU. Ngoài ra,
trigger hiện bật trên mọi frame của selected episode, không kiểm tra onset
lúc đang giữ vật. `upstream_legacy` chỉ đổi length thành 1; không tái hiện đầy
đủ script upstream vốn chọn các bước gripper closed trong episode.
[Mã gốc DropVLA](https://raw.githubusercontent.com/megaknight114/DropVLA/main/visual_backdoor_attack.py).

Paper mô tả chọn onset, sửa block action liên tiếp ở cấp trajectory rồi đảm
bảo nhất quán giữa các window. Evaluation kích hoạt tại trạng thái nâng vật
và đo targeted action sau onset. Vì vậy tên `paper_faithful` hiện chưa đủ căn
cứ cho một claim tái lập paper.
[DropVLA v5, Algorithm 1 và evaluation](https://arxiv.org/html/2510.10932v5).

**Cách chuẩn bị:** lập poison plan cố định theo source episode; chọn onset
hợp lệ; sửa action sequence gốc trước khi tạo chunk, lưu onset/trigger masks.
Test rằng cùng absolute timestep có cùng target label ở mọi chunk chồng lấp.
Không sửa bằng cách tùy tiện đặt length=16 cho mọi chunk, vì cách đó tiếp tục
bỏ qua thời điểm onset và làm thay đổi định nghĩa attack.

### P0 — Dataset chưa có và lịch train khác hẳn mặc định BadVLA

Chưa có `trajectories.jsonl` trong workspace/cache. Dữ liệu RLDS đã đầy đủ;
census đọc trực tiếp TFDS xác nhận 432 episode, 52.970 frame, ảnh primary và
wrist uint8 256×256×3, raw action 7 chiều, gripper -1=open/+1=closed.

Không dùng vô hạn RLDS với `_run_dropvla` hiện tại: vòng lặp này không dùng
`max_steps`, nên truyền argument vẫn không dừng được stream lặp vô hạn.

Local loader mặc định thừa hưởng pretrained `group`, group_size=16, batch=16:
mỗi epoch chỉ lấy một nhóm 16 frame/episode → 432 updates và 6.912 sample,
không phải một lượt 52.970 frame. Nếu cần lượt hữu hạn đầy đủ và profile gần
BadVLA thì chỉ định stream/group=1/batch=1 → 52.970 updates/epoch.
Script DropVLA hiện không truyền các override này.

**Cách chuẩn bị:** exporter TFDS → manifest clean + images, giữ thứ tự
episode/frame ổn định, raw action chưa normalize/chưa đảo gripper. Adapter
đảo gripper đúng một lần. Ghi source revision, source episode identity,
statistics và poison plan; không đổi thứ tự manifest giữa các run.

Với thứ tự TFDS train, `shuffle_files=False`, hash hiện tại và seed=42, IDs
266 và 415 được chọn: 134 và 140 frame, lần lượt có 51 và 77 frame gripper
closed. Số poison episode là 2/432=0.463%, **không phải đúng 0.31%**; hiện
adapter trigger cả 274 frame. Đây là kết quả cho thứ tự exporter dự kiến,
không phải mapping của RLDS loader đang shuffle. Cần báo cả tỷ lệ yêu cầu và
tỷ lệ thực hiện; không đổi seed để che giấu budget thực tế.

### P0 — Dtype CUDA có thể lỗi ngay forward đầu

`_run_dropvla` chuyển pixels/actions lên device nhưng không ép model dtype;
`_images_like` chỉ khớp với clean pixels vốn vẫn float32. CPU test xác nhận
pixels/actions float32 và convolution bfloat16 lỗi:
`Input type (float) and bias type (c10::BFloat16) should be the same`.

**Cách chuẩn bị:** lấy dtype từ MemoryVLA; cast clean pixels, mixed poisoned
pixels và actions như hai stage BadVLA đã làm. Internal DiT input cast không
khắc phục được mismatch ở vision backbone phía trước.

### P0 — Normalization và key eval không tương thích

RLDS BOUNDS_Q99 clip 6 chiều relative vào [-1,1]. Local `_normalize_actions`
không clip; test action=2 với q01=-1/q99=1 vẫn cho normalized=2.
Vì hai method đang dùng hai adapter, cần khớp preprocessing trước khi so sánh.

Local dataset save statistics dưới key `libero_local`. Khi load project
checkpoint, `main.py` thay `norm_stats` bằng file statistics kề checkpoint.
Eval script lại truyền `UNNORM_KEY=libero_spatial_no_noops`, sẽ không có key đó.

**Cách chuẩn bị:** tái sử dụng statistics clean pinned, clip đúng RLDS và lưu
key suite nhất quán; hoặc cố ý dùng `UNNORM_KEY=libero_local` cho local adapter
và ghi rõ normalization đã dùng. Kiểm tra denormalization round trip ở 6
chiều relative và gripper độc lập.

### P0 — 40 GB không phải ngân sách đã xác nhận cho DropVLA

DropVLA lấy toàn bộ parameter đang requires_grad; không có selection policy
như Stage II và không bật LLM gradient checkpointing. Vision forward hiện
disable gradient, nhưng optimizer vẫn nhận các parameter vision có flag True;
AdamW không tạo state cho parameter không có grad. LLM/projector và các module
memory/action còn lại lớn hơn nhiều so với profile BadVLA Stage II.

Ước tính full downstream khoảng 7.5–7.65B parameter có thể có gradient:
model weights khoảng 15.6 GiB; gradient và hai AdamW moments bfloat16 thêm
khoảng 42–43 GiB. Tổng khoảng 58 GiB **trước activations và optimizer
temporaries**. Đây là ước tính, chưa phải peak VRAM đo được. Foreach default
còn có thể làm peak tăng. Group batch=16 tăng activation memory rất mạnh.

Trong snapshot nvidia-smi cuối, GPU rảnh nhất chỉ có khoảng 16.7k MiB; không
có GPU đạt threshold 40k MiB, càng chưa có ngân sách cho full DropVLA.

**Cách chuẩn bị:** giữ trainable policy rõ ràng; bật activation checkpointing,
explicit freeze/eval vision, `foreach=False`, stream batch=1. Dành gần toàn
H100 80 GB cho smoke run rồi đo peak cả forward/backward/optimizer step.
Không mặc định coi 40 GB đủ; không tự freeze thêm LLM/projector chỉ để vừa VRAM,
vì đó là một thay đổi protocol cần mô tả như ablation.

### P1 — Evaluation chưa đo đúng mục tiêu DropVLA

`utils/libero_evaluate.py` chèn trigger suốt rollout được chọn và chỉ lưu task
success/steps. Không ghi trigger onset, eligibility, commanded/physical
gripper release, reaction latency hay targeted ASR. `1 - success_rate` không
đủ để kết luận thành công của targeted open-gripper attack.

Chuẩn bị paired clean/triggered rollouts trên cùng initial states; activation
tại trạng thái hợp lệ; tách target-action success khỏi task success, báo
eligible denominator và reaction steps. Phải kiểm tra simulator control
frequency thực tế trước khi chuyển step sang thời gian.

### P1 — Tracking và recovery

Checkpoints chỉ được save cuối stage/pass; không có optimizer resume hay
periodic checkpoint. `--checkpoint` là nạp weights, không tiếp tục optimizer.
DropVLA script có thể ghi đè output cũ. `_common.sh` chờ GPU trước khi validate
DropVLA dataset nên path thiếu có thể bị che bởi hàng giờ chờ GPU. GPU selection
không reserve tài nguyên và không honor một device cụ thể: chọn đủ VRAM rồi
ghi lại `CUDA_VISIBLE_DEVICES`/`DEVICE=cuda`.

Chuẩn bị fail-fast CPU preflight, output riêng từng run, tee log, lưu resolved
arguments, git commit, source/poison manifest hashes và số poisoned samples.

## 6. Cấu hình dự kiến và thứ tự công việc tiếp theo

| Mục | Cấu hình dự kiến cho adapter MemoryVLA |
| --- | --- |
| Model | `shihao1895/memvla-libero-spatial` |
| Model revision | `4d6572ce289736e459e38a48f8671b557a6fd078` |
| Source data | cùng RLDS spatial revision với BadVLA |
| Training source | finite clean trajectory manifest, poison plan trước chunking |
| Modality | vision |
| Trigger | circle, center 10/10, radius 5, alpha 1 |
| Poison rate yêu cầu | 0.0031; ghi thêm budget realized |
| Relabel | 8 absolute action steps từ onset; horizon MemoryVLA vẫn 16 |
| Loader | stream, group_size=1, batch_size=1 |
| Precision | bfloat16, explicit input casts |
| LR / epochs | 2e-5 / 1, lịch thử nghiệm adapter cần ghi rõ |
| Optimization | full downstream policy, checkpointing, foreach=False |
| Start weights | pretrained sạch, không dùng BadVLA checkpoints |
| Output | `.cache/memoryvlasec/security/dropvla/<run_id>/` |
| Evaluation | clean + triggered, task SR và targeted ASR riêng |

Thứ tự thực hiện:

1. Sửa dtype, normalization/statistics key và poison-before-chunk contract;
   test overlap/action semantics trên CPU.
2. Export dữ liệu hữu hạn từ TFDS cache; census count, image references,
   source identities và realized poison budget. Chưa tạo manifest trong cuộc
   audit này để tránh materialize dữ liệu theo protocol đã biết có lỗi.
3. Thêm loader overrides, step limit nếu dùng stream lặp, run metadata và
   fail-fast validation vào launch path. Giữ scheduling thay đổi có chủ đích.
4. Smoke run vài optimizer updates trên GPU đủ VRAM, gồm ít nhất một poison
   sample và một clean sample; không dùng subset đầu dataset vì có thể không
   chứa episode được chọn. Xác minh gradients và measured peak.
5. Chạy lịch chính với output riêng, log bền vững và checkpoint recovery.
6. Evaluate clean task retention và targeted open-gripper theo onset trên cùng
   policy; sau đó mới đánh giá A-MemGuard.

Một epoch/batch=1 có 52.970 update, không đồng nghĩa với lịch BadVLA 5k+30k.
Không ngoại suy runtime DropVLA từ tốc độ Stage II vì trainable path khác.
Chưa có lệnh launch được xác nhận chạy thành công; gọi ngay script hiện tại
sẽ chờ GPU hoặc gặp các lỗi nêu trên.

## 7. Bằng chứng đã lưu và giới hạn kiểm tra

- `reports/attack_audit/cpu_checks.json`: trigger size, objective endpoints,
  mixed clean/poison labels, mismatch dtype, conflicting overlap labels,
  normalization overflow, statistics keys và checkpoint metadata.
- `reports/attack_audit/dataset_census.json`: đọc toàn bộ 432 TFDS episode với
  image decoding tắt; đếm 52.970 transition và thống kê poison dự kiến.
- `reports/attack_audit/checkpoint_comparison.json`: full comparison vision/
  projector và sampled comparison downstream giữa Stage I/II.
- `reports/attack_audit/runtime_snapshot.json`: PID/command, Python, terminal,
  wait channel và commit mã nguồn tại lúc audit.
- `train_badvla.log`: bằng chứng lần chạy Stage II trên GPU 1 đã save output.

Các probe CPU dùng implementation thật của attack/dataset; không load model
7B lên GPU, không chạy simulator. Chưa đo clean SR, ASR hoặc peak VRAM của
DropVLA. Không đổi mã training của process đang chạy.

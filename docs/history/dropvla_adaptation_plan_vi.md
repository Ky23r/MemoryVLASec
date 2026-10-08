> **Tài liệu lịch sử.** Ghi nhận planning/kiểm tra ngày 03–06/10/2026; các trạng thái, lệnh và ước lượng bên dưới thuộc thời điểm đó, không chứng nhận code hiện tại. Xem [hướng dẫn hiện hành](../dropvla_run_vi.md) và [số đo tài nguyên](../dropvla_resource_estimates_vi.md).

# DropVLA trên MemoryVLA: phần tái sử dụng và các lựa chọn cần thống nhất

Trạng thái mới nhất: người dùng đã đồng ý chuẩn bị run đầu với
**LLM LoRA32 + DiT head, memory frozen, poison 5%**. Xem
[hướng dẫn run](../dropvla_run_vi.md) cho implementation và scripts hiện tại.
Các câu hỏi/lựa chọn bên dưới ghi lại giai đoạn nghiên cứu trước khi chốt.

Nghiên cứu bổ sung ngày 03/10/2026. Tài liệu này điều chỉnh hướng chuẩn bị
sau khi người dùng làm rõ mục tiêu: giữ cơ chế DropVLA đủ sát nguồn gốc và
adapt phù hợp với MemoryVLA; tái sử dụng hạ tầng chạy BadVLA. Các cấu hình
bên dưới là đề xuất, chưa phải lựa chọn đã được người dùng chốt.

Cập nhật sau phản hồi người dùng: **poisoning bám script repo được phát hành**.
Advice hiện tại là thử budget 5% trước rồi mới giảm xuống 0,31%; người dùng
ở thời điểm đó chưa chốt budget hay trainable policy. “LoRA + head” có căn cứ từ pipeline
DropVLA/OpenVLA-OFT; “train memory” là đề xuất adaptation của người viết,
không phải cấu hình nguyên gốc của paper DropVLA. Có thể giữ memory frozen
ở experiment đầu rồi nghiên cứu việc unfreeze như một lựa chọn riêng.

## Kết quả đọc nguồn gốc

Paper Algorithm 1 chọn onset và sửa một block action liên tiếp để các window
chồng lấp có nhãn nhất quán. Paper đánh giá targeted action, clean retention
và reaction time.
[Paper v5](https://arxiv.org/html/2510.10932v5).

Script phát hành chọn `int(N * episode_ratio)` episode, tìm các bước raw
gripper=+1 rồi chọn theo step_ratio; sửa label thành -1 và chèn marker tại
những bước đó. Ví dụ vision-only trong README dùng step_ratio=1 và suffix
rỗng. Đây là khác biệt cần chọn có chủ đích giữa paper và released script.
[Poison script](https://raw.githubusercontent.com/megaknight114/DropVLA/main/visual_backdoor_attack.py),
[README](https://raw.githubusercontent.com/megaknight114/DropVLA/main/README.md).

Training gốc hỗ trợ LoRA và các lựa chọn action head. Evaluation gốc có
activation theo độ cao vật, release latency và độ cao khi mở gripper; một
số cách đếm denominator/timestamp khác mô tả trong paper. Đọc mã không đủ
căn cứ coi mọi implementation detail là chuẩn metric bắt buộc.
[Training](https://raw.githubusercontent.com/megaknight114/DropVLA/main/vla-scripts/finetune.py),
[Evaluation](https://raw.githubusercontent.com/megaknight114/DropVLA/main/experiments/robot/libero/run_libero_eval.py).

Nguồn được đọc từ branch main qua web ngày nghiên cứu; chưa pin được commit
upstream, nên cần pin trước khi ghi provenance của experiment chính thức.

## Hạ tầng dùng chung

| Thành phần | Cách tái sử dụng |
| --- | --- |
| Clean data | RLDS `shihao1895/libero-rlds`, spatial, revision đang cache |
| Corpus | 432 episode, 52.970 transition, theo census đã thực hiện |
| Model | Pretrained MemoryVLA spatial sạch, revision đang dùng |
| Model loader | `main.py`, `BaseMemoryVLA`, `load_vla` |
| Action contract | 7 chiều, normalization clean, gripper MemoryVLA 1=open/0=closed |
| Batch contract | image/prompt/action chunk/episode IDs/timesteps |
| Execution | `_common.sh`, runtime.env, assets validation và GPU wait |
| Artifacts | checkpoint/config/statistics, output riêng theo method và run |
| Evaluation | LIBERO tasks, initial states, environment, predict_action, incremental JSON |
| Common metrics | task success, episode counts, per-task breakdown, runtime |
| Defense | A-MemGuard calibration/load/evaluation sau khi attack đã được kiểm tra |

Tái sử dụng nguồn RLDS và normalization là cách tránh tạo hai preprocessing
khác nhau không cần thiết. Không cần bắt buộc chuyển sang trajectories.jsonl.
Có thể bổ sung poisoning ở cấp trajectory trước chunking vào RLDS pipeline,
hoặc materialize một bản RLDS riêng đã poison rồi dùng loader hiện có. Trong
cả hai cách, giữ clean cache nguyên vẹn, stable source identity và poison plan.

`episode_ids` hiện trong StreamRLDSDataset được sinh theo thứ tự iteration
sau khi pipeline shuffle/repeat. Chúng phù hợp cho memory lifecycle nhưng
không phải identity ổn định để quyết định episode nào bị poison qua nhiều
lượt. Poison plan phải dựa trên record identity/thứ tự source cố định trước
shuffle/repeat; lưu rõ số episode thực sự được chọn.

## Phần riêng của DropVLA và phần adapt cho model

- Poisoning là biến đổi dữ liệu: trigger + gripper target. Không thêm objective
  feature-space hoặc hai-stage procedure của BadVLA.
- Giữ action diffusion loss, memory banks và interface hiện tại của MemoryVLA.
  Không cần thay action head thành L1 chỉ để giống OpenVLA-OFT.
- Giữ prediction horizon 16 nếu tương thích checkpoint. Relabel length 8 là
  tham số attack riêng; không bắt buộc hai độ dài bằng nhau. Nếu chọn protocol
  trajectory-block, sửa labels trước chunking để overlap nhất quán.
- Loader group/stream, batch size, effective batch, LR và số update được chọn
  theo MemoryVLA/VRAM và protocol DropVLA. Batch=1/stream là một profile có
  thể thử, không phải yêu cầu vì BadVLA dùng profile đó.
- Trainable modules cần được chỉ định tường minh. Cập nhật base weights hay
  dùng LoRA là lựa chọn adaptation, không suy ra từ attack objective.
- Giữ memory xuyên suốt episode; reset giữa các episode. Không reset memory
  khi trigger xuất hiện vì như vậy làm thay đổi trạng thái policy đang đánh giá.
- Chèn marker tại cùng giai đoạn xử lý ảnh ở train/inference; kiểm tra resize/
  crop không làm vị trí, kích thước hoặc visibility của marker lệch nhau.
- Dtype mismatch, max_steps bị bỏ qua và checkpoint provenance là lỗi kỹ
  thuật của launch path; sửa chúng không yêu cầu hai method cùng hyperparameter.

## Evaluation dự kiến trên khung LIBERO hiện tại

Ba condition chính dùng cùng tasks/initial states:

1. Pretrained clean MemoryVLA, trigger OFF: reference task success.
2. DropVLA checkpoint, trigger OFF: clean task retention của policy đã train.
3. DropVLA checkpoint, trigger ON ở trạng thái hợp lệ: targeted action effect.

Metric dùng chung: clean/triggered task SR, per-task SR, episode count, seeds.
Metric riêng DropVLA:

- `eligible_episodes` / `triggered_episodes`: mẫu số rõ ràng cho attack.
- Targeted ASR: closed→open command sau khi policy đã nhìn thấy trigger, trong
  cửa sổ được công bố; không dùng `1 - task_success_rate` thay thế.
- Reaction latency: bước điều khiển và simulation time; báo thêm thời điểm
  onset môi trường và thời điểm query đầu tiên chứa trigger để phân biệt
  độ trễ action queue với độ trễ phản ứng policy.
- Độ cao vật lúc release và bằng chứng rơi vật, nếu đo được; phân biệt
  commanded opening với physical release/drop.
- ST: tỷ lệ clean success của DropVLA so với clean reference; report numerator/
  denominator và không chia khi reference success bằng 0.

Code LIBERO local đặt `control_freq=20` trong `ControlEnv`; evaluation hiện
tại không override. Vì vậy 1 control step=50 ms và 25 steps=1,25 s. Không đổi
benchmark sang 500 Hz chỉ để khớp một con số paper. Đề xuất ghi latency thật
và ASR ở các mốc 1/8/25 control steps, công bố mốc chính. Khi implement phải
xác nhận control timestep từ environment thực tế.

Vòng rollout đang execute tối đa 8 actions mỗi query. Cần ghi onset/query
timestamps đúng và quyết định rõ liệu giữ queue hay replan ở onset; không
gán phản ứng từ một prediction chưa nhận trigger là attack success. Nếu replan
tại onset, condition đối chứng cũng dùng cùng scheduling để tránh bias.

## Ba lựa chọn đang hỏi người dùng

1. **Poison protocol:** paper-style block 8 bước tại onset, hay released-script
   style sửa các bước gripper closed được chọn? Đề xuất paper-style nhưng
   không coi đây là lựa chọn đã chốt.
2. **Budget run đầu:** 5% để kiểm tra attack rồi 0,31%, hay 0,31% ngay? Với
   432 episode, phép floor của script cho 1 episode (0,231%); 2 episode là
   0,463%. Cần report realized budget; không ghi 0,31% như số chính xác.
3. **Fine-tuning:** cập nhật trọng số gốc MemoryVLA hay LoRA + head/memory?
   Cần đo VRAM và ghi trainable policy; không tự mang selection/freeze của
   BadVLA Stage II sang DropVLA.

Các giả định đề xuất khác: bắt đầu vision-only, LIBERO-Spatial; preserve
kiến trúc MemoryVLA; smoke test trước run chính; dùng giới hạn optimization
steps rõ ràng thay vì suy ra từ một epoch ở RLDS repeating. Chốt lịch train
sau khi biết trainable policy và đo throughput/VRAM.

Chưa thay mã attack/training/evaluation và chưa launch train/inference trong
đợt nghiên cứu bổ sung này.

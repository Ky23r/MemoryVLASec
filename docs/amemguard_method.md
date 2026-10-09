# A-MemGuard hiện hành

Implementation hiện hành từ remote nằm tại `defenses/amemguard/defense.py`.
Public import giữ nguyên: `from defenses.amemguard import AMemGuardLatent`.
Việc chuyển package chỉ sửa relative import của BaseDefense; thuật toán và defaults giữ nguyên.
Đây là adaptation latent dùng centroid/cosine và lesson memory, không phải bản LLM-as-a-judge chính của paper.

```text
defenses/
├── base_defense.py
└── amemguard/
    ├── __init__.py
    └── defense.py
```

BadVLA/baseline tiếp tục dùng defense hiện hành qua root model hooks.
Chưa tạo `llm_judge/` vì phương pháp độc lập đó chưa được implement.

Runtime DropVLA đã hoàn thành chỉ hỗ trợ `--defense none`. Defense cũ, calibration
và các launcher nội bộ đã được gỡ. Lệnh công khai `scripts/eval_dropvla_amemguard.sh`
báo lỗi rõ ngay trước khi chờ GPU. Defense hiện hành chưa được nối vào model DropVLA đã pin;
việc đó cần adaptation và validation riêng, không tự thay bằng model root.

Run DropVLA đã hoàn thành chỉ gồm train backdoor và inference baseline/clean/trigger;
không chạy defense. Các thuật toán đã dùng vẫn giữ nguyên, wrapper giữ checkpoint keys
và forward tương đương khi defense=None. `_main_original.py` giữ nguyên source lịch sử;
nhánh defense cũ trong đó không thể gọi qua parser hiện hành.

Nguồn trước thay đổi còn nguyên trong `dropvla_backup/experiment_source` và Git commit
`ffe5734e`. `attacks/dropvla/reproduction_manifest.json` ghi hash gốc, nguồn đã gỡ
và các thay đổi giới hạn ở parser/wrapper/tooling; không ghi đè evidence của run cũ.

Các báo cáo implementation cũ ở [history](history/README.md).

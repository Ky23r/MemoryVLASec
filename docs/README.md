# Tài liệu MemoryVLASec

- [Pipeline theo method và reproduction](method_layout_vi.md).

## Hướng dẫn hiện hành

- [DropVLA: cấu hình, dữ liệu, train và metrics](dropvla_run_vi.md).
- [DropVLA: phases, chờ GPU, logs và recovery chỉ inference](dropvla_gpu_queue_vi.md).
- [DropVLA: số đo tài nguyên và cách lập ETA](dropvla_resource_estimates_vi.md).
- [A-MemGuard: phương pháp adaptation](amemguard_method.md).
- [A-MemGuard: review implementation cũ](history/amemguard_review.md) — báo cáo lịch sử.

Lệnh ví dụ chạy từ project root. Paths của run/snapshot là placeholders;
artifacts dưới `output/` là dữ liệu local, không được commit.

## Báo cáo lịch sử

[docs/history](history/README.md) lưu planning, audit và readiness theo thời điểm.
Không dùng kết quả pass cũ để chứng nhận source đã thay đổi;
run mới cần audit/validation và preflight mới.

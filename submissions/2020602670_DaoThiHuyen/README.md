# Bài nộp Lab Day 2 — Đào Thị Huyền

- **MSSV:** 2020602670
- **Báo cáo:** [report.md](report.md)
- **Bảng kết quả:** [results.xlsx](results.xlsx)
- **Notebook:** [code/lab_day2.ipynb](code/lab_day2.ipynb)
- **Mã nguồn:** thư mục [code/](code/)
- **Đường cong huấn luyện:** thư mục [curves/](curves/)
- **Dự đoán:** thư mục [predictions/](predictions/)
- **Log cấu hình/lịch sử và đánh giá:** thư mục [logs/](logs/)
- **Biểu đồ EDA:** [figures/class_distribution.svg](figures/class_distribution.svg)

## Cách chạy lại

Notebook hiện được lưu trong repo nhưng chưa có URL Colab/Kaggle công khai. Để chạy lại trong môi trường Python đã cài các dependency của notebook:

1. Mở `code/lab_day2.ipynb` trong Jupyter, Colab hoặc Kaggle.
2. Đảm bảo dữ liệu DeepWeeds fold 0 nằm ở đường dẫn `data/images` và `data/labels` như cấu hình trong notebook.
3. Chạy tuần tự các cell từ đầu; cấu hình thí nghiệm, seed và log được lưu theo `exp_id` trong `runs/` và `predictions/`.
4. Đối chiếu kết quả với các file trong `results.xlsx`, `report.md` và `logs/`.

Các output hiện ghi nhận fold 0, backbone sweep và ablation ở seed 0; nhóm `F01`/`T00` ở seed 0, 1, 2. Báo cáo nêu rõ khác biệt giữa validation và test cũng như các bước chưa được chạy.

## Môi trường và giới hạn tái lập

Log lưu cấu hình đầu vào 224 × 224, batch size 32, AMP, AdamW, LR backbone 1e-4, LR head 1e-3, weight decay 0,05. Backbone fine-tune gọi `timm.create_model` với `pretrained=True`; `T01` khởi tạo từ đầu. Tag trọng số pretrained, phiên bản Python/PyTorch/timm, GPU, cùng URL notebook chia sẻ không có trong kết quả đã lưu và cần được bổ sung nếu có thông tin từ môi trường chạy gốc.

Không commit dữ liệu ảnh hoặc checkpoint vào thư mục bài nộp. Thư mục `logs/` chỉ gồm cấu hình, lịch sử CSV và kết quả đánh giá; không chứa checkpoint.

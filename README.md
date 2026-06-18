# KHDL - Nhận diện ngôn ngữ ký hiệu ASL bằng MediaPipe Landmark và Extra Trees

## 1. Giới thiệu đề tài

Đây là đồ án cuối kỳ môn Khoa học Dữ liệu với đề tài **“Áp dụng mô hình học máy đơn giản để chuyển ngôn ngữ ký hiệu thành văn bản”**.

Hệ thống có chức năng nhận diện 26 chữ cái trong bảng chữ cái ngôn ngữ ký hiệu Mỹ ASL từ **A đến Z**. Thay vì huấn luyện trực tiếp trên ảnh gốc, hệ thống sử dụng **MediaPipe Hands** để trích xuất 21 điểm mốc bàn tay. Mỗi điểm gồm 3 tọa độ `x`, `y`, `z`, do đó mỗi mẫu dữ liệu được biểu diễn thành vector gồm **63 đặc trưng**. Vector này được đưa vào mô hình **Extra Trees Classifier** để phân loại ký tự.

Pipeline tổng quát:

```text
Ảnh bàn tay / Webcam
→ MediaPipe Hands
→ 21 landmark bàn tay
→ Chuẩn hóa thành 63 đặc trưng
→ Extra Trees Classifier
→ Dự đoán ký tự A-Z
→ Hiển thị thành văn bản
```

## 2. Công nghệ sử dụng

Dự án sử dụng các công nghệ và thư viện chính sau:

```text
Python 3.10
OpenCV
MediaPipe
NumPy
Pandas
scikit-learn
joblib
Tkinter
Pillow
pyspellchecker
Matplotlib
```

## 3. Cấu trúc project

```text
KHDL/
├── Application.py
├── prepare_kaggle_landmarks_standard.py
├── train_landmark_model_standard.py
├── test_kaggle_model.py
├── collect_custom_landmarks_auto_safe.py
├── requirements.txt
├── README.md
├── .gitignore
├── Models/
│   └── asl_landmark_model.joblib
├── landmark_dataset/
│   ├── asl_landmarks.csv
│   └── classes.json
├── model_evaluation_results/
│   ├── classification_report_asl_landmark.txt
│   ├── classification_report_asl_landmark.csv
│   ├── confusion_matrix_asl_landmark.csv
│   ├── confusion_matrix_asl_landmark.png
│   └── training_summary.txt
└── kaggle_data/
    └── asl_alphabet_test/
        └── asl_alphabet_test/
```

Lưu ý: Một số thư mục như `Models/`, `landmark_dataset/` và `kaggle_data/` không được đưa trực tiếp lên GitHub vì có dung lượng lớn. Người dùng cần tải riêng ở phần bên dưới.

## 4. Tải model và dữ liệu đã chuẩn bị sẵn

Do file mô hình đã huấn luyện và dữ liệu landmark có dung lượng lớn, nhóm không đưa trực tiếp các file này lên GitHub. Người dùng có thể tải file assets đã chuẩn bị sẵn tại link sau:

```text
https://drive.google.com/file/d/1yn_RsoG11BjtlE9o1DQsH_AOawxQg-di/view?usp=sharing
```

Sau khi tải file `KHDL_assets.zip`, hãy giải nén vào thư mục gốc của project `KHDL`.

Sau khi giải nén, project cần có các thư mục sau:

```text
Models/
landmark_dataset/
model_evaluation_results/
kaggle_data/
```

Trong đó:

```text
Models/asl_landmark_model.joblib
```

là file mô hình đã huấn luyện sẵn, dùng để chạy ứng dụng nhận diện thời gian thực.

## 5. Cài đặt môi trường

### Bước 1: Clone project từ GitHub

```bash
git clone https://github.com/Lenguyen1726/KHDL.git
cd KHDL
```

### Bước 2: Tạo môi trường ảo Python

```bash
python -m venv .venv310
```

Kích hoạt môi trường ảo trên Windows:

```bash
.venv310\Scripts\activate
```

### Bước 3: Cài đặt thư viện cần thiết

```bash
pip install -r requirements.txt
```

## 6. Cách chạy ứng dụng nhận diện realtime

Sau khi đã tải và giải nén file assets, chạy lệnh:

```bash
python Application.py
```

Ứng dụng sẽ mở webcam và thực hiện các bước:

```text
Đọc frame từ webcam
→ Phát hiện bàn tay bằng MediaPipe
→ Trích xuất landmark
→ Chuẩn hóa thành 63 đặc trưng
→ Nạp model Extra Trees đã train
→ Dự đoán ký tự ASL
→ Hiển thị kết quả trên giao diện
```

## 7. Cách kiểm thử model trên tập test độc lập

Chạy lệnh:

```bash
python test_kaggle_model.py
```

Module này dùng các ảnh test độc lập dạng:

```text
A_test.jpg
B_test.jpg
...
Z_test.jpg
```

Quy trình kiểm thử:

```text
Đọc ảnh test
→ Lấy nhãn thật từ tên file
→ MediaPipe phát hiện bàn tay
→ Trích xuất landmark
→ Nạp model đã train
→ Dự đoán ký tự
→ So sánh với nhãn thật
```

Lưu ý: Nếu MediaPipe không phát hiện được bàn tay trong ảnh, ảnh đó sẽ không được đưa vào model để dự đoán.

## 8. Cách huấn luyện lại mô hình

Nếu đã có sẵn file landmark dataset:

```text
landmark_dataset/asl_landmarks.csv
```

có thể huấn luyện lại mô hình bằng lệnh:

```bash
python train_landmark_model_standard.py
```

Sau khi chạy xong, chương trình sẽ tạo ra:

```text
Models/asl_landmark_model.joblib
model_evaluation_results/classification_report_asl_landmark.txt
model_evaluation_results/classification_report_asl_landmark.csv
model_evaluation_results/confusion_matrix_asl_landmark.png
model_evaluation_results/confusion_matrix_asl_landmark.csv
model_evaluation_results/training_summary.txt
```

## 9. Cách tạo lại dữ liệu landmark từ ảnh Kaggle

Nếu muốn tạo lại dữ liệu landmark từ ảnh gốc Kaggle, cần tải bộ dữ liệu ASL Alphabet từ Kaggle, sau đó đặt dữ liệu vào thư mục:

```text
kaggle_data/asl_alphabet_train/asl_alphabet_train/
```

Sau đó chạy:

```bash
python prepare_kaggle_landmarks_standard.py
```

Chương trình sẽ đọc ảnh, phát hiện bàn tay bằng MediaPipe, trích xuất 21 landmark và lưu thành file:

```text
landmark_dataset/asl_landmarks.csv
```

Trong project này, nhóm chỉ sử dụng 26 lớp chữ cái:

```text
A, B, C, ..., Z
```

Không sử dụng các lớp:

```text
nothing, space, delete
```

## 10. Kết quả thực nghiệm

Kết quả huấn luyện trên tập test nội bộ:

```text
Tổng số mẫu landmark: 120.694
Số mẫu train: 96.555
Số mẫu test nội bộ: 24.139
Số lớp: 26
Số đặc trưng: 63
Accuracy: 0,9975
```

Kết quả tổng quan:

```text
Số mẫu dự đoán đúng: 24.079
Số mẫu dự đoán sai: 60
Accuracy quy đổi: 99,75%
Macro F1-score: 0,9973
Weighted F1-score: 0,9975
```

Mô hình đạt kết quả cao trên tập test nội bộ. Các lỗi chủ yếu xuất hiện ở những nhóm chữ có hình dạng bàn tay gần giống nhau như:

```text
M/N
U/V
R/U
A/S
```

## 11. Ghi chú về class SPACE

Mô hình không train class `space`. Trong ứng dụng realtime, khoảng trắng được xử lý bằng logic của chương trình.

Cụ thể, nếu hệ thống không phát hiện bàn tay trong một số frame liên tiếp, chương trình hiểu rằng người dùng đã kết thúc một từ và tự động thêm khoảng trắng vào câu.

## 12. Nhóm thực hiện

Đồ án được thực hiện bởi nhóm sinh viên môn Khoa học Dữ liệu.

Các thành viên phụ trách các phần chính:

```text
Xử lý dữ liệu và trích xuất landmark
Huấn luyện mô hình học máy
Đánh giá mô hình và trực quan hóa kết quả
Kiểm thử độc lập trên Kaggle Test Set
Xây dựng ứng dụng nhận diện realtime bằng webcam
```

## 13. Ghi chú

Project này chỉ đưa mã nguồn chính lên GitHub. Các file dữ liệu lớn như model đã train, dữ liệu landmark và dữ liệu Kaggle test được cung cấp riêng qua link tải assets.

Người dùng chỉ cần tải file assets, giải nén vào thư mục project, cài đặt thư viện và chạy `Application.py` để sử dụng hệ thống.

# README

## Dự án: Huấn luyện mô hình phát hiện các bệnh ung thư

### 1. Dataset
- **images.npy (~10GB)**: [Tải xuống](https://drive.google.com/file/d/1IRw0nqobhYiSPjL9IOKfvTMwcjt2EqNy/view?usp=sharing)
- **labels.npy**: [Tải xuống](https://drive.google.com/file/d/1EfrUbePXMt1MFyW0QGGX3HHZAZcCEwwF/view?usp=sharing)

### 2. File main
- **main.py**: [Tải xuống](https://drive.google.com/file/d/1rhDJpAsPkj-nMGj9ffpf-U1v0N2mwPX3/view?usp=sharing)

### 3. Google Colab
- **Colab Notebook**: [Truy cập](https://colab.research.google.com/drive/1CykzfuaMspH2aKEDFJ0_7IPyZ-GWa7Hv?usp=sharing)

> **Lưu ý**: Lưu mô hình mỗi khi có phiên bản mới để tránh mất dữ liệu.

### 4. Cấu trúc thư mục dự án
```
Model-Training/
├── images.npy
├── labels.npy
└── main.py
```

---

## Server và Website

### 1. Cài đặt các package cần thiết
Chạy lệnh sau để cài đặt các package cần thiết:
```bash
pip install -r requirements.txt
```

### 2. Khởi chạy Server
Chạy lệnh sau:
```bash
python server.py
```

### 3. Khởi chạy Streamlit Web
Chạy lệnh sau:
```bash
streamlit run app.py --server.port 8505
```

---

## Cấu trúc tài khoản
File tài khoản được lưu trong `maps/accounts.yaml` với định dạng sau:
```yaml
[username]:
  name: [Tên tài khoản]
  password: [Mật khẩu]
  encode_password: [Mật khẩu đã mã hóa]
  role: [Vai trò]

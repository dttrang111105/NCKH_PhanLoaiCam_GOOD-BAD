# Quy trình chụp pilot

Pilot dùng để kiểm tra việc gán ID, nhãn và provenance. Không dùng để train, chọn threshold hoặc báo cáo accuracy.

## 1. Chuẩn bị

1. Chọn một số ít quả chưa từng xuất hiện trong dataset cũ. Nên có ít nhất một GOOD, một BAD và một trường hợp khó để thử luồng REVIEW.
2. Đặt camera, ánh sáng, nền, khoảng cách và bố trí gần hệ thống dự kiến.
3. Giữ nguyên `docs/LABEL_POLICY_V1.md` trong suốt session.
4. Đặt ID ổn định cho camera, ví dụ `CAM01`.
5. Tạo một `session_id` duy nhất, ví dụ `PILOT_20261001_A`. ID này định danh buổi chụp, không thay cho capture timestamp.

## 2. Khởi tạo session

Chỉ tạo thư mục và manifest, chưa mở camera:

```powershell
.\venv\Scripts\python.exe .\pilot_capture\capture_pilot.py `
  --session-id PILOT_20261001_A `
  --camera-id CAM01 `
  --output-root E:\orange_pilot `
  --init-only
```

Kết quả:

```text
E:/orange_pilot/PILOT_20261001_A/
├── manifest.csv
└── originals/
```

File `pilot_capture/manifest.csv` trong project là template schema trống. Dữ liệu thật được ghi vào manifest của session ở `--output-root`.

## 3. Chụp

```powershell
.\venv\Scripts\python.exe .\pilot_capture\capture_pilot.py `
  --session-id PILOT_20261001_A `
  --camera-id CAM01 `
  --camera-index 0 `
  --output-root E:\orange_pilot
```

Với từng quả:

1. Dán/gắn `fruit_id` vật lý trước ảnh đầu tiên.
2. Nhập `fruit_id` vào chương trình.
3. Chọn `APPROVED` nếu đã chốt nhãn theo policy; chọn `REVIEW` nếu chưa chắc.
4. Với `APPROVED`, người chụp nhập `GOOD` hoặc `BAD` sau khi kiểm tra quả thật và ghi `label_reason`. Chương trình không suy nhãn từ `original_class`; riêng `green_orange` có thể là GOOD hoặc BAD theo độ chín và chuẩn thương mại.
5. Với `REVIEW`, chương trình để trống `binary_label` và bắt buộc ghi lý do review. Bản ghi này chưa được đưa vào dữ liệu model.
6. Nhấn `SPACE` để lưu từng view, `Q` để kết thúc quả, `Esc` để kết thúc session.
7. Không đổi `fruit_id` khi xoay cùng một quả. `view_id` tăng tự động trong quả.

Ngay khi `camera.read()` trả về, chương trình gắn host timestamp vào chính frame đó. Timestamp được giữ cùng frame trong lúc hiển thị và hỏi `defect_visible`, rồi mới dùng để đặt tên file và ghi manifest. Đây là thời điểm **host nhận frame**, không phải hardware exposure timestamp của camera.

Ảnh được lưu PNG lossless theo đúng chiều rộng/cao frame camera trả về. Chương trình không resize, augmentation hoặc inference. Mỗi lần lưu sẽ tính `source_sha256`, append một dòng CSV rồi flush xuống đĩa.

## 4. Schema manifest

Các cột bắt buộc theo yêu cầu:

```text
image_path,fruit_id,session_id,camera_id,capture_time,
original_class,binary_label,view_id,defect_visible,review_status
```

Các cột provenance bổ sung:

```text
capture_time_source,label_reason,source_sha256,
capture_width,capture_height,policy_version,notes
```

Quy ước:

- `capture_time`: ISO 8601 có UTC offset, gắn với frame ngay khi `camera.read()` trả về và trước mọi câu hỏi nhập liệu.
- `capture_time_source`: luôn là `HOST_FRAME_RECEIVED`; OpenCV workflow này không cung cấp hardware capture timestamp đáng tin cậy.
- `image_path`: đường dẫn tương đối từ manifest tới ảnh gốc.
- `review_status`: `APPROVED` hoặc `REVIEW`.
- `binary_label`: lưu dạng text `GOOD` hoặc `BAD`, giống cột `binary_label` trong manifest training hiện tại. Pipeline ánh xạ `GOOD=0`, `BAD=1` khi tạo tensor; CSV pilot không lưu `0/1` trong cột này. `REVIEW` phải để trống.
- `label_reason`: lý do người chụp chọn GOOD/BAD hoặc REVIEW; bắt buộc, đặc biệt quan trọng với `green_orange`.
- `defect_visible`: `true`, `false`, hoặc `unknown` theo view.
- `source_sha256`: SHA-256 của chính file PNG gốc đã lưu.

## 5. Kiểm tra sau pilot

Chưa chạy model. Trước khi xem pilot đạt quy trình hay chưa, kiểm tra:

- mỗi ảnh tồn tại và SHA-256 khớp manifest;
- mọi view cùng quả dùng chung `fruit_id`, `session_id`, class, label và review status;
- `view_id` không trùng trong cùng quả;
- ảnh REVIEW không có `binary_label`;
- ảnh APPROVED có `binary_label` là GOOD/BAD và có `label_reason`;
- `green_orange` không bị suy nhãn tự động từ tên lớp;
- `capture_time_source=HOST_FRAME_RECEIVED` và timestamp đã gắn với frame trước bước hỏi `defect_visible`;
- kích thước lưu đúng frame camera, không có file resize ghi đè;
- vài ảnh mở được, đúng quả và đúng góc đã ghi.

Chỉ sau khi có ảnh thật mới gửi vài dòng manifest để review. Không điền dòng ví dụ vào manifest sản xuất vì chúng có thể bị nhầm là dữ liệu thật.

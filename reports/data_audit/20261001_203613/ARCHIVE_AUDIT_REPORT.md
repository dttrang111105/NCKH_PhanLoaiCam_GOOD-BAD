# Audit archive `OrangeClassificationDataset.zip`

**Thời điểm audit:** 2026-10-01  
**Archive:** `C:/Users/ADMIN/Downloads/OrangeClassificationDataset.zip`  
**Dataset đối chiếu:** `E:/dataset`  
**Chế độ:** chỉ đọc. Member ZIP được đọc trực tiếp vào bộ nhớ; không giải nén ra ổ đĩa, không sửa archive hoặc dataset.

## 1. Kết luận ngắn

### ĐÃ XÁC MINH

1. ZIP tồn tại, đọc được và `testzip()` không phát hiện member lỗi CRC.
2. ZIP chứa đúng 15.000 file, tất cả là JPEG trong 5 thư mục lớp. Không có video, ảnh độ phân giải cao, CSV, JSON, XLSX, XML, README, manifest, log, notebook hoặc script.
3. Toàn bộ 15.000 JPEG trong ZIP là ảnh 224×224, không có EXIF, IPTC, XMP, ICC, camera, thời gian chụp hoặc GPS. Metadata và marker layout giống trạng thái đã thấy trong `E:/dataset`.
4. **15.000/15.000 member có SHA-256 trùng chính xác với đúng một file trong `E:/dataset`.** Không có member không khớp và không có file dataset bị thiếu đối sánh.
5. Vì SHA-256 và kích thước byte bằng nhau, các cặp là cùng một byte stream; pixel cũng tất yếu giống nhau. Không cần chạy pixel hash hay perceptual hash.
6. Archive không chứa ảnh nguồn trước resize/export và không bổ sung bằng chứng về `fruit_id`, `session_id`, `capture_time`, `camera_id`, video/frame hoặc quan hệ nhiều ảnh của cùng quả.

### SUY LUẬN CÓ GIỚI HẠN

- ZIP này là bản đóng gói của chính 15.000 JPEG đang có trong `E:/dataset`, không phải kho ảnh gốc độ phân giải cao.
- Thời gian member ZIP từ `2025-04-27 06:43:10` đến `06:43:30` tạo thành batch 20 giây và trùng dạng timestamp đã quan sát trên file dataset. Đây phù hợp với timestamp đóng gói/khôi phục file, không phải bằng chứng chụp.

### CHƯA CÓ BẰNG CHỨNG

- Không có căn cứ trong archive để nhóm nhiều ảnh theo cùng quả hoặc cùng phiên chụp.
- Không thể biết ảnh 224×224 được tạo từ file, frame video hay pipeline nào.
- Không thể tạo group-aware split đáng tin cậy chỉ từ archive này.

**Kết luận:** archive xác nhận tuyệt đối quan hệ **member ZIP ↔ file dataset hiện tại**, nhưng không khôi phục được **ảnh gốc** hoặc **source group**. Mức tin cậy của split vẫn chỉ là split theo file/ảnh, chưa phải split độc lập theo quả hoặc phiên chụp.

## 2. Thông tin archive

| Thuộc tính | Giá trị |
|---|---|
| Đường dẫn | `C:/Users/ADMIN/Downloads/OrangeClassificationDataset.zip` |
| Kích thước | 112.489.867 byte |
| CreationTime Windows | `2026-07-26 17:01:07 +07:00` |
| LastWriteTime Windows | `2026-07-26 17:01:09 +07:00` |
| SHA-256 archive | `03d43104a0c6eda6681c23415bec8fe404ff6c919d44de35b247bb9b72105251` |
| Số entry | 15.000 |
| Số file | 15.000 |
| Tổng uncompressed | 111.642.109 byte |
| Tổng compressed member | 109.357.845 byte |
| Member mã hóa | 0 |
| Tên member trùng | 0 |
| Đường dẫn member không an toàn | 0 |
| CRC lỗi | 0 |

CreationTime/LastWriteTime ở trên là metadata filesystem của file ZIP hiện tại, không phải thời gian chụp hoặc thời gian chắc chắn archive được tạo ban đầu.

## 3. Inventory member

ZIP chỉ có cấu trúc sau:

| Thư mục member | Số file | Loại |
|---|---:|---|
| `dataset/blackspotted_orange/` | 3.000 | JPEG |
| `dataset/cankered_orange/` | 3.000 | JPEG |
| `dataset/fresh_orange/` | 3.000 | JPEG |
| `dataset/green_orange/` | 3.000 | JPEG |
| `dataset/rotten_orange/` | 3.000 | JPEG |

Không có member thuộc các loại cần tìm khác. Inventory đầy đủ nằm trong [`archive_inventory.csv`](archive_inventory.csv).

## 4. Metadata ảnh trong ZIP

| Trường | File có dữ liệu |
|---|---:|
| JPEG hợp lệ | 15.000 |
| Kích thước 224×224 | 15.000 |
| EXIF | 0 |
| IPTC | 0 |
| XMP / extended XMP | 0 |
| ICC profile | 0 |
| Photoshop APP13 / Adobe APP14 | 0 |
| JPEG comment | 0 |
| Camera Make/Model | 0 |
| DateTimeOriginal | 0 |
| GPS | 0 |

Tất cả ảnh có marker signature:

```text
FFE0-FFDB-FFDB-FFC0-FFC4-FFC4-FFC4-FFC4-FFDA
```

Archive không giữ một phiên bản metadata phong phú hơn dataset trên ổ đĩa.

## 5. Đối chiếu archive với `E:/dataset`

### Phương pháp

1. Tính SHA-256 cho mỗi file hiện có trong `E:/dataset`.
2. Đọc từng member ZIP trực tiếp vào bộ nhớ và tính SHA-256.
3. Ghép theo SHA-256; kiểm tra số lượng đối sánh và kích thước byte.
4. Chỉ cân nhắc pixel/perceptual hash nếu còn member không có exact match.

### Kết quả

| Quan hệ | Số lượng |
|---|---:|
| Member ZIP có đúng một exact match trong dataset | 15.000 |
| Member ZIP không có exact match | 0 |
| Member ZIP khớp nhiều file dataset | 0 |
| File dataset không được ZIP bao phủ | 0 |
| SHA-256 duy nhất trong dataset | 15.000 |
| Ứng viên gần giống cần review | 0 |
| Cặp không thể kết luận | 0 |

Không chạy perceptual hash vì exact SHA-256 đã bao phủ toàn bộ hai phía. Chạy thêm perceptual hash không thể nâng mức chắc chắn cao hơn byte identity và chỉ tạo kết quả dư thừa.

### Dẫn chứng mẫu

| Member ZIP | File dataset | SHA-256 |
|---|---|---|
| `dataset/blackspotted_orange/blackspotted_orange_0000.jpg` | `E:/dataset/blackspotted_orange/blackspotted_orange_0000.jpg` | `2eab171fbdd224e3ecfa070721c809e459928885a02c493e359f282f31bf9080` |
| `dataset/cankered_orange/cankered_orange_0000.jpg` | `E:/dataset/cankered_orange/cankered_orange_0000.jpg` | `cbea607af548bb399ea411253b528681ae16fd0f50c74a13b51c9f6a7e8cff98` |
| `dataset/fresh_orange/fresh_orange_0000.jpg` | `E:/dataset/fresh_orange/fresh_orange_0000.jpg` | `9709b41f8bbbed154e7016e2689296746562203d2f6ad95b57c0a015cc46bdad` |
| `dataset/green_orange/green_orange_0000.jpg` | `E:/dataset/green_orange/green_orange_0000.jpg` | `828ef24da01369336bcc93c83e1a6bc981173138f5d76ff581617fd2d487e2d3` |
| `dataset/rotten_orange/rotten_orange_0000.jpg` | `E:/dataset/rotten_orange/rotten_orange_0000.jpg` | `15fc953099967dc0f35f909fe2a6fab72797ad41d1f02e8ace35d8925eb4c359` |

Mapping đầy đủ nằm trong [`source_mapping_candidates.csv`](source_mapping_candidates.csv). Dù tên file dùng chữ “candidates” theo yêu cầu artifact, 15.000 dòng đều có:

```text
relationship=EXACT_BYTE_IDENTITY
confidence=CONFIRMED
```

Đây là xác nhận file identity, không phải xác nhận hai ảnh thuộc cùng một quả.

## 6. Archive có giúp khôi phục source group không?

**Không.** Archive chỉ bổ sung prefix `dataset/` trước cấu trúc thư mục và tên file hiện tại. Nó không chứa:

- ảnh độ phân giải cao hoặc tên file camera gốc;
- manifest/log ghi quan hệ ảnh;
- `fruit_id`, `session_id`, `camera_id`, capture timestamp;
- video ID/frame ID;
- parent/source hash khác với hash JPEG 224×224;
- script hoặc recipe resize/export.

Vì mỗi member chỉ ánh xạ 1:1 tới chính file dataset hiện tại, archive không tạo thêm cạnh quan hệ nào giữa các ảnh.

## 7. Mức split có thể tin cậy

| Mức đánh giá | Có thể xác nhận? | Lý do |
|---|---|---|
| Không trùng byte giữa các file dataset | Có | 15.000 SHA-256 duy nhất |
| Archive bao phủ dataset hiện tại | Có | 15.000 exact mapping, 0 thiếu |
| Split theo file hiện tại | Có | Mapping bảo toàn file identity |
| Split độc lập theo ảnh nguồn | Không | Không có parent/source identity |
| Split độc lập theo quả | Không | Không có `fruit_id` |
| Split độc lập theo session/video | Không | Không có session/video metadata |

Archive không làm giảm rủi ro near-duplicate/group leakage đã nêu trong audit dữ liệu trước. Exact-hash uniqueness chỉ loại trừ trùng byte, không loại trừ các ảnh khác byte nhưng cùng quả hoặc cùng chuỗi chụp.

## 8. Đúng một việc nên làm tiếp theo

> **Chụp một bộ holdout mới có manifest bắt buộc ghi `fruit_id`, `session_id`, `camera_id`, capture timestamp và source SHA-256 ngay lúc ingest, rồi chỉ dùng bộ này để đánh giá khả năng tổng quát hóa theo quả.**

Archive hiện tại đã được kiểm tra đầy đủ và không còn bằng chứng để khôi phục group ID một cách không suy đoán. Không nên tạo ID giả từ filename, timestamp hoặc độ giống ảnh.

## 9. Artifacts và giới hạn

- [`ARCHIVE_AUDIT_REPORT.md`](ARCHIVE_AUDIT_REPORT.md)
- [`archive_inventory.csv`](archive_inventory.csv): 15.000 dòng member inventory.
- [`source_mapping_candidates.csv`](source_mapping_candidates.csv): 15.000 mapping exact đã xác nhận.
- [`archive_metadata_summary.json`](archive_metadata_summary.json): số liệu tổng hợp máy đọc được.
- [`audit_archive.py`](audit_archive.py): script standard-library dùng cho lần audit này.

Giới hạn:

- Chỉ truy cập đúng ZIP được chỉ định và `E:/dataset` để đối chiếu.
- Không quét thư mục Downloads, ổ đĩa khác, cloud, điện thoại hoặc camera.
- Không giải nén member ra ổ đĩa.
- ZIP timestamp không có timezone và có độ phân giải thời gian hạn chế; không dùng làm capture time.
- Không tạo `fruit_id` hoặc `session_id`.

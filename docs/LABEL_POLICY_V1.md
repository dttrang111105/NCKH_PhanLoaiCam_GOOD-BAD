# Chính sách nhãn GOOD/BAD — phiên bản 1

**Trạng thái:** đã chốt cho pilot và holdout mới  
**Ngày hiệu lực:** 2026-10-01  
**Bài toán:** phân loại chấp nhận/loại bỏ trong quy trình phân loại cam thương mại  
**Không phải:** chẩn đoán bệnh cây hoặc chỉ phát hiện hư hỏng bề mặt

## 1. Quyết định nhãn

| Nhãn | Giá trị | Định nghĩa vận hành |
|---|---:|---|
| `GOOD` | 0 | Quả được chấp nhận cho luồng sản phẩm mục tiêu: đã đạt độ chín yêu cầu và không có khuyết tật thuộc tiêu chí loại bỏ. |
| `BAD` | 1 | Quả phải bị loại khỏi luồng sản phẩm mục tiêu vì chưa chín/không đạt độ chín thương mại hoặc có bệnh, thối hỏng hay khuyết tật bề mặt thuộc tiêu chí loại bỏ. |

`BAD` là nhãn positive của model. Xác suất đầu ra và Recall BAD đều dùng theo nghĩa trên.

## 2. Quy tắc cho từng lớp hiện có

| `original_class` | `binary_label` | Tiêu chí |
|---|---|---|
| `fresh_orange` | `GOOD` | Đạt độ chín thương mại và không có dấu hiệu loại bỏ quan sát được. |
| `green_orange` | `BAD` | Quả được xác nhận là chưa chín hoặc chưa đạt tiêu chuẩn màu/độ chín của luồng sản phẩm mục tiêu. |
| `blackspotted_orange` | `BAD` | Có đốm đen/khuyết tật tương ứng thuộc tiêu chí loại bỏ. |
| `cankered_orange` | `BAD` | Có dấu hiệu canker thuộc tiêu chí loại bỏ. |
| `rotten_orange` | `BAD` | Có dấu hiệu thối, mục hoặc hư hỏng tương ứng thuộc tiêu chí loại bỏ. |

### Trường hợp `green_orange`

- **Chưa chín và không đạt chuẩn thương mại:** gán `BAD`.
- **Chỉ trông xanh trong ảnh do ánh sáng, white balance, bóng đổ hoặc phản xạ:** không được tự động gán `BAD`.
- **Giống cam có màu xanh tự nhiên nhưng đã chín và được chấp nhận theo tiêu chuẩn sản phẩm:** không gán `BAD` chỉ vì màu vỏ.
- Quyết định độ chín phải dựa trên kiểm tra quả thật tại lúc chụp và tiêu chuẩn sản phẩm đã chọn, không suy lại từ pixel sau đó.

## 3. Cấp quyết định và ảnh nhiều góc

1. Gán quyết định thương mại ở cấp **quả** trước khi chụp nhiều góc.
2. Mọi ảnh/frame của cùng quả dùng chung `fruit_id` và cùng quyết định GOOD/BAD ở cấp quả.
3. Với quả BAD có lỗi chỉ nhìn thấy ở một số góc, nên ghi thêm `defect_visible=true/false` cho từng view. Không dùng một view không nhìn thấy lỗi để kết luận model ảnh đơn có khả năng phát hiện lỗi ẩn.
4. Báo cáo classifier trên crop chuẩn phải tách khỏi báo cáo quyết định theo quả/toàn hệ thống.

## 4. Trường hợp không chắc chắn

Không ép nhãn nếu người chụp không đủ bằng chứng. Đánh dấu bản ghi là `REVIEW` hoặc `EXCLUDE` trong trường kiểm soát chất lượng riêng và chưa đưa vào train/validation/test cho đến khi được kiểm tra lại.

Các ví dụ cần review:

- độ chín nằm sát ranh giới chấp nhận;
- màu bị thay đổi rõ bởi nguồn sáng;
- khuyết tật bị che hoặc không đủ nét;
- nhiều loại lỗi mâu thuẫn hoặc không xác định được;
- quả không thuộc phạm vi sản phẩm đã định.

`REVIEW`/`EXCLUDE` là trạng thái quản trị dữ liệu, không phải lớp thứ ba của model nhị phân.

## 5. Quy tắc gán nhãn tại buổi chụp

1. Chỉ dùng những quả chưa có trong dataset cũ.
2. Gán `fruit_id` trước ảnh đầu tiên của quả.
3. Kiểm tra quả thật và ghi `original_class`, `binary_label` trước hoặc ngay trong phiên chụp.
4. Một quả có nhiều lỗi vẫn chỉ có một `binary_label=BAD`; ghi chi tiết lỗi ở trường riêng nếu cần.
5. Không đổi nhãn theo dự đoán của model hiện tại.
6. Không dùng filename, thời gian sửa file, màu pixel hoặc độ giống ảnh để suy `fruit_id`, độ chín hay nhãn.
7. Khi sửa nhãn sau review, phải giữ lịch sử người sửa, thời điểm, lý do và phiên bản policy.

## 6. Phạm vi baseline v1 lịch sử

Checkpoint MobileNetV3-Small, threshold `0.14`, split và metric hiện có được lưu dưới tên **baseline v1 lịch sử**.

- Mapping lịch sử giữ nguyên: `fresh_orange → GOOD`; `green_orange`, `blackspotted_orange`, `cankered_orange`, `rotten_orange → BAD`.
- Không dùng policy này để viết lại manifest cũ hoặc tuyên bố dữ liệu cũ có provenance theo quả.
- Khi đánh giá checkpoint trên holdout mới, phải khóa checkpoint và threshold trước khi mở test.
- Nếu dùng holdout để chọn threshold, sửa model hoặc thay policy, tập đó trở thành validation và không còn là test giữ kín.

## 7. Điều kiện trước khi chụp pilot

Người gán nhãn phải thống nhất bằng ví dụ quả thật về:

- mức độ chín tối thiểu được chấp nhận;
- mức đốm/canker/hư hỏng nhỏ nhất khiến quả bị loại;
- cách xử lý quả có nhiều lỗi;
- cách đánh dấu `REVIEW` và người có quyền chốt nhãn.

Pilot chỉ dùng để kiểm tra thao tác chụp, manifest và tính nhất quán gán nhãn; không dùng để tuyên bố hiệu năng cuối.

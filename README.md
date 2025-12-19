Thành viên 1 – Tiền xử lý + Edge Detection + Adaptive Threshold
Nhiệm vụ chính:
	Chuẩn hóa ảnh (grayscale, blur, noise removal)
	Thực hiện edge detection (Canny)
	Làm adaptive threshold để nhị phân hóa vùng text
	Tạo module tiền xử lý trong code
Kết quả đầu ra:
	Ảnh sau khi lọc nhiễu
	Ảnh cạnh (edges)
	Ảnh nhị phân hóa rõ ràng
Phần viết báo cáo:
	Giải thích Canny
	Các loại threshold (Otsu, adaptive mean, adaptive gaussian)
	Đánh giá độ rõ chữ sau nhị phân

Thành viên 2 – Phát hiện vùng văn bản (MSER + Morphology)
Nhiệm vụ chính
	Áp dụng MSER để tìm candidate text regions
	Dùng morphology (dilation, closing) để gom các vùng text
	Tạo bounding boxes vùng text
	Lọc vùng sai (false positives)
Kết quả đầu ra
	Hộp bao (bounding boxes) vùng chữ
	Vùng text đã gom lại đúng hình dạng
Phần viết báo cáo
	Giải thích MSER hoạt động thế nào
	Vì sao morphology giúp gom text
	Vẽ hình minh họa vùng phát hiện

Thành viên 3 – OCR + Đánh giá + Dataset ICDAR
Nhiệm vụ chính
	Chuẩn bị dataset (ICDAR + ảnh tự thu thập)
	Tách từng vùng text từ module 2
	Chạy OCR (Tesseract OCR hoặc EasyOCR)
	Đánh giá độ chính xác nhận dạng
	Gộp kết quả final pipeline
Kết quả đầu ra
	Văn bản OCR
	So sánh text thật vs OCR output
	Hồ sơ dataset & hướng dẫn chạy repo
Phần viết báo cáo
	OCR hoạt động thế nào
	Precision / Recall của OCR
	Đánh giá pipeline tổng thể

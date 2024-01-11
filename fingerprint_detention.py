# Đọc thư viện và tệp cần thiết
import os
import cv2
import matplotlib.pyplot as plt

# Đọc ảnh mẫu (sample)
sample = cv2.imread("test/1123.BMP")

# Khởi tạo biến lưu điểm số tốt nhất và thông tin về ảnh có điểm số tốt nhất
best_score = 0
filename = None
image = None
kp1, kp2, mp = None, None, None

# Lặp qua từng tệp ảnh trong thư mục "SOCOFing/Real"
counter = 0
for file in os.listdir("SOCOFing\\Real"):
    if counter % 1 == 0:
        print(file)
    counter += 1

    # Đọc ảnh vân tay từ thư mục
    fingerprint_image = cv2.imread(os.path.join("SOCOFing\\Real", file))

    # Kiểm tra nếu ảnh mẫu và ảnh vân tay có tồn tại
    if sample is not None and fingerprint_image is not None:
        # Sử dụng thuật toán SIFT để tìm điểm đặc trưng và mô tả
        sift = cv2.SIFT_create()
        keypoints_1, descriptors_1 = sift.detectAndCompute(sample, None)
        keypoints_2, descriptors_2 = sift.detectAndCompute(fingerprint_image, None)

        # Kiểm tra xem các descriptors có giá trị không rỗng
        if descriptors_1 is not None and descriptors_2 is not None:
            # So khớp các điểm đặc trưng giữa ảnh mẫu và ảnh vân tay
            matches = cv2.FlannBasedMatcher({'algorithm': 1, 'trees': 10}, {}).knnMatch(descriptors_1, descriptors_2, k=2)
            match_points = []

            # Lọc các điểm khớp dựa trên ngưỡng
            for p, q in matches:
                if p.distance < 0.1 * q.distance:
                    match_points.append(p)

            keypoints = min(len(keypoints_1), len(keypoints_2))

            # Tính toán điểm số dựa trên số điểm khớp và số điểm đặc trưng
            if keypoints > 0:
                current_score = len(match_points) / keypoints * 1

                # Cập nhật điểm số tốt nhất và lưu thông tin về ảnh có điểm số tốt nhất
                if current_score > best_score:
                    best_score = current_score
                    filename = file
                    image = fingerprint_image
                    kp1, kp2, mp = keypoints_1, keypoints_2, match_points

# In ra tên file và điểm số khớp tốt nhất
print("BEST MATCH:", filename)
print("Score:", best_score)

# Hiển thị kết quả của điểm khớp tốt nhất (nếu có)
if image is not None:
    result = cv2.drawMatches(sample, kp1, image, kp2, mp, None)
    result = cv2.resize(result, None, fx=4, fy=4)

    # Sử dụng Matplotlib để hiển thị ảnh
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.axis('off')  # Tắt trục
    plt.show()
else:
    print("Không tìm thấy ảnh với điểm khớp tốt nhất.")

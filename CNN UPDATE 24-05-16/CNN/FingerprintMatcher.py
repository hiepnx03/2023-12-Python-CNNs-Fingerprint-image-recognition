import os
import cv2
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


class FingerprintMatcher:
    def __init__(self, root):
        self.root = root
        self.root.title("Fingerprint Matching")

        # Nút để chọn ảnh
        self.browse_button = tk.Button(self.root, text="Browse Image", command=self.browse_image)
        self.browse_button.pack(pady=10)

        # Nhãn để hiển thị ảnh mẫu
        self.label_image = tk.Label(self.root)
        self.label_image.pack()

        # Thanh tiến trình
        self.progress_bar = ttk.Progressbar(self.root, orient="horizontal", length=300, mode="determinate")
        self.progress_bar.pack(pady=10)

        # Nhãn để hiển thị kết quả
        self.result_label = tk.Label(self.root, text="")
        self.result_label.pack(pady=10)

        # Nhãn để hiển thị ảnh đối chiếu
        self.match_image_label = tk.Label(self.root)
        self.match_image_label.pack()

    def match_fingerprint(self, sample_image_path):
        # Đọc ảnh mẫu
        sample = cv2.imread(sample_image_path)

        # Khởi tạo biến lưu điểm số tốt nhất và thông tin về ảnh có điểm số tốt nhất
        best_score = 0
        best_match_filename = None
        best_match_image = None
        best_match_kp1, best_match_kp2, best_match_mp = None, None, None

        # Lặp qua từng tệp ảnh trong thư mục database
        database_folder = "SOCOFing/test100IMGMatch"
        num_files = len(os.listdir(database_folder))
        for i, file in enumerate(os.listdir(database_folder)):
            self.progress_bar["value"] = (i + 1) * 100 / num_files
            self.progress_bar.update()

            # Đường dẫn tới ảnh trong thư mục database
            fingerprint_image_path = os.path.join(database_folder, file)

            # Đọc ảnh vân tay từ thư mục
            fingerprint_image = cv2.imread(fingerprint_image_path)

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
                            best_match_filename = file
                            best_match_image = fingerprint_image
                            best_match_kp1, best_match_kp2, best_match_mp = keypoints_1, keypoints_2, match_points

        return sample, best_match_filename, best_score, best_match_image, best_match_kp1, best_match_kp2, best_match_mp

    def browse_image(self):
        # Mở hộp thoại để chọn ảnh
        filename = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")])
        if filename:
            # Đọc và hiển thị ảnh mẫu
            image = Image.open(filename)
            image.thumbnail((600, 600))  # Thay đổi kích thước ảnh để hiển thị
            photo = ImageTk.PhotoImage(image)
            self.label_image.configure(image=photo)
            self.label_image.image = photo

            # Tìm kiếm ảnh tương đối nhất
            sample_image, best_match_filename, best_score, best_match_image, kp1, kp2, mp = self.match_fingerprint(filename)

            # Hiển thị kết quả
            similarity_percentage = best_score * 100
            self.result_label.config(text=f"BEST MATCH: {best_match_filename}\nScore: {best_score}\nSimilarity: {similarity_percentage:.2f}%")

            if best_match_image is not None:
                # Hiển thị ảnh đối chiếu
                result_image = cv2.cvtColor(best_match_image, cv2.COLOR_BGR2RGB)
                result_image = Image.fromarray(result_image)
                result_image.thumbnail((600, 600))
                result_photo = ImageTk.PhotoImage(result_image)
                self.match_image_label.configure(image=result_photo)
                self.match_image_label.image = result_photo

                # Hiển thị plot các điểm khớp
                self.show_plot(sample_image, best_match_image, kp1, kp2, mp)
            else:
                self.match_image_label.configure(image="")

    def show_plot(self, sample_image, best_match_image, kp1, kp2, mp):
        # Tạo một cửa sổ mới để hiển thị plot
        plot_window = tk.Toplevel(self.root)
        plot_window.title("Matching Plot")

        # Vẽ các điểm khớp
        match_img = cv2.drawMatches(sample_image, kp1, best_match_image, kp2, mp, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        # Chuyển đổi ảnh sang định dạng RGB
        match_img_rgb = cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB)

        # Tạo một figure cho plot
        fig = plt.figure(figsize=(10, 5))

        # Hiển thị ảnh với các điểm khớp
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(match_img_rgb)
        ax.set_title("Matching Keypoints")
        ax.axis('off')

        # Điều chỉnh bố cục
        plt.tight_layout()

        # Hiển thị plot trong cửa sổ mới
        canvas = FigureCanvasTkAgg(fig, master=plot_window)
        canvas.draw()
        canvas.get_tk_widget().pack()
        toolbar = NavigationToolbar2Tk(canvas, plot_window)
        toolbar.update()
        canvas.get_tk_widget().pack()


if __name__ == "__main__":
    root = tk.Tk()
    app = FingerprintMatcher(root)
    root.mainloop()

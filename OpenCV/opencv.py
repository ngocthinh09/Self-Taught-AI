import cv2  
import numpy as np  
import matplotlib.pyplot as plt  
import math

def load_and_process_image(image_path):  
    # Load ảnh
    img = cv2.imread(image_path)
    # Tham số flags mặc định là cv2.IMREAD_COLOR -> dưới dạng BGR (Blue, Green, Red)
    # Các đối số khác cho flags: cv2.IMREAD_GRAYSCALE -> ảnh xám, cv2.IMREAD_UNCHANGED -> ảnh trong suốt (độ alpha)
     
    height, width, channels = img.shape  
    print(f"Kích thước ảnh: {width}x{height}, Kênh màu: {channels}")  
    
    # Chuyển sang grayscale  
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
    
    # Chuyển sang RGB
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
    
    # Hiển thị ảnh  
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)  
    plt.imshow(rgb)  
    plt.title('Ảnh màu')  
    plt.axis('off')  
    
    plt.subplot(1, 2, 2)  
    plt.imshow(gray, cmap='gray')  
    plt.title('Ảnh xám')  
    plt.axis('off')  
    
    plt.tight_layout()  
    plt.show()  
    
    return img, gray  

def load_and_resize_img(image_path):
    # Đọc ảnh  
    img = cv2.imread(image_path)  
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Chuyển sang RGB để hiển thị với matplotlib  
 
    methods = [  
        ("INTER_NEAREST", cv2.INTER_NEAREST), 
        ("INTER_LINEAR", cv2.INTER_LINEAR),         # Mặc định opencv sẽ sử dụng method này
        ("INTER_CUBIC", cv2.INTER_CUBIC),  
        ("INTER_AREA", cv2.INTER_AREA),  
        ("INTER_LANCZOS4", cv2.INTER_LANCZOS4)  
    ]  

    upscale_dim = (img.shape[1]*2, img.shape[0]*2)      # Phóng to
    downscale_dim = (img.shape[1]//5, img.shape[0]//5)  # Thu nhỏ

    # Hiển thị ảnh với matplotlib 
    plt.figure(figsize=(15, 10))  
    
    plt.subplot(3, 3, 1)  
    plt.imshow(img_rgb)  
    plt.title('Ảnh gốc')  
    plt.axis('off')  
 
    i = 2  
    for name, method in methods:  
        # Thu nhỏ ảnh  
        resized_small = cv2.resize(img_rgb, downscale_dim, interpolation=method)  
        plt.subplot(3, 3, i)  
        plt.imshow(resized_small)  
        plt.title(f'Thu nhỏ - {name}')  
        plt.axis('off')  
        i += 1  

    plt.tight_layout()  
    plt.show() 
    
    
def apply_kernel(img, kernel, kernel_name=None, show_original=True, normalize_result=False, add_offset=False, cmap=None, figsize=(12, 6), show_result = True, save_path=None):  
    """  
    Áp dụng kernel lên ảnh và hiển thị kết quả  
    
    Parameters:  
    -----------  
    image_path: str  
        Đường dẫn đến ảnh cần xử lý  
    kernel: numpy.ndarray  
        Kernel để áp dụng lên ảnh  
    kernel_name: str, default=None  
        Tên của kernel để hiển thị (nếu None, sẽ dùng "Custom Kernel")  
    show_original: bool, default=True  
        Hiển thị ảnh gốc bên cạnh kết quả nếu True  
    normalize_result: bool, default=False  
        Chuẩn hóa kết quả về dải [0, 255] nếu True  
    add_offset: bool, default=False  
        Thêm offset 128 vào kết quả (thường dùng cho emboss) nếu True  
    cmap: str, default=None  
        Bảng màu cho hiển thị kết quả, nếu None thì tự động chọn  
    figsize: tuple, default=(12, 6)  
        Kích thước của figure matplotlib  
    save_path: str, default=None  
        Đường dẫn để lưu kết quả (nếu None, không lưu)  
    
    Returns:  
    --------  
    result: numpy.ndarray  
        Ảnh sau khi áp dụng kernel  
    """  
    
    # Kiểm tra kernel  
    if not isinstance(kernel, np.ndarray):  
        try:  
            kernel = np.array(kernel, dtype=np.float32)  
        except:  
            raise ValueError("The kernel must be a NumPy array or convertible to a NumPy array.")  
    
    # Chuyển đổi sang RGB cho hiển thị  
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
    
    # Áp dụng kernel  
    result = cv2.filter2D(img, -1, kernel)  
    
    # Xử lý kết quả  
    if normalize_result:  
        result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)  
    
    if add_offset:  
        # Chuyển sang grayscale nếu là ảnh màu  
        if len(result.shape) == 3:  
            result_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)  
            result_gray = cv2.add(result_gray, 128)  
            result = cv2.cvtColor(result_gray, cv2.COLOR_GRAY2BGR)  
        else:  
            result = cv2.add(result, 128)  
    
    # Xác định tên kernel  
    if kernel_name is None:  
        kernel_name = "Custom Kernel"  
    
    # Chuyển đổi sang RGB cho hiển thị  
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)  
    
    # Xác định cmap  
    if cmap is None:  
        if len(result_rgb.shape) == 2 or (len(result_rgb.shape) == 3 and result_rgb.shape[2] == 1):  
            cmap = 'gray'  
    
    # Hiển thị kết quả  
    if show_result:
        if show_original:  
            plt.figure(figsize=figsize)  
            plt.subplot(121)  
            plt.imshow(img_rgb)  
            plt.title('Original image')  
            plt.axis('off')  
            
            plt.subplot(122)  
            plt.imshow(result_rgb, cmap=cmap)  
            plt.title(f"After applying {kernel_name}")  
            plt.axis('off')
              
            plt.tight_layout()
            plt.show()
        else:  
            plt.figure(figsize=(figsize[0]//2, figsize[1]))  
            plt.imshow(result_rgb, cmap=cmap)  
            plt.title(f'After applying {kernel_name}')  
            plt.axis('off')  

            plt.tight_layout()
            plt.show()
    
    # Lưu kết quả nếu có đường dẫn  
    if save_path:  
        try:  
            cv2.imwrite(save_path, result)  
            print(f"Saved at: {save_path}")  
        except Exception as e:  
            print(f"Error when saving results.: {e}")  
    
    return result
    
image_path = 'img/nhp.jpg'
image = cv2.imread(image_path, cv2.IMREAD_COLOR)

kernels = {  
    "Box Blur 3x3": np.ones((3, 3), np.float32) / 9,  
    "Box Blur 5x5": np.ones((5, 5), np.float32) / 25,  
    "Gaussian 5x5": cv2.getGaussianKernel(5, 1) * cv2.getGaussianKernel(5, 1).T,  
    "Sobel X": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),  
    "Sobel Y": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),  
    "Laplacian": np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]),  
    "Sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),  
    "Strong Sharpen": np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]),  
    "Emboss NW": np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]]),  
    "Edge Detection": np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),  
} 

apply_kernel(img = image, kernel=kernels['Box Blur 5x5'], kernel_name='Box Blur 5x5', cmap = 'gray', normalize_result = True, show_result=False)
apply_kernel(img = image, kernel=kernels['Edge Detection'], kernel_name='Edge Detection', cmap = 'gray', normalize_result = True)
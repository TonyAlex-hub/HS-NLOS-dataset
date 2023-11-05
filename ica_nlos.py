
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
from PIL import Image
# import cv
# 加载RGB图像
image_path = r'D:\dataset\dm\test\k\501.png'
image = Image.open(image_path)
image=image.resize((512,512))
image_array = np.array(image)

##高斯噪声
# mean = 0
# stddev = 75
# noise = np.random.normal(mean, stddev, image_array.shape).astype(np.uint8)
# noisy_image_array = np.clip(image_array + noise, 0, 255)

##椒盐噪声
# salt_prob = 0.8  # 添加白点的概率
# pepper_prob = 0.2  # 添加黑点的概率
# salt = (np.random.random(image_array.shape) < salt_prob) * 255
# pepper = (np.random.random(image_array.shape) < pepper_prob) * 0
# noisy_image_array = np.clip(image_array + salt + pepper, 0, 255).astype(np.uint8)
# 获取图像的形状
height, width, channels = image_array.shape
#
# 重塑图像数据以进行ICA
reshaped_image = image_array.reshape(-1, channels)

# 进行独立成分分解
n_components = 3  # 通常与通道数相同
ica = FastICA(n_components=n_components)
ica.fit(reshaped_image)

# 获取独立成分
independent_images = ica.transform(reshaped_image)

# 恢复独立成分为图像
restored_images = ica.inverse_transform(independent_images)

# 重塑图像数据为原始形状
restored_image_array = restored_images.reshape(height, width, channels).astype(np.uint8)

# 显示原始图像和分离的图像
plt.figure(figsize=(10, 5))
plt.subplot(1, 6, 1)
plt.title("Original Image")
plt.imshow(image_array)
plt.axis('off')

# plt.subplot(1, n_components + 4, 2)
# plt.title("Noise Image")
# plt.imshow(noisy_image_array)
# plt.axis('off')

plt.subplot(1, 6, 2)
plt.title("reconstructed")
plt.imshow(restored_image_array)
plt.axis('off')
for i in range(n_components):
    plt.subplot(1, 6, i + 3)
    plt.title(f"Independent {i+1}")

    plt.imshow(restored_image_array[..., i], cmap='gray')
    plt.axis('off')
R = restored_image_array[..., 0]
G = restored_image_array[..., 1]
B = restored_image_array[..., 2]
rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
rgb_image[:, :, 0] = R  # 红色通道
rgb_image[:, :, 1] = G  # 绿色通道
rgb_image[:, :, 2] = 0  # 蓝色通道
image=Image.fromarray(rgb_image)
image.save("procesed_image.png")
# 将两个图像合并为一个三通道RGB图
# rgb_image = np.dstack((G, G, G))
plt.subplot(1, 6, 6)
plt.title("RG0")
plt.imshow(image)
plt.axis('off')
plt.show()

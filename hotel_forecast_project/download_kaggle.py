import kagglehub
import os

print("开始下载酒店预订数据集...")
path = kagglehub.dataset_download("jessemostipak/hotel-booking-demand")
print(f"数据集已下载到: {path}")

# 检查下载的文件
print("\n下载的文件:")
for file in os.listdir(path):
    file_path = os.path.join(path, file)
    size_kb = os.path.getsize(file_path) / 1024
    print(f"- {file} ({size_kb:.2f} KB)")

print("\n完成! 数据已准备好进行处理。") 
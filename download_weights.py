import requests
from tqdm import tqdm

url = 'https://pku-milab-model.oss-ap-northeast-1.aliyuncs.com/fold0.pth'
response = requests.get(url, stream=True)
total_size_in_bytes = int(response.headers.get('content-length', 0))
with open('weights/fold0.pth', 'wb') as file:
    with tqdm(total=total_size_in_bytes, unit='B', unit_scale=True) as bar:
        # 分块下载文件
        for chunk in response.iter_content(chunk_size=1024):
            file.write(chunk)
            bar.update(len(chunk))  # 更新进度条
url = 'https://pku-milab-model.oss-ap-northeast-1.aliyuncs.com/fold1.pth'
response = requests.get(url, stream=True)
total_size_in_bytes = int(response.headers.get('content-length', 0))
with open('weights/fold1.pth', 'wb') as file:
    with tqdm(total=total_size_in_bytes, unit='B', unit_scale=True) as bar:
        # 分块下载文件
        for chunk in response.iter_content(chunk_size=1024):
            file.write(chunk)
            bar.update(len(chunk))  # 更新进度条
url = 'https://pku-milab-model.oss-ap-northeast-1.aliyuncs.com/fold2.pth'
response = requests.get(url, stream=True)
total_size_in_bytes = int(response.headers.get('content-length', 0))
with open('weights/fold2.pth', 'wb') as file:
    with tqdm(total=total_size_in_bytes, unit='B', unit_scale=True) as bar:
        # 分块下载文件
        for chunk in response.iter_content(chunk_size=1024):
            file.write(chunk)
            bar.update(len(chunk))  # 更新进度条

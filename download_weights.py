import requests
from tqdm import tqdm

url = 'https://pku-milab-model.oss-ap-northeast-1.aliyuncs.com/fold0.pth?Expires=1737170150&OSSAccessKeyId=TMP.3Ke4Ku8muLzHYpqCYYqxepgdvKSiKFBzaEzfX4wwX1mWUa8aEjJqAUy2tZPbiZMmcwBqpFtXm7JgnVkUW3HRYqYVDhnRMM&Signature=FZxa4jjlqolf9EShX%2Bk9OzRvQfM%3D'

response = requests.get(url, stream=True)
total_size_in_bytes = int(response.headers.get('content-length', 0))
with open('.weights/fold0.pth', 'wb') as file:
    with tqdm(total=total_size_in_bytes, unit='B', unit_scale=True) as bar:
        # 分块下载文件
        for chunk in response.iter_content(chunk_size=1024):
            file.write(chunk)
            bar.update(len(chunk))  # 更新进度条
print('文件下载成功')

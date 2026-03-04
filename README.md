移动版 label studio模型后端

本地构建运行如下命令:
```
embedded_python\python.exe get-pip.py
embedded_python\python.exe -m pip install setuptools wheel --target embedded_python\Lib\site-packages
```

```
embedded_python\python.exe -m pip install -r requirements.txt --target embedded_python\Lib\site-packages
```


创建模型仓库：
```
embedded_python\python.exe -m label_studio_ml.server create my_ml_backend
```


本地运行:
设置本地存储路径
```
$env:LABEL_STUDIO_BASE_DATA_DIR="D:\ls-data"
```
```
set LABEL_STUDIO_URL=http://127.0.0.1:8080
embedded_python\python.exe -m label_studio_ml.server start my_ml_backend
```

# 前端模型访问格式
http://127.0.0.1:9090/model/<task>/<model_type>/<model_name>

<task>支持 detect segment obb
<model_type>支持yolov5 yolov8 yolov9 yolov10 yolo11 yolo26 sam2
model_name为pt文件名，后缀如果是_640会用640尺寸推理，其他数字用对应的尺寸，默认640.

# SAM2 内存参数（portable 推荐）
大图在 portable 环境下可能触发 `MemoryError`。可设置：
```
$env:SAM2_MAX_IMAGE_SIDE="1536"
```
默认值为 `2048`，值越小越省内存（但分割细节会下降）。设为 `0` 表示不缩放。
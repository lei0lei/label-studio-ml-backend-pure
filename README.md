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
```
embedded_python\python.exe -m label_studio_ml.server start my_ml_backend
```

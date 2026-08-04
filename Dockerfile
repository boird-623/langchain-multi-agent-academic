FROM python:3.11-slim

WORKDIR /app

# 先复制依赖文件安装（利用Docker缓存，代码变了不用重新安装依赖）
COPY requirements.txt .
RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 再复制代码
COPY . .

EXPOSE 7860

CMD ["python", "app.py"]

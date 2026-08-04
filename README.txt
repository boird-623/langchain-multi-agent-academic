=== 项目一：LangChain多Agent学术助手 ===

【第一步】把 .env.example 复制一份，改名为 .env
【第二步】打开 .env，把 DEEPSEEK_API_KEY= 后面换成你的真实Key

【第三步】安装依赖（在终端里运行）：
pip install langchain langchain-openai langchain-core openai gradio python-dotenv arxiv -i https://pypi.tuna.tsinghua.edu.cn/simple

【第四步】运行程序：
python app.py

【第五步】打开浏览器访问：
http://localhost:7860

【停止程序】在终端按 Ctrl + C

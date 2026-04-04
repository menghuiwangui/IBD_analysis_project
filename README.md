在飞桨平台运行的步骤

步骤1：上传项目到飞桨

1. 访问 aistudio.baidu.com

2. 创建新项目

3. 上传项目文件

步骤2：设置环境
```bash
# 在飞桨的Terminal中执行
pip install -r requirements_paddle.txt
```

步骤3：修改配置
```bash
# 修改 config/settings_paddle.py
EMAIL = "your_real_email@example.com"  # 改成你的邮箱

# 如果使用spaCy
python -m spacy download en_core_web_sm
```

步骤4：运行程序
```bash
# 测试模式（获取1万篇）
python paddle_main.py --total 10000 --batch-size 1000 --topics 20

# 正式运行（获取60万篇）
python paddle_main.py --total 600000 --batch-size 10000 --topics 80

# 运行完整分析
python paddle_main.py --total 600000 --topics 80 --optimize-memory

# 如果中断，可以继续
python paddle_main.py --skip-crawl --topics 80

# 查看监控日志
ls -la data/logs/

# 查看检查点
ls -la data/checkpoints/

# 查看进度
cat data/checkpoints/ibd_analysis_state.json
```
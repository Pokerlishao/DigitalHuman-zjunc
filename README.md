# DigitalHuman-zjunc
浙江大学南昌研究院数字人项目

## 环境配置
使用 pip 安装相关 python 库
```shell
pip install -r requirements.txt
```

### 文本转语音（Text to Speech）
#### Linux
输入命令自动配置并下载模型参数：
```shell
sh config.sh
```
#### Windows
Powershell输入命令自动配置：
```
config.bat
```
手动下载 [C_latest.pth](https://huggingface.co/colin1639/VITS-fast-fine-tuning/blob/main/G_latest.pth) 到目录 `./tts/checkpoint`
### 语音转文本（Speech to Text）
采用 whisper 开发，运行时自动下载`medium`模型。

## 运行
输入命令运行测试，会通过文本生成一段语音 `output.wav`， 并输出从语音转换后的文本：
```python
python3 run.py
```
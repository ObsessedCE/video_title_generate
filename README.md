# 字幕生成
自动根据视频音频生成视频字幕并内嵌至原视频中生成一个新的视频文件

## 依赖
- 基于python3
- 需要本地下载ffmpeg,并加入到环境变量中
- 基于openwhisper进行语音识别

## 用法
```python
# 先安装好依赖
pip -r requirement.txt

# -i 指定视频文件位置 -o 指定新视频的生成的目录位置
python -i input_video_path -o output_video_directory video_util.py
```
## 代做
- 视频字幕翻译
- 显卡优化翻译速度
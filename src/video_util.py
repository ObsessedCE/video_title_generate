__author__ = 'ObsessedCE'
import os
import multiprocessing
import subprocess
import time
import psutil
import whisper
import tqdm
from decimal import Decimal
import shutil
import argparse


# 从视频中提取音频
def extract_video_audio(input_video, output_audio):
    """
    提取目标视频的字幕
    :param input_video:
    :param output_video:
    :return:
    """
    command = [
        "ffmpeg", "-i", input_video, "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1", output_audio
    ]
    subprocess.run(command, check=True)
    print(f"音频文件提取成功")


# srt字幕和视频结合
def merge_srt_with_video(video_path, srt_path, output_path):
    # 检查输入文件是否存在
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频文件未找到: {video_path}")
    if not os.path.exists(srt_path):
        raise FileNotFoundError(f"SRT 字幕文件未找到: {srt_path}")

    command = [
        "ffmpeg",
        "-i", video_path,
        "-vf", f"subtitles={srt_path}",
        "-c:a", "copy",
        "-scodec", "mov_text",  # 使用mov_text编码
        output_path
    ]
    try:
        # 使用 subprocess.run 执行命令
        subprocess.run(command, check=True)
        print(f"合并完成，输出文件为: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg 合并字幕时出错: {e}")
    except Exception as ex:
        print(f"发生异常: {ex}")


# 将音频进行分割
def split_audio(input_audio, segment_time=60, output_dir="../output/"):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "output%03d.wav")
    command = [
        "ffmpeg", "-i", input_audio,
        "-f", "segment", "-segment_time", str(segment_time),
        "-c", "copy", output_path
    ]
    # check 检查是否执行失败
    subprocess.run(command, check=True)
    print(f"音频分割完成, 音频文件存储在{output_dir}")
    return output_dir


# 获取指定目录下的wav文件
def get_audio_files_from_folder(folder_path):
    """
    获取音频片段文件列表
    :param folder_path:
    :return:
    """
    audio_files = [os.path.join(folder_path, f) for f in sorted(os.listdir(folder_path)) if f.endswith(".wav")]
    return audio_files


# 并行音频识别
def process_audio_files_in_parallel(audio_files):
    """
    多进程进行音频识别
    :param audio_files:
    :return:
    """
    physical_cores = psutil.cpu_count(logical=False)
    cpu_count = min(physical_cores, len(audio_files))
    # 加载模型文件
    model = init_model()
    with multiprocessing.Manager() as manager:
        shared_model = manager.Value("model", model)
        with multiprocessing.Pool(processes=cpu_count) as pool:
            args = [(audio_file, shared_model.value) for audio_file in audio_files]
            # 使用 tqdm 包装 pool.imap，保证结果顺序一致
            all_segments = list(tqdm.tqdm(pool.imap(transcribe_audio, args), total=len(audio_files)))
            return all_segments


def merge_subtitles_in_memory(all_segments, segment_time):
    subtitles = []
    index = 0
    last_end = Decimal("0.0")
    for i, segment_list in enumerate(all_segments):
        offset = Decimal(i) * Decimal(str(segment_time))
        for segment in segment_list:
            # 计算当前段的起始和结束时间，基于分段的偏移量
            start_time = Decimal(str(segment["start"])) + offset
            end_time = Decimal(str(segment["end"])) + offset
            # 浮点计算会缺失精度， 防止出现字幕重叠
            if start_time < last_end:
                start_time = last_end + Decimal("0.1")

            last_end = end_time

            # 将时间戳转换为SRT格式
            start_time_str = convert_seconds_to_srt_time(start_time)
            end_time_str = convert_seconds_to_srt_time(end_time)

            text = segment["text"]
            subtitles.append({
                "index": index + 1,
                "start_time": start_time_str,
                "end_time": end_time_str,
                "text": text
            })
            index = index + 1
    return subtitles


def convert_seconds_to_srt_time(seconds):
    """
    将秒数转换为 SRT 字幕时间格式: HH:MM:SS,SSS
    """
    hours = int(seconds // 3600)  # 计算小时数
    minutes = int((seconds % 3600) // 60)  # 计算分钟数
    seconds = seconds % 60  # 计算秒数
    milliseconds = int((seconds % 1) * 1000)  # 计算毫秒数
    # 格式化时间为 SRT 格式，确保毫秒部分为三位数
    time_str = f"{hours:02}:{minutes:02}:{int(seconds):02},{milliseconds:03}"
    return time_str


# 将合并后的字幕内容输出为 SRT 格式（或用于其他操作）
def generate_srt_from_memory(subtitles):
    srt_content = ""
    for subtitle in subtitles:
        srt_content += f"{subtitle['index']}\n"
        srt_content += f"{subtitle['start_time']} --> {subtitle['end_time']}\n"
        srt_content += f"{subtitle['text']}\n\n"

    return srt_content


def save_srt_file(subtitles, file_path):
    srt_content = generate_srt_from_memory(subtitles)
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(srt_content)
    return file_path


def generate_tmp_directory():
    # 根据时间戳生临时文件夹存储中间文件
    sub_folder = int(time.time())
    output_dir = "./tmp/" + str(sub_folder) + "/"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def get_filename_from_path(file_path):
    return os.path.basename(file_path)


def init_model():
    start_time = int(time.time())
    model_name = "large-v1"
    model = whisper.load_model(model_name, download_root="../data")
    end_time = int(time.time())
    print(f"加载模型花费时间{end_time - start_time} s")
    return model


def transcribe_audio(args):
    global model
    audio_file, model = args
    if model is None:
        raise ValueError("Model is not loaded in the process.")
    result = model.transcribe(audio_file, fp16=False, word_timestamps=True)
    return result["segments"]


def delete_folder(folder_path):
    if os.path.exists(folder_path):
        try:
            shutil.rmtree(folder_path)
            print(f"删除临时目录成功 {folder_path}")
        except Exception  as e:
            print(f"删除临时目录失败")
    else:
        print(f"指定要删除的临时目录不存在 {folder_path}")


def argparse_setup():
    parser = argparse.ArgumentParser(description="这是一个示范程序，演示如何使用命令行参数。")
    parser.add_argument('-i', '--input_file', type=str, help='指定输入视频文件位置', required=True)
    parser.add_argument('-o', '--output_folder', type=str, help='指定目标视频文件所在文件夹', required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = argparse_setup()
    input_video = args.input_file
    output_folder = args.output_folder
    start_time = int(time.time())

    delete_tmp_directory_flag = True
    # 临时文件夹
    tmp_directory = generate_tmp_directory()

    segment_time = 60
    file_name = get_filename_from_path(input_video)
    output_audio = tmp_directory + file_name.split(".")[0] + ".wav"
    # 提取音频
    extract_video_audio(input_video, output_audio)
    # 音频分段
    segment_audio_dir = split_audio(output_audio, segment_time=segment_time,
                                    output_dir=tmp_directory + "/segment-audio/")
    # 获取指定文件夹下的音频文件
    audio_files = get_audio_files_from_folder(segment_audio_dir)
    # 并行识别音频
    all_segment = process_audio_files_in_parallel(audio_files)
    subtitles = merge_subtitles_in_memory(all_segment, segment_time=segment_time)
    # # 保存汇总的srt文件
    srt_file_path = save_srt_file(subtitles, tmp_directory + file_name.split(".")[0] + ".srt")
    # 将视频和字幕合并生成新的视频
    output_file_path = os.path.join(output_folder, file_name)
    merge_srt_with_video(input_video, srt_file_path, output_file_path)
    end_time = int(time.time())
    print(f"视频字幕生成耗费时间{end_time - start_time}")
    # 溢出中间临时目录
    if delete_tmp_directory_flag:
        delete_folder(tmp_directory)

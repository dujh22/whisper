# -*- coding:utf-8 -*-

import os
import sounddevice as sd
import numpy as np

fs = 16000  # 采样率

# 获取当前脚本所在的路径
current_script_path = os.path.dirname(os.path.abspath(__file__))
# 拼接保存的音频文件路径
audio_file_path = os.path.join(current_script_path, f'recording.pcm')

def get_and_process_audio(seconds = 10): # 录制时长（秒）
    # 录音
    channels = 1  # 通道数

    print("录制中...")
    # myrecording是numpy数组
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=channels, dtype='int16')
    sd.wait()  # 等待录制完成
    print("录制完成")

    # 保存音频为pcm格式
    # 直接将 numpy 数组写入文件
    myrecording.tofile(audio_file_path)

def play_record():
    # 读取已有的 PCM 文件
    pcm_file = audio_file_path
    pcm_data = np.fromfile(pcm_file, dtype='int16')

    # 播放音频
    print("播放PCM音频...")
    sd.play(pcm_data, samplerate=fs)
    sd.wait()  # 等待播放完成
    print("播放完成")

if __name__ == "__main__":
    seconds = 5
    get_and_process_audio(seconds)
    play_record()


# -*- coding:utf-8 -*-

import os
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wavfile

fs = 16000  # 采样率

# 获取当前脚本所在的路径
current_script_path = os.path.dirname(os.path.abspath(__file__))
# 拼接保存的音频文件路径
wav_file_path = os.path.join(current_script_path, 'recording.wav')

def get_and_process_audio(seconds = 10): # 录制时长（秒）
    # 录音
    channels = 1  # 通道数

    print("录制中...")
    # myrecording是numpy数组
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=channels, dtype='int16')
    sd.wait()  # 等待录制完成
    print("录制完成")

    # 保存音频为wav格式
    wavfile.write(wav_file_path, fs, myrecording)

def play_record():
    # 读取已有的 WAV 文件
    sample_rate, data = wavfile.read(wav_file_path)

    # 播放音频
    print("播放PCM音频...")
    sd.play(data, samplerate=sample_rate)
    sd.wait()  # 等待播放完成
    print("播放完成")

if __name__ == "__main__":
    seconds = 5
    get_and_process_audio(seconds)
    play_record()


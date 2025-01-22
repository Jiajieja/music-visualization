import subprocess
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
import numpy as np

# 设置 sound 文件夹的路径
sound_dir = "/Users/xiaojie/Desktop/Sound"

# 执行 spleeter 命令
command = [
    "spleeter", "separate", 
    "-o", "audio_output", 
    "-p", "spleeter:4stems", 
    "1.mp3"
]

# 运行命令并捕获输出
result = subprocess.run(command, cwd=sound_dir, capture_output=True, text=True)

# 输出命令结果
print("STDOUT:", result.stdout)
print("STDERR:", result.stderr)


# 指定分离后的音频文件目录
output_dir = "/Users/xiaojie/Desktop/Sound/audio_output/1"

# 加载分离后的音轨
vocals, sr = librosa.load(os.path.join(output_dir, "vocals.wav"), sr=None)
drums, sr = librosa.load(os.path.join(output_dir, "drums.wav"), sr=None)
bass, sr = librosa.load(os.path.join(output_dir, "bass.wav"), sr=None)
other, sr = librosa.load(os.path.join(output_dir, "other.wav"), sr=None)

# 绘制波形图
plt.figure(figsize=(10, 6))
plt.subplot(4, 1, 1)
librosa.display.waveshow(vocals, sr=sr, alpha=0.7)
plt.title("Vocals")

plt.subplot(4, 1, 2)
librosa.display.waveshow(drums, sr=sr, alpha=0.7)
plt.title("Drums")

plt.subplot(4, 1, 3)
librosa.display.waveshow(bass, sr=sr, alpha=0.7)
plt.title("Bass")

plt.subplot(4, 1, 4)
librosa.display.waveshow(other, sr=sr, alpha=0.7)
plt.title("Other")

plt.tight_layout()
plt.show()


def plot_spectrogram(y, sr, title):
    D = librosa.amplitude_to_db(abs(librosa.stft(y)), ref=np.max)
    plt.figure(figsize=(8, 4))
    librosa.display.specshow(D, sr=sr, x_axis="time", y_axis="log")
    plt.colorbar(format="%+2.0f dB")
    plt.title(title)
    plt.show()

# 绘制各个分离后的音轨频谱图
plot_spectrogram(vocals, sr, "Vocals Spectrogram")
plot_spectrogram(drums, sr, "Drums Spectrogram")
plot_spectrogram(bass, sr, "Bass Spectrogram")
plot_spectrogram(other, sr, "Other Spectrogram")

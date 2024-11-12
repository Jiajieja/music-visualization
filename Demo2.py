import librosa
import librosa.display
import numpy as np
import plotly.graph_objects as go
import os
import IPython.display as ipd
import sklearn
import matplotlib.pyplot as plt
import pitchscapes.plotting as pt
from sklearn.preprocessing import minmax_scale
from musicflower.loader import load_file, load_corpus
from triangularmap import TMap
from musicflower.plotting import create_fig, plot_points, plot_all, plot_key_scape, key_colors
from musicflower.util import get_fourier_component, remap_to_xyz
import pitchscapes.plotting as pt

# 1. 读取音频文件并提取 Chroma 特征
audio_path = '/Users/xiaojie/Downloads/Color_Out_-_Host.mp3'
y, sr = librosa.load(audio_path, sr=None)
print(f"Audio Shape: {y.shape}, Sample Rate (KHz): {sr}")

# 获取音频时长
duration = librosa.get_duration(y=y, sr=sr)
print(f"Audio Duration (seconds): {duration}")

# 提取 Chroma 特征
chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=2048, n_fft= 512)
# 使用 Plotly 可视化 Chroma 特征
fig_chroma = go.Figure(data=go.Heatmap(z=chroma, x=np.linspace(0, duration, chroma.shape[1]), y=np.arange(12)))
fig_chroma.update_layout(
    title='Chroma Feature Visualization',
    xaxis_title='Time (seconds)',
    yaxis_title='Pitch Classes',
    showlegend=False
)
fig_chroma.show()

# 2. 使用三角形映射加载音乐文件（需要 MusicXML 文件）
# 设置 MusicXML 文件路径和分辨率
mxl_file_path = audio_path
# 使用较高分辨率
resolution = 50  

# 加载单个文件的三角形映射
scape = load_file(data=mxl_file_path, n=resolution)
print(f"Triangular Map Shape: {scape.shape}")

# 加载多个文件
corpus, files = load_corpus(data=[mxl_file_path, mxl_file_path], n=resolution)
print(f"Corpus Shape: {corpus.shape}")

# 3. 假设第二个片段是音高变换后的版本（旋转半音模拟不同片段）
corpus[1] = np.roll(corpus[1], shift=1, axis=-1)

# 4. 绘制键景观图 (Key Scape Plot)
plot_key_scape(corpus)

# 显示键图例
_ = pt.key_legend()
plt.show()

# 5. 生成颜色对应的三角形映射
colors = key_colors(corpus)
print(f"Color Map Shape: {colors.shape}")

# 6. 计算傅里叶变换的第 5 分量的幅值和相位
amplitude, phase = get_fourier_component(pcds=corpus, fourier_component=5)

# 7. 将傅里叶分量映射到 3D 空间
x, y, z = remap_to_xyz(amplitude=amplitude, phase=phase)

# 8. 绘制 3D 键景观图
plot_all(x=x, y=y, z=z, colors=colors)

# 9. 绘制时间轨迹的 3D 图
plot_all(x=x, y=y, z=z, colors=colors, do_plot_time_traces=True)

# ------------------------------------

# 11. 生成时间轨迹的 Key Scape Plot（2D 平面）
n = 20
x_2d = np.concatenate([np.arange(row + 1) - row / 2 for row in range(n)]) / (n - 1) + 0.5
y_plane = np.zeros_like(x_2d)
z = np.concatenate([np.ones(row + 1) * (n - 1 - row) for row in range(n)]) / (n - 1)
colors = np.random.uniform(0, 1, x_2d.shape + (3,))

# 创建 Key Scape Plot
fig_keyscape = create_fig(dark=False, axes_off=False)
fig_keyscape.add_trace(plot_points(x=x_2d, y=y_plane, z=z, colors=colors, marker_kwargs=dict(size=1)))
plot_all(x=x_2d, y=y_plane, z=z, colors=colors, fig=fig_keyscape, do_plot_time_traces=True, do_plot_points=False)
fig_keyscape.update_layout(scene_camera=dict(eye=dict(x=0.2, y=-2, z=0.2), center=dict(x=0, y=0, z=0)))
fig_keyscape.show()

# ---------------------------------
# 12. 在 3D 空间中绘制时间轨迹
x_3d = np.cos(10 * x_2d) * (x_2d + z ** 2)
y_3d = np.sin(10 * x_2d) * (x_2d + z ** 2)
z_3d = z
plot_all(x=x_3d, y=y_3d, z=z_3d, colors=colors, do_plot_time_traces=True)

# 13. 在 3D 空间中绘制带有时间轨迹的螺旋形
theta = np.linspace(0, 4 * np.pi, len(x_2d))  # 设置螺旋的旋转角度范围
radius = np.linspace(0.1, 2, len(x_2d))  # 从内向外扩展的半径
x_spiral = radius * np.cos(theta)  # 螺旋的 x 坐标
y_spiral = radius * np.sin(theta)  # 螺旋的 y 坐标
z_spiral = np.linspace(-2, 2, len(x_2d))  # z 坐标用于控制螺旋的高度变化

# 绘制螺旋形图
fig_spiral = create_fig(dark=False, axes_off=False)
plot_all(x=x_spiral, y=y_spiral, z=z_spiral, colors=colors, fig=fig_spiral, do_plot_time_traces=True, do_plot_points=True)
fig_spiral.update_layout(
    title="3D Spiral with Time Traces",
    scene_camera=dict(eye=dict(x=1.5, y=1.5, z=1.5), center=dict(x=0, y=0, z=0))
)
fig_spiral.show()

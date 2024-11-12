import librosa
import librosa.display
import numpy as np
import plotly.graph_objects as go
import os
import IPython.display as ipd
import sklearn
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale
from musicflower.loader import load_file
from triangularmap import TMap
from musicflower.plotting import create_fig, plot_points, plot_all

# 1. 读取音频文件并提取 Chroma 特征
audio_path = '/Users/xiaojie/Downloads/Color_Out_-_Host.mp3'
y, sr = librosa.load(audio_path, sr=None)
print(f"Audio Shape: {y.shape}, Sample Rate (KHz): {sr}")
# 获取音频时长
duration = librosa.get_duration(y=y, sr=sr)
print(f"Audio Duration (seconds): {duration}")

# 提取 Chroma 特征
chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=2048)
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
# 设置三角形映射分辨率
resolution = 10
mxl_file_path = '/Users/xiaojie/Downloads/Color_Out_-_Host.mp3'  # 请使用实际的 MusicXML 文件路径
scape = load_file(data=mxl_file_path, n=resolution)
print(f"Triangular Map Shape: {scape.shape}")
print(f"Expected Number of Elements: {resolution * (resolution + 1) / 2}")

# 3. 构建三角形映射对象
tmap = TMap(np.arange(resolution * (resolution + 1) // 2))
print("Triangular Map Representation:")
print(tmap.pretty())

# 4. 可视化三角形映射层级切片
# 顶层深度切片
print("Top Depth Slice:", tmap.dslice[0])
# 底层一级切片
print("Bottom Level Slice:", tmap.lslice[1])

# ---------------------------------------
# 5. 绘制频谱质心图（音高分布额外可视化）
hop_length = 512
spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
frames = range(len(spectral_centroids))
t = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)
# 归一化频谱质心数据
def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)
# 归一化处理频谱质心数据
normalized_spectral_centroids = normalize(spectral_centroids)
# 使用 Plotly 绘制频谱质心图
fig_spectral_centroid = go.Figure()
# 绘制原始音频的波形图
fig_spectral_centroid.add_trace(go.Scatter(
    x=np.linspace(0, duration, len(y)),
    y=y,
    mode='lines',
    name='Waveform',
    line=dict(color='blue', width=1),
    opacity=0.4
))
# 绘制频谱质心曲线
fig_spectral_centroid.add_trace(go.Scatter(
    x=t,
    y=normalized_spectral_centroids,
    mode='lines',
    name='Spectral Centroid',
    line=dict(color='red', width=2)
))
# 更新图表布局
fig_spectral_centroid.update_layout(
    title="Spectral Centroid",
    xaxis_title="Time (seconds)",
    yaxis_title="Normalized Spectral Centroid",
    template="plotly_dark",
    width=800,
    height=400
)
# 显示图表
fig_spectral_centroid.show()
# ------------------------------------

# 6. 生成时间轨迹的 Key Scape Plot（2D 平面）
n = 20
x = np.concatenate([np.arange(row + 1) - row / 2 for row in range(n)]) / (n - 1) + 0.5
y_plane = np.zeros_like(x)
z = np.concatenate([np.ones(row + 1) * (n - 1 - row) for row in range(n)]) / (n - 1)
colors = np.random.uniform(0, 1, x.shape + (3,))
# 创建 Key Scape Plot
fig_keyscape = create_fig(dark=False, axes_off=False)
fig_keyscape.add_trace(plot_points(x=x, y=y_plane, z=z, colors=colors, marker_kwargs=dict(size=1)))
plot_all(x=x, y=y_plane, z=z, colors=colors, fig=fig_keyscape, do_plot_time_traces=True, do_plot_points=False)
fig_keyscape.update_layout(scene_camera=dict(eye=dict(x=0.2, y=-2, z=0.2), center=dict(x=0, y=0, z=0)))
fig_keyscape.show()

# ---------------------------------
# 7. 在 3D 空间中绘制时间轨迹
x_3d = np.cos(10 * x) * (x + z ** 2)
y_3d = np.sin(10 * x) * (x + z ** 2)
z_3d = z
plot_all(x=x_3d, y=y_3d, z=z_3d, colors=colors, do_plot_time_traces=True)

# 8. 在 3D 空间中绘制带有时间轨迹的螺旋形
# 生成螺旋形坐标
theta = np.linspace(0, 4 * np.pi, len(x))  # 设置螺旋的旋转角度范围
radius = np.linspace(0.1, 2, len(x))  # 从内向外扩展的半径
x_3d = radius * np.cos(theta)  # 螺旋的 x 坐标
y_3d = radius * np.sin(theta)  # 螺旋的 y 坐标
z_3d = np.linspace(-2, 2, len(x))  # z 坐标用于控制螺旋的高度变化

# 绘制螺旋形图
fig_spiral = create_fig(dark=False, axes_off=False)
plot_all(x=x_3d, y=y_3d, z=z_3d, colors=colors, fig=fig_spiral, do_plot_time_traces=True, do_plot_points=True)
fig_spiral.update_layout(
    title="3D Spiral with Time Traces",
    scene_camera=dict(eye=dict(x=1.5, y=1.5, z=1.5), center=dict(x=0, y=0, z=0))
)
fig_spiral.show()







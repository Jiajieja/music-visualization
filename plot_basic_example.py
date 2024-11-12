"""
Basic Example
===========================

This is a brief walk through some basic MusicFlower functionality.
"""

# %%
# Loading a File
# ------------------------
#
# We are using a MusicXML file as it is small enough to ship with the documentation.
# It can be loaded using the :func:`~musicflower.loader.load_file` function
# 我们使用 MusicXML 文件，因为它足够小，可以随文档一起提供。
# 可使用 :func:`~musicflower.loader.load_file` 函数加载文件
from musicflower.loader import load_file

# path to file
file_path = 'Prelude_No._1_BWV_846_in_C_Major.mxl'
# split piece into this many equal-sized time intervals
#  分割成这么多大小相等的时间段
resolution = 50
# get pitch scape at specified resolution
# 以指定的分辨率获取 pitch scape
scape = load_file(data=file_path, n=resolution)
print(scape.shape)

# %%
# The result is an array with pitch-class distributions (PCDs), stored in a triangular map (see
# :doc:`plot_triangular_maps`). You can load multiple pieces using the :func:`~musicflower.loader.load_corpus`function
# 结果是一个包含音高类分布（PCD）的数组，存储在一个三角形图中（见:doc:`plot_triangular_maps`）。
# 您可以使用：func:`~musicflower.loader.load_corpus` 函数加载多个片段。
from musicflower.loader import load_corpus
corpus, files = load_corpus(data=[file_path, file_path], n=resolution)
print(corpus.shape)

# %%
# The resulting array has the different pieces as its first dimension. Since we have loaded the same pices twice, we
# will transpose one version by a semitone to fake a different second piece
# 得到的数组的第一个维度是不同的棋子。由于我们已经两次加载了相同的乐曲
# 所以我们将其中一个版本移调半音，以伪造不同的第二首乐曲
import numpy as np
corpus[1] = np.roll(corpus[1], shift=1, axis=-1)

# %%
# Key Scape Plots
# ---------------
# Pitch scapes can be visualised in traditional key scape plots, which are the basis for the 3D visualisation provided
# by the MusicFlower package.
# 音高景象可以在传统的音调景象图中可视化，这也是 MusicFlower 软件包提供的 3D 可视化的基础。
from musicflower.plotting import plot_key_scape
plot_key_scape(corpus)


# %%
# More functionality, such as a legend for the used colours, is available via the
# `PitchScapes <https://robert-lieck.github.io/pitchscapes/>`_ library
# 更多的功能，例如所用颜色的图例，可通过 `PitchScapes <https://robert-lieck.github.io/pitchscapes/>`_ 库获得
import pitchscapes.plotting as pt
_ = pt.key_legend()

# %%
# Colour
# ------
# The :func:`~musicflower.plotting.key_colors` function can be used to get the corresponding triangular map of colours
# as RGB in [0, 1]. These are computed by matching each PCD against templates for the 12 major and 12 minor keys and
# then interpolating between the respective colours shown in the legend above.
# 使用 :func:`~musicflower.plotting.key_colors` 函数可以得到相应的三角形颜色映射图。
# 以 [0, 1] 中的 RGB 表示。计算方法是将每个 PCD 与 12 个大调和 12 个小调的模板进行匹配，然后在各自的模板之间进行插值。
# 然后在上述图例中显示的相应颜色之间进行插值。
from musicflower.plotting import key_colors
colors = key_colors(corpus)
print(colors.shape)

# %%
# Mapping to Fourier Space
# ------------------------
# The discrete Fourier transform of a PCD contains musically relevant information. In particular, the 5th coefficient
# is strongest for distributions that correspond to diatonic scales and can therefore be associated to the tonality of a
# piece: its amplitude indicates how "strongly tonal" the piece is (e.g. atonal/12-tone pieces have a low amplitude);
# its phase maps to the circle of fifths. The :func:`~musicflower.util.get_fourier_component` function provides
# amplitudes and phases of an array of PCDs
# PCD 的离散傅立叶变换包含与音乐相关的信息。尤其是第 5 个系数对于与二声调音阶相对应的分布来说是最强的
# 因此可与乐曲的调性相关联：其振幅表示乐曲的 “强调性 ”程度（例如无调性/12 调性乐曲）。
# 其振幅表示乐曲的 “强调性 ”程度（例如，无调性/12 调性乐曲的振幅较低）
# 它的相位映射到五度圆。:func:`~musicflower.util.get_fourier_component` 函数提供了数组 PCD 的振幅和相位

from musicflower.util import get_fourier_component
amplitude, phase = get_fourier_component(pcds=corpus, fourier_component=5)


# %%
# Mapping to 3D Space
# -------------------
# The *amplitude* and *phase* of the Fourier component provide polar coordinates for each of the PCDs. (The phase also
# strongly correlates with the colours, even though they were computed using template matching, not Fourier components.)
# In the key scape plot above, we have two other dimensions for each PCD: *time* on the horizontal axis and the
# *duration* on the vertical axis (i.e. center and width of the respective section of the piece). Together, this can be
# used to map each PCD to a point in 3D space as follows.
# 傅立叶分量的*振幅*和*相位*为每个 PCD 提供了极坐标。
# (相位也与颜色密切相关，尽管它们是通过模板匹配而非傅立叶分量计算得出的）。
# 在上面的关键轮廓图中，每个 PCD 还有两个维度： 横轴上的*时间*和纵轴上的*持续时间*（即乐曲各部分的中心和宽度）。
# 这两个维度可以用于将每个 PCD 映射到三维空间中的一个点，如下所示。
#
# We use spherical or cylindrical coordinates with the *phase* as the azimuthal/horizontal angle, the *duration*
# for the radial component, and the *amplitude* for the vertical component/angle (cylinder/sphere). This is done
# by the :func:`~musicflower.util.remap_to_xyz` function, which also provides some additional tweaks (see its
# documentation for details). Note that *time* is not explicitly represented anymore, but can be included through
# interactive animations (see below).
# 我们使用球面或圆柱坐标，*相位*为方位角/水平角，*持续时间*#为径向分量，*振幅*为垂直分量/角度（圆柱/球面）表示径向分量，
# *振幅*表示垂直分量/角度（圆柱/球体）。这可以通过由 :func:`~musicflower.util.remap_too_xyz` 函数完成，
# 该函数还提供了一些额外的调整（详见其文档了解详情）。
# 请注意，*时间* 不再明确表示，但可以通过互动动画（见下文）。

from musicflower.util import remap_to_xyz
x, y, z = remap_to_xyz(amplitude=amplitude, phase=phase)

# %%
# 3D Plot
# ------------------------
# This can be visualised in a 3D plot as follows
# 这可以用三维图直观地表示出来，如下所示

from musicflower.plotting import plot_all
plot_all(x=x, y=y, z=z, colors=colors)


# %%
# Time as Animation
# -------------------------
# We can add the time dimension using an interactive slider and/or animation. The slider represents time in a normalised
# [0, 1] interval over the piece duration. When moving the slider, a line is drawn from the top, through the triangular
# map to the point at the bottom corresponding to the current slider position. In the normal key scape plot from above,
# this would simply be a straight line from the top down to the bottom; in the 3D plot it winds through tonal space
# (also see :doc:`plot_time_traces`).
# 时间是动画
# -------------------------
# 我们可以使用互动滑块和/或动画来添加时间维度。滑动条表示的时间是[0, 1] 的时间间隔。
# 移动滑块时，一条线会从顶部通过三角形图，到达底部与当前滑块位置相对应的点。
# 在上面的普通键景图中、这只是一条从上到下的直线；
# 而在三维图中，它蜿蜒穿过音调空间
# 另请参阅 :doc:`plot_time_traces`）。



plot_all(x=x, y=y, z=z, colors=colors, do_plot_time_traces=True)

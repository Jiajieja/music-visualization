import os
import tempfile
import base64
import subprocess
import librosa
import librosa.beat
import numpy as np
import pickle
import json
import soundfile as sf
from dash.dependencies import Input, Output, State, ALL
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, no_update, dash_table
from dash.exceptions import PreventUpdate
from pitchtypes import EnharmonicPitchClass
from pydub import AudioSegment

SPLEETER_ENV = "spleeter_env"
rad_to_deg = 180 / np.pi

def run_command(env_name, command):
    activate_script = f". {env_name}/bin/activate"
    full_command = f"{activate_script} && {command}; deactivate"
    try:
        subprocess.run(full_command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running command: {e}")
        raise

def run_spleeter(input_file, output_dir):
    print("Running Spleeter...")
    spleeter_command = f"spleeter separate -o {output_dir} -p spleeter:4stems {input_file}"
    run_command(SPLEETER_ENV, spleeter_command)
    print("Spleeter completed.")

def load_separated_tracks(temp_dir, subdir_name):
    tracks = {}
    subdir = os.path.join(temp_dir, subdir_name)
    expected_files = ["vocals.wav", "drums.wav", "bass.wav", "other.wav"]

    for file in expected_files:
        file_path = os.path.join(subdir, file)
        if not os.path.exists(file_path):
            print(f"警告：{file_path} 不存在，跳过...")
            continue

        y, sr = librosa.load(file_path, sr=None)
        track_name = file.split(".")[0].capitalize()
        
        # 对人声轨道进行静音检测
        if track_name == "Vocals":
            # 计算RMS（均方根）能量，检测非静音帧
            rms = librosa.feature.rms(y=y)[0]
            threshold = 0.01  # 根据音频调整此阈值（RMS能量）
            non_silent_frames = rms > threshold
            # 创建非静音区域的掩码
            times = librosa.times_like(rms, sr=sr)
            non_silent_mask = non_silent_frames
            
            # 仅为非静音区域计算色度特征
            chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12)
            # 将静音帧的色度值设为零
            chroma[:, ~non_silent_frames] = 0
        else:
            chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12)
        
        times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr)
        
        temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        sf.write(temp_wav.name, y, sr)
        with open(temp_wav.name, "rb") as f:
            audio_base64 = base64.b64encode(f.read()).decode('utf-8')
        os.unlink(temp_wav.name)

        tracks[track_name] = {
            "audio": y,
            "sr": sr,
            "duration": librosa.get_duration(y=y, sr=sr),
            "chroma": {"chroma": chroma, "time": times},
            "audio_base64": f"data:audio/wav;base64,{audio_base64}"
        }
        print(f"已加载 {track_name} 从 {file_path}")
    return tracks

def compute_chroma_features(tracks, n_frames=200):
    features_list = []
    for track_name, track_data in tracks.items():
        y = track_data["audio"]
        sr = track_data["sr"]
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12)
        if chroma.shape[1] > n_frames:
            chroma = chroma[:, :n_frames]
        elif chroma.shape[1] < n_frames:
            chroma = np.pad(chroma, ((0, 0), (0, n_frames - chroma.shape[1])), mode='constant')
        features_list.append(chroma)
    features = np.array(features_list)
    features = np.transpose(features, (1, 2, 0))
    return features

def compute_tempo_with_beats(tracks, temp_dir, filename):
    multi_track_tempo = {}
    for track_name, track_data in tracks.items():
        y = track_data["audio"]
        sr = track_data["sr"]
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        times = librosa.times_like(onset_env, sr=sr)
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr, onset_envelope=onset_env)
        
        beat_times = librosa.frames_to_time(beats, sr=sr)
        strong_beats = beats[::4]
        weak_beats = [b for b in beats if b not in strong_beats]
        
        onset_waveform = onset_env / (np.max(onset_env) + 1e-10)
        strong_waveform = np.zeros_like(times)
        weak_waveform = np.zeros_like(times)
        
        for beat_time in beat_times:
            idx = np.argmin(np.abs(times - beat_time))
            if beat_time in librosa.frames_to_time(strong_beats, sr=sr):
                strong_waveform[idx] = 1
            else:
                weak_waveform[idx] = 1
        
        temp_file_path = os.path.join(temp_dir, f"tempo_data_{filename}_{track_name}.pkl")
        with open(temp_file_path, 'wb') as f:
            pickle.dump({
                'times': times,
                'onset_waveform': onset_waveform,
                'strong_waveform': strong_waveform,
                'weak_waveform': weak_waveform,
                'overall_tempo': tempo
            }, f)
        
        multi_track_tempo[track_name] = temp_file_path
    return multi_track_tempo

def load_tempo_with_beats(temp_file_path):
    if not os.path.exists(temp_file_path):
        print(f"Error: Tempo file {temp_file_path} does not exist.")
        return None, None, None, None, None
    try:
        with open(temp_file_path, 'rb') as f:
            data = pickle.load(f)
        print(f"Loaded tempo data from {temp_file_path}: {list(data.keys())}")
        return (data['times'], data['onset_waveform'], data['strong_waveform'], 
                data['weak_waveform'], data['overall_tempo'])
    except Exception as e:
        print(f"Error loading tempo file {temp_file_path}: {str(e)}")
        return None, None, None, None, None

def extract_piano_roll(y, sr, hop_length=512):
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr, hop_length=hop_length)
    times = librosa.times_like(pitches, sr=sr, hop_length=hop_length)

    onsets = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop_length)
    onset_times = librosa.frames_to_time(onsets, sr=sr, hop_length=hop_length)

    piano_roll = []
    for i, onset in enumerate(onsets[:-1]):
        start_time = float(onset_times[i])
        end_time = float(onset_times[i + 1] if i + 1 < len(onsets) else times[-1])
        duration = end_time - start_time

        segment_start_idx = onset
        segment_end_idx = onsets[i + 1] if i + 1 < len(onsets) else pitches.shape[1]
        segment_pitches = pitches[:, segment_start_idx:segment_end_idx]
        segment_magnitudes = magnitudes[:, segment_start_idx:segment_end_idx]

        max_magnitude_idx = np.argmax(np.mean(segment_magnitudes, axis=1))
        pitch = pitches[max_magnitude_idx, segment_start_idx]
        if pitch > 0:
            midi_pitch = int(round(librosa.hz_to_midi(pitch)))
            pitch_class = midi_pitch % 12  # 转换为音高类别索引（0-11）
            piano_roll.append({
                "start": start_time,
                "end": end_time,
                "pitch_class": pitch_class,  # 存储音高类别而不是 MIDI 音高
                "duration": duration
            })
    return piano_roll

def compute_piano_roll_data(tracks):
    piano_roll_data = {}
    for track_name, track_data in tracks.items():
        y = track_data["audio"]
        sr = track_data["sr"]
        piano_roll = extract_piano_roll(y, sr)
        piano_roll_data[track_name] = piano_roll
    return piano_roll_data

def piano_roll_visualizer(piano_roll_data, position, duration, window_sec=5, selected_tracks=None):
    if not piano_roll_data:
        return go.Figure()

    current_time = position * duration
    start_time = max(0, current_time - window_sec / 2)
    end_time = min(duration, current_time + window_sec / 2)

    fig = go.Figure()
    colors = {"Vocals": '#DC143C', "Drums": '#1E90FF', "Bass": '#008000', "Other": '#9400D3'}
    track_names = list(piano_roll_data.keys())
    
    if selected_tracks is None or not selected_tracks:
        selected_tracks = track_names

    for track_name in selected_tracks:
        if track_name not in piano_roll_data:
            continue
        notes = piano_roll_data[track_name]
        for note in notes:
            start = note["start"]
            end = note["end"]
            pitch_class = note["pitch_class"]  # 使用音高类别
            if end >= start_time and start <= end_time:
                fig.add_shape(
                    type="rect",
                    x0=start, x1=end,
                    y0=pitch_class - 0.4, y1=pitch_class + 0.4,  # 调整矩形高度以适应音高类别
                    fillcolor=colors.get(track_name, '#FFCC99'),
                    opacity=0.7,
                    line=dict(width=0),
                    name=track_name
                )

    fig.add_vline(x=current_time, line=dict(color="red", dash="dash"))

    for track_name in selected_tracks:
        if track_name in colors:
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=10, color=colors[track_name]),
                legendgroup=track_name,
                showlegend=True,
                name=track_name
            ))

    fig.update_layout(
        title="Piano Roll Visualization (Pitch Classes)",
        xaxis_title="Time (s)",
        yaxis_title="Pitch Class",
        template="plotly_white",
        height=600,
        xaxis_range=[start_time, end_time],
        yaxis=dict(
            range=[-0.5, 11.5],  # 音高类别范围 0-11
            tickvals=list(range(12)),
            ticktext=PITCH_CLASSES  # 使用音高类别标签
        ),
        showlegend=True,
        legend=dict(title="Tracks", orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    return fig

PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
CHORD_INTERVALS = {"major": [0, 4, 7], "minor": [0, 3, 7], "aug": [0, 4, 8], "dim": [0, 3, 6]}

CONSONANT_CHORDS = set()
DISSONANT_CHORDS = set()
CHORD_TEMPLATES = {}
for root_idx, root in enumerate(PITCH_CLASSES):
    for chord_type, intervals in CHORD_INTERVALS.items():
        template = np.zeros(12)
        for interval in intervals:
            note_idx = (root_idx + interval) % 12
            template[note_idx] = 1
        chord_name = root + ("" if chord_type == "major" else "m" if chord_type == "minor" else chord_type)
        CHORD_TEMPLATES[chord_name] = template
        if chord_type in ["major", "minor"]:
            CONSONANT_CHORDS.add(chord_name)
        else:
            DISSONANT_CHORDS.add(chord_name)

def detect_chord(chroma_vector):
    if np.all(chroma_vector == 0):
        return "None"
    norm_vec = chroma_vector / (np.linalg.norm(chroma_vector) + 1e-10)
    max_similarity = -1
    detected_chord = None
    for chord_name, template in CHORD_TEMPLATES.items():
        similarity = np.dot(norm_vec, template)
        if similarity > max_similarity:
            max_similarity = similarity
            detected_chord = chord_name
    return detected_chord if max_similarity > 0.5 else "None"

def is_consonant(chord):
    return chord in CONSONANT_CHORDS

def detect_chord_segments(tracks, segment_duration=2.0):
    chord_segments = {}
    for track_name, track_data in tracks.items():
        chroma = track_data["chroma"]["chroma"]
        times = track_data["chroma"]["time"]
        total_duration = times[-1]
        segments = []
        current_time = 0.0
        previous_chord = None

        while current_time < total_duration:
            start_frame = np.searchsorted(times, current_time, side='left')
            end_time = min(current_time + segment_duration, total_duration)
            end_frame = np.searchsorted(times, end_time, side='right')

            if start_frame >= chroma.shape[1]:
                break

            segment_chroma = np.mean(chroma[:, start_frame:end_frame], axis=1)
            chord = detect_chord(segment_chroma)

            if chord != "None" and chord != previous_chord:
                segments.append({
                    "chord": chord,
                    "start": current_time,
                    "duration": min(segment_duration, total_duration - current_time),
                    "pitch_shift": 0
                })
                previous_chord = chord

            current_time += segment_duration

        chord_segments[track_name] = segments
    return chord_segments

def chord_statistics(features):
    if features is None or features.shape[0] != 12:
        return {"Consonant": 0, "Dissonant": 0, "None": 0}

    n_frames = features.shape[1]
    consonant_count = 0
    dissonant_count = 0
    none_count = 0

    for frame_idx in range(n_frames):
        chroma_vector = np.sum(features[:, frame_idx, :], axis=1)
        chord_name = detect_chord(chroma_vector)
        if chord_name == "None":
            none_count += 1
        elif is_consonant(chord_name):
            consonant_count += 1
        else:
            dissonant_count += 1

    return {"Consonant": consonant_count, "Dissonant": dissonant_count, "None": none_count}

def chord_stat_bar_figure(chord_stat):
    categories = ["Consonant", "Dissonant", "None"]
    values = [chord_stat["Consonant"], chord_stat["Dissonant"], chord_stat["None"]]
    colors = ["green", "red", "gray"]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=categories, y=values, marker_color=colors))
    fig.update_layout(
        title="Real-time Chord Statistics",
        xaxis_title="Chord Category",
        yaxis_title="Occurrences",
        template="plotly_white",
        height=400
    )
    return fig

def multi_chroma_visualizer(*, features, position, tracks, **kwargs):
    if not tracks:
        return go.Figure()

    rows, cols = 2, 2
    fig = make_subplots(
        rows=rows, cols=cols,
        shared_xaxes=True,
        subplot_titles=[f"{track_name} Chroma" for track_name in tracks.keys()],
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )

    colors = {
        "Vocals": '#DC143C',
        "Drums": '#1E90FF',
        "Bass": '#008000',
        "Other": '#9400D3'
    }

    track_list = list(tracks.items())

    for idx, (track_name, track_data) in enumerate(track_list):
        if "chroma" not in track_data:
            continue

        chroma_data = track_data["chroma"]
        time_axis = chroma_data["time"]
        chroma_matrix = chroma_data["chroma"]

        current_time = position * track_data["duration"]
        window_sec = 5
        mask = (time_axis >= current_time - window_sec / 2) & (time_axis <= current_time + window_sec / 2)

        row = (idx // cols) + 1
        col = (idx % cols) + 1

        track_color = colors.get(track_name, '#FFCC99')
        custom_colorscale = [
            [0, 'rgba(0,0,0,0)'],
            [1, track_color]
        ]

        fig.add_trace(
            go.Heatmap(
                z=chroma_matrix[:, mask],
                x=time_axis[mask],
                y=PITCH_CLASSES,
                colorscale=custom_colorscale,
                zmin=0, zmax=1,
                showscale=False
            ),
            row=row, col=col
        )

    fig.update_layout(
        title="Multi-Track Chroma",
        xaxis_title="Time (s)",
        yaxis_title="Pitch Class",
        template="plotly_white",
        height=600,
        width=800,
        margin=dict(l=20, r=20, t=50, b=50),
        autosize=False
    )
    return fig

def position_idx(position, n_frames):
    return max(0, min(n_frames - 1, int(round(position*(n_frames-1)))))

def check_chroma_features(features):
    if not isinstance(features, np.ndarray) or features.ndim != 3:
        raise ValueError("Chroma features must be a (12, n_frames, n_tracks) array!")
    return features

def Chroma_bar_visualiser(features, position):
    features = check_chroma_features(features)
    n_frames = features.shape[1]
    idx = position_idx(position, n_frames)
    data_vector = np.sum(features[:, idx, :], axis=1)

    fig = go.Figure([go.Bar(
        x=[str(EnharmonicPitchClass(i)) for i in range(12)],
        y=data_vector
    )])
    fig.update_layout(
        title="Chroma Bars",
        template="plotly_white",
        xaxis_title="Pitch Class",
        yaxis_title="Intensity",
        height=400,
        uirevision="chroma"
    )
    fig.update_yaxes(range=[0, max(1, data_vector.max())])
    return fig

def chord_visualizer(features, position, duration, n_frames=200):
    if features is None or features.shape[1] < n_frames:
        return go.Figure()

    idx = position_idx(position, n_frames)
    current_time = position * duration
    chroma_vector = np.sum(features[:, idx, :], axis=1)
    chroma_vector = chroma_vector / (np.linalg.norm(chroma_vector) + 1e-10)

    chord = detect_chord(chroma_vector)
    if chord == "None":
        color = "gray"
        label = "No chord"
    else:
        is_cons = is_consonant(chord)
        color = "green" if is_cons else "red"
        label = f"{chord} ({'Consonant' if is_cons else 'Dissonant'})"

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Current Chord", "Chord Progression"),
        vertical_spacing=0.15,
        row_heights=[0.3, 0.7]
    )

    text_html = (
        "<span style='font-size:16px; color:#1f2c51;'>Current Chord</span>"
        "<br>"
        f"<span style='font-size:20px; font-weight:bold; color:{color};'>{label}</span>"
    )
    fig.add_trace(go.Scatter(
        x=[current_time],
        y=[1.2],
        mode="text",
        text=[text_html],
        textposition="middle center",
        hoverinfo="none",
        showlegend=False
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=PITCH_CLASSES,
        y=chroma_vector,
        opacity=0.6,
        name="Chroma Intensity"
    ), row=1, col=1)

    window_frames = 20
    idx_start = max(0, idx - window_frames // 2)
    idx_end = min(n_frames, idx + window_frames // 2)
    time_axis = np.linspace(
        current_time - (window_frames/2) * duration / n_frames,
        current_time + (window_frames/2) * duration / n_frames,
        idx_end - idx_start
    )
    chord_labels = []
    chord_colors = []
    for i in range(idx_start, idx_end):
        vec = np.sum(features[:, i, :], axis=1)
        vec = vec / (np.linalg.norm(vec) + 1e-10)
        chd = detect_chord(vec)
        if chd == "None":
            chord_labels.append("No chord")
            chord_colors.append("gray")
        else:
            chord_labels.append(chd)
            chord_colors.append("green" if is_consonant(chd) else "red")

    fig.add_trace(go.Scatter(
        x=time_axis,
        y=[1]*len(time_axis),
        mode="text",
        text=chord_labels,
        textposition="top center",
        textfont=dict(size=12, color=chord_colors),
        hoverinfo="none",
        showlegend=False
    ), row=2, col=1)

    fig.add_vline(x=current_time, line=dict(color="black", dash="dash"), row=2, col=1)

    fig.update_layout(
        title="Chord Detection and Progression",
        height=600,
        template="plotly_white",
        uirevision="chord"
    )
    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_yaxes(showgrid=False, showticklabels=False, row=1, col=1)
    fig.update_yaxes(showgrid=False, showticklabels=False, row=2, col=1, range=[0, 2])

    return fig

def create_tempo_with_beats_figure(multi_track_tempo_paths, position, duration):
    rows, cols = 2, 2
    fig = make_subplots(
        rows=rows, cols=cols,
        shared_xaxes=True,
        subplot_titles=["Vocals Tempo", "Drums Tempo", "Bass Tempo", "Other Tempo"],
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )

    colors = {"Vocals": '#DC143C', "Drums": '#1E90FF', "Bass": '#008000', "Other": '#9400D3'}
    current_time = position * duration
    window_sec = 5
    for idx, track_name in enumerate(['Vocals', 'Drums', 'Bass', 'Other']):
        tempo_file_path = multi_track_tempo_paths.get(track_name)
        if not tempo_file_path:
            continue

        times, onset_waveform, strong_waveform, weak_waveform, overall_tempo = load_tempo_with_beats(tempo_file_path)
        if times is None:
            continue

        mask = (times >= current_time - window_sec/2) & (times <= current_time + window_sec/2)
        
        row = (idx // cols) + 1
        col = (idx % cols) + 1

        fig.add_trace(go.Scatter(
            x=times[mask],
            y=onset_waveform[mask],
            mode='lines',
            line=dict(color=colors[track_name], width=1),
            name=f"{track_name} Onset Strength",
            showlegend=(idx == 0)
        ), row=row, col=col)

        strong_times = times[mask][strong_waveform[mask] > 0]
        strong_values = onset_waveform[mask][strong_waveform[mask] > 0]
        fig.add_trace(go.Scatter(
            x=strong_times,
            y=strong_values,
            mode='markers',
            marker=dict(size=8, color=colors[track_name], symbol='triangle-up'),
            name=f"{track_name} Strong Beats",
            showlegend=(idx == 0)
        ), row=row, col=col)

        weak_times = times[mask][weak_waveform[mask] > 0]
        weak_values = onset_waveform[mask][weak_waveform[mask] > 0]
        fig.add_trace(go.Scatter(
            x=weak_times,
            y=weak_values,
            mode='markers',
            marker=dict(size=6, color=colors[track_name], symbol='circle', opacity=0.5),
            name=f"{track_name} Weak Beats",
            showlegend=(idx == 0)
        ), row=row, col=col)

        fig.add_vline(x=current_time, line=dict(color="red", dash="dash"), row=row, col=col)

    fig.update_layout(
        title="Multi-Track Tempo Waveforms",
        height=600,
        template="plotly_white",
        showlegend=True,
        legend=dict(title="Tracks", orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    fig.update_xaxes(title_text="Time (s)")
    fig.update_yaxes(title_text="Onset Strength", range=[0, 1])

    return fig

class CustomWebApp:
    def __init__(self, verbose=False):
        self.app = Dash(__name__, suppress_callback_exceptions=True)
        self.verbose = verbose
        self.tracks = {}
        self.chroma_features = None
        self.piano_roll_data = {}
        self.chord_segments = {}
        self.duration = 0
        self.audio_src = None
        self.tempo_file_paths = {}
        self.is_playing = False
        self.chord_sequence = []
        self.chord_segments_file = "chord_segments.json"
        self.track_audio_store = {}

    def _audio_element(self, audio_file, audio_src=None):
        src = {} if audio_src is None else dict(src=audio_src, preload="auto")
        return html.Div(id='_sound-file-display', children=html.Div([
            html.P(audio_file if audio_file else "No audio uploaded"),
            html.Audio(**src, controls=True, id='_audio-controls', autoPlay=False, style={'width': '100%'}),
        ]))

    def _track_audio_element(self, track_name, audio_src):
        return html.Audio(
            id=f'track-audio-{track_name.lower()}',
            src=audio_src,
            controls=True,
            autoPlay=False,
            style={'width': '100%', 'display': 'block'}
        )

    def run(self, *args, **kwargs):
        self.app.run_server(*args, **kwargs)

    def cleanup(self):
        for tempo_file_path in self.tempo_file_paths.values():
            if os.path.exists(tempo_file_path):
                os.remove(tempo_file_path)
                if self.verbose:
                    print(f"Cleaned up tempo file: {tempo_file_path}")
        if os.path.exists(self.chord_segments_file):
            os.remove(self.chord_segments_file)
            if self.verbose:
                print(f"Cleaned up chord segments file: {self.chord_segments_file}")
        sequence_file = "chord_sequence.mp3"
        if os.path.exists(sequence_file):
            os.remove(sequence_file)
            if self.verbose:
                print(f"Cleaned up sequence file: {sequence_file}")

    def save_chord_segments(self, chord_segments):
        with open(self.chord_segments_file, 'w') as f:
            json.dump(chord_segments, f, indent=4)
        if self.verbose:
            print(f"Chord segments saved to {self.chord_segments_file}")

    def load_chord_segments(self):
        if os.path.exists(self.chord_segments_file):
            with open(self.chord_segments_file, 'r') as f:
                return json.load(f)
        return {}

    def process_uploaded_file(self, contents, filename):
        if contents is None:
            raise PreventUpdate

        self.cleanup()

        temp_dir = tempfile.gettempdir()
        temp_audio_path = os.path.join(temp_dir, filename)
        with open(temp_audio_path, 'wb') as f:
            f.write(base64.b64decode(contents.split(',')[1]))

        try:
            run_spleeter(temp_audio_path, temp_dir)
        except Exception as e:
            return (
                f"Error during Spleeter: {str(e)}",
                no_update, no_update, [], [], 0, no_update, no_update, {}, [], {}, [], {}
            )

        subdir_name = os.path.splitext(filename)[0]
        splitted_tracks = load_separated_tracks(temp_dir, subdir_name)
        if not splitted_tracks:
            return (
                "No tracks loaded after Spleeter.",
                no_update, no_update, [], [], 0, no_update, no_update, {}, [], {}, [], {}
            )

        chroma_features = compute_chroma_features(splitted_tracks, n_frames=200)
        duration = max(t["duration"] for t in splitted_tracks.values())

        self.tempo_file_paths = compute_tempo_with_beats(splitted_tracks, temp_dir, os.path.splitext(filename)[0])

        self.tracks = splitted_tracks
        self.chroma_features = chroma_features
        self.piano_roll_data = compute_piano_roll_data(splitted_tracks)
        self.chord_segments = detect_chord_segments(splitted_tracks, segment_duration=2.0)
        self.duration = duration
        self.audio_src = contents

        self.save_chord_segments(self.chord_segments)

        audio_element = self._audio_element(audio_file=filename, audio_src=contents)
        tracks_data = {
            "track_names": list(splitted_tracks.keys()),
            "piano_roll": {
                k: [
                    {
                        "start": float(n["start"]),
                        "end": float(n["end"]),
                        "pitch_class": n["pitch_class"],
                        "duration": float(n["duration"])
                    } for n in v
                ] for k, v in self.piano_roll_data.items()
            }
        }

        print("Generated chord_segments:", self.chord_segments)
        print("Precomputed piano_roll_data:", {k: len(v) for k, v in self.piano_roll_data.items()})

        track_options = [{'label': track_name, 'value': track_name} for track_name in self.chord_segments.keys()]
        if not track_options:
            print("Warning: No track options generated for chord-track-selector")

        track_audio_data = {track_name: track_data["audio_base64"] for track_name, track_data in self.tracks.items()}
        original_track_audio_data = track_audio_data.copy()

        track_audio_elements = [
            self._track_audio_element(track_name, audio_src)
            for track_name, audio_src in track_audio_data.items()
        ]

        return (
            "Audio processed successfully.",
            audio_element,
            contents,
            tracks_data,
            chroma_features.tolist() if chroma_features is not None else None,
            duration,
            0,
            self.tempo_file_paths,
            self.chord_segments,
            track_options,
            track_audio_data,
            track_audio_elements,
            original_track_audio_data
        )

    def chord_sequence_visualizer(self, chord_sequence, duration):
        if not chord_sequence:
            return go.Figure()

        track_names = sorted(set(seg["track"] for seg in chord_sequence))
        if not track_names:
            return go.Figure()

        track_to_y = {track: i for i, track in enumerate(track_names)}
        colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99']

        fig = go.Figure()

        for segment in chord_sequence:
            track = segment["track"]
            start = float(segment["start"])
            duration_seg = float(segment["duration"])
            end = start + duration_seg
            chord = segment["chord"]
            pitch_shift = segment.get("pitch_shift", 0)
            y_pos = track_to_y[track]

            fig.add_shape(
                type="rect",
                x0=start, x1=end,
                y0=y_pos - 0.4, y1=y_pos + 0.4,
                fillcolor=colors[track_to_y[track] % len(colors)],
                opacity=0.6,
                line=dict(width=0),
                name=track
            )

            fig.add_trace(go.Scatter(
                x=[(start + end) / 2],
                y=[y_pos],
                text=[f"{chord} ({pitch_shift:+d})"],
                mode="text",
                textposition="middle center",
                textfont=dict(size=12, color="black"),
                showlegend=False,
                hoverinfo="text",
                hovertext=f"{track}: {chord} (Pitch: {pitch_shift:+d}) ({start:.2f}s - {end:.2f}s)"
            ))

        for track, y_pos in track_to_y.items():
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=10, color=colors[y_pos % len(colors)]),
                legendgroup=track,
                showlegend=True,
                name=track
            ))

        fig.update_layout(
            title="Edited Chord Sequence Visualization",
            xaxis_title="Time (s)",
            yaxis_title="Tracks",
            yaxis=dict(
                tickvals=list(track_to_y.values()),
                ticktext=list(track_to_y.keys()),
                range=[-0.5, len(track_names) - 0.5]
            ),
            xaxis_range=[0, duration if duration else max(float(seg["start"]) + float(seg["duration"]) for seg in chord_sequence)],
            height=200 + 50 * len(track_names),
            template="plotly_white",
            showlegend=True,
            legend=dict(title="Tracks", orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
        )

        print(f"Rendering chord sequence visualization: {[{k: v for k, v in seg.items() if k in ['chord', 'start', 'duration', 'track']} for seg in chord_sequence]}")
        return fig

    def setup_app(self):
        self.app.layout = html.Div([
            # Main container
            html.Div([
                # Upload Area
                dcc.Upload(
                    id='upload-audio',
                    children=html.Div(
                        ['Drag and Drop or Click to Select File'],
                        style={
                            'color': '#666',
                            'fontSize': '18px',
                            'fontWeight': '500',
                            'lineHeight': '60px'
                        }
                    ),
                    style={
                        'width': '100%',
                        'height': '60px',
                        'border': '2px dashed #ccc',
                        'borderRadius': '8px',
                        'textAlign': 'center',
                        'margin': '10px 0',
                        'backgroundColor': '#ffffff',
                        'cursor': 'pointer',
                        'transition': 'all 0.3s ease'
                    },
                    multiple=False
                ),
                # Status Messages
                html.Div(
                    id='upload-status',
                    style={
                        'color': '#555',
                        'fontSize': '14px',
                        'textAlign': 'center',
                        'margin': '10px 0'
                    }
                ),
                html.Div(
                    id='audio-load-status',
                    children="Loading audio...",
                    style={
                        'color': '#555',
                        'fontSize': '14px',
                        'textAlign': 'center',
                        'margin': '10px 0'
                    }
                ),
                # Audio Element
                self._audio_element(audio_file=None),
                # Play/Pause Button
                html.Button(
                    "Play/Pause",
                    id='play-pause-button',
                    n_clicks=0,
                    style={
                        'backgroundColor': '#007bff',
                        'color': '#ffffff',
                        'border': 'none',
                        'borderRadius': '6px',
                        'padding': '10px 20px',
                        'fontSize': '14px',
                        'fontWeight': '500',
                        'cursor': 'pointer',
                        'margin': '10px',
                        'transition': 'background-color 0.3s ease'
                    }
                ),
                html.Div(
                    id='playback-status',
                    children="Ready",
                    style={
                        'color': '#555',
                        'fontSize': '14px',
                        'textAlign': 'center',
                        'margin': '10px 0'
                    }
                ),
                # Tabs
                dcc.Tabs(
                    id="visualisation-tabs",
                    value='tab-intro',
                    style={
                        'marginTop': '20px'
                    },
                    children=[
                        # Intro Tab
                        dcc.Tab(
                            label='Intro',
                            value='tab-intro',
                            children=[
                                html.H3(
                                    "Welcome to the Music Analysis Webapp",
                                    style={
                                        'fontSize': '25px',
                                        'textAlign': 'center',
                                        'margin': '20px 0',
                                        'color': '#1f2c51',
                                        'fontWeight': '600'
                                    }
                                ),
                                html.P(
                                    "This webapp is designed for beginner users to explore and analyze music in an interactive and intuitive way. Whether you're a music enthusiast, a student, or a budding producer, this tool helps you break down audio files into their core components and visualize their musical structure. By uploading an audio file, you can separate it into individual tracks (vocals, drums, bass, and other instruments) and explore various aspects such as harmony, rhythm, and melody through dynamic visualizations. You can also edit chord sequences to create custom remixes, making it a powerful tool for both learning and creativity.",
                                    style={
                                        'fontSize': '16px',
                                        'margin': '20px',
                                        'color': '#555'
                                    }
                                ),
                                html.H4(
                                    "What You Will See",
                                    style={
                                        'fontSize': '20px',
                                        'margin': '20px 20px 10px',
                                        'color': '#34495e',
                                        'fontWeight': '500'
                                    }
                                ),
                                html.P(
                                    "This introduction page provides an overview of the webapp’s features. After uploading an audio file, you’ll see a control panel with playback controls, a slider for navigating the audio, and multiple tabs showcasing different visualizations (e.g., Chroma Bars, Chord Detection, Piano Roll). Each tab includes detailed explanations to guide you through the analysis process.",
                                    style={
                                        'fontSize': '16px',
                                        'margin': '10px 20px',
                                        'color': '#555'
                                    }
                                ),
                                html.H4(
                                    "Why It Matters",
                                    style={
                                        'fontSize': '20px',
                                        'margin': '20px 20px 10px',
                                        'color': '#34495e',
                                        'fontWeight': '500'
                                    }
                                ),
                                html.P(
                                    "Understanding the components of music—such as pitch, chords, and rhythm—can deepen your appreciation and enhance your skills in music analysis or production. This webapp makes complex audio analysis accessible by providing visual tools and interactive features, allowing you to explore music in a hands-on way without needing advanced technical knowledge.",
                                    style={
                                        'fontSize': '16px',
                                        'margin': '10px 20px',
                                        'color': '#555'
                                    }
                                ),
                                html.H4(
                                    "How to Use",
                                    style={
                                        'fontSize': '20px',
                                        'margin': '20px 20px 10px',
                                        'color': '#34495e',
                                        'fontWeight': '500'
                                    }
                                ),
                                html.P(
                                    "1. **Upload an Audio File**: Click the 'Drag and Drop or Click to Select File' area at the top to upload an MP3 or WAV file. The webapp will process the file and separate it into vocals, drums, bass, and other tracks. The separation process will take some time, please be patient.\n"
                                    "2. **Navigate the Tabs**: Use the tabs (e.g., Chroma Bars, Chord Detection) to explore different visualizations. Each tab provides insights into specific musical elements, such as pitch distribution or chord progressions.\n"
                                    "3. **Interact with Visualizations**: Use the playback slider to move through the audio and see how the visualizations update in real-time. Adjust settings like track selection in tabs like Piano Roll or Chord Segments.\n"
                                    "4. **Edit and Create**: In the Chord Segments tab, modify chord timings or pitch shifts, build a custom sequence, and export it as an audio file for remixing or further analysis.\n"
                                    "Now start by uploading a song you love and exploring the tabs to discover its musical structure!",
                                    style={
                                        'fontSize': '16px',
                                        'margin': '10px 20px',
                                        'color': '#555',
                                        'whiteSpace': 'pre-line'
                                    }
                                )
                            ],
                            style={
                                'backgroundColor': '#ffffff',
                                'borderRadius': '0 6px 6px 6px',
                                'padding': '20px',
                                'boxShadow': '0 2px 8px rgba(0, 0, 0, 0.1)'
                            }
                        ),
                        # Chroma Bars Tab
                        dcc.Tab(
                            label='Chroma Bars',
                            value='tab-0',
                            children=[
                                html.H3(
                                    "Chroma Bars",
                                    style={
                                        'fontSize': '25px',
                                        'textAlign': 'center',
                                        'margin': '20px 0',
                                        'color': '#1f2c51',
                                        'fontWeight': '600'
                                    }
                                ),
                                html.P(
                                    "The Chroma Bars visualization displays the intensity of different pitch classes (C, C#, D, etc.) at a specific point in the audio. Chroma features represent the distribution of energy across the 12 pitch classes in the musical octave, which helps identify the harmonic content of the audio. This bar chart shows how prominent each pitch class is at the current playback position, providing a snapshot of the musical notes being played. For example, a high bar for 'C' indicates a strong presence of the C note or its octaves. This visualization is useful for understanding the harmonic structure of the audio at any given moment.",
                                    style={
                                        'fontSize': '16px',
                                        'margin': '20px',
                                        'color': '#555'
                                    }
                                ),
                                html.H4(
                                    "What You Will See",
                                    style={
                                        'fontSize': '20px',
                                        'margin': '20px 20px 10px',
                                        'color': '#34495e',
                                        'fontWeight': '500'
                                    }
                                ),
                                html.P(
                                    "A bar chart with 12 bars, each labeled with a pitch class (C, C#, D, etc.), where the height of each bar represents the intensity of that pitch class.",
                                    style={
                                        'fontSize': '16px',
                                        'margin': '10px 20px',
                                        'color': '#555'
                                    }
                                ),
                                html.H4(
                                    "Why It Matters",
                                    style={
                                        'fontSize': '20px',
                                        'margin': '20px 20px 10px',
                                        'color': '#34495e',
                                        'fontWeight': '500'
                                    }
                                ),
                                html.P(
                                    "It helps users identify which notes are dominant at a specific time, aiding in tasks like melody analysis or chord detection.",
                                    style={
                                        'fontSize': '16px',
                                        'margin': '10px 20px',
                                        'color': '#555'
                                    }
                                ),
                                html.H4(
                                    "How to Use",
                                    style={
                                        'fontSize': '20px',
                                        'margin': '20px 20px 10px',
                                        'color': '#34495e',
                                        'fontWeight': '500'
                                    }
                                ),
                                html.P(
                                    "Move the playback slider to explore how the pitch class intensities change over time. This is particularly useful for analyzing melodies or harmonic progressions in real-time.",
                                    style={
                                        'fontSize': '16px',
                                        'margin': '10px 20px',
                                        'color': '#555'
                                    }
                                ),
                                dcc.Graph(
                                    id='chroma-figure',
                                    style={
                                        'backgroundColor': '#ffffff',
                                        'borderRadius': '6px',
                                        'padding': '10px',
                                        'margin': '20px 0',
                                        'boxShadow': '0 2px 8px rgba(0, 0, 0, 0.05)'
                                    }
                                )
                            ],
                            style={
                                'backgroundColor': '#ffffff',
                                'borderRadius': '0 6px 6px 6px',
                                'padding': '20px',
                                'boxShadow': '0 2px 8px rgba(0, 0, 0, 0.1)'
                            }
                        ),
                        # Chroma Heatmaps Tab
                        dcc.Tab(
                            label='Chroma Heatmaps',
                            value='tab-1',
                            children=[
                                html.H3(
                                    "Chroma Heatmaps",
                                    style={
                                        'fontSize': '25px',
                                        'textAlign': 'center',
                                        'margin': '20px 0',
                                        'color': '#1f2c51',
                                        'fontWeight': '600'
                                    }
                                ),
                                html.P(
                                    "The Chroma Heatmaps visualization provides a time-based view of the chroma features for each separated track (Vocals, Drums, Bass, Other). Each heatmap shows the intensity of the 12 pitch classes over a time window centered at the current playback position. The x-axis represents time, the y-axis lists the pitch classes, and the color intensity indicates the strength of each pitch class at a given time. Each track has its own heatmap, allowing you to compare the harmonic content across different instruments. This visualization is ideal for observing how the harmonic structure evolves over time and how different tracks contribute to the overall harmony.",
                                    style={
                                        'fontSize': '16px',
                                        'margin': '20px',
                                        'color': '#555'
                                    }
                                ),
                                html.H4(
                                    "What You Will See",
                                    style={
                                        'fontSize': '20px',
                                        'margin': '20px 20px 10px',
                                        'color': '#34495e',
                                        'fontWeight': '500'
                                    }
                                ),
                                html.P(
                                    "Four heatmaps (one per track) arranged in a 2x2 grid, with color gradients showing pitch class intensities over a 5-second window around the current playback position.",
                                    style={
                                        'fontSize': '16px',
                                        'margin': '10px 20px',
                                        'color': '#555'
                                    }
                                ),
                                html.H4(
                                    "Why It Matters",
                                    style={
                                        'fontSize': '20px',
                                        'margin': '20px 20px 10px',
                                        'color': '#34495e',
                                        'fontWeight': '500'
                                    }
                                ),
                                html.P(
                                    "It reveals the harmonic evolution of each track, helping users understand how different instruments contribute to the music’s harmony and identify patterns like repeating motifs or chord changes.",
                                    style={
                                        'fontSize': '16px',
                                        'margin': '10px 20px',
                                        'color': '#555'
                                    }
                                ),
                                html.H4(
                                    "How to Use",
                                    style={
                                        'fontSize': '20px',
                                        'margin': '20px 20px 10px',
                                        'color': '#34495e',
                                        'fontWeight': '500'
                                    }
                                ),
                                html.P(
                                    "Use the playback slider to navigate through the audio. Watch how the color patterns shift to see changes in pitch class prominence, such as a strong presence of certain notes during a vocal melody or bass line.",
                                    style={
                                        'fontSize': '16px',
                                        'margin': '10px 20px',
                                        'color': '#555'
                                    }
                                ),
                                html.Div(
                                    dcc.Graph(id='waveform-figure'),
                                    style={
                                        'display': 'flex',
                                        'justifyContent': 'center',
                                        'alignItems': 'center',
                                        'width': '100%',
                                        'backgroundColor': '#ffffff',
                                        'borderRadius': '6px',
                                        'padding': '10px',
                                        'margin': '20px 0',
                                        'boxShadow': '0 2px 8px rgba(0, 0, 0, 0.05)'
                                    }
                                )
                            ],
                            style={
                                'backgroundColor': '#ffffff',
                                'borderRadius': '0 6px 6px 6px',
                                'padding': '20px',
                                'boxShadow': '0 2px 8px rgba(0, 0, 0, 0.1)'
                            }
                        ),
                        # Chord Detection Tab
                        dcc.Tab(
                            label='Chord Detection',
                            value='tab-2',
                            children=[
                                html.H3(
                                    "Chord Detection",
                                    style={
                                        'fontSize': '25px',
                                        'textAlign': 'center',
                                        'margin': '20px 0',
                                        'color': '#1f2c51',
                                        'fontWeight': '600'
                                    }
                                ),
                                html.P(
                                    "The Chord Detection visualization identifies and displays the chords present in the audio at the current playback position, along with a progression of chords over a time window. The top plot shows the current chord (e.g., 'C', 'Am', or 'None') and its harmonic type (consonant or dissonant), accompanied by a bar chart of the chroma vector’s intensity across pitch classes. The bottom plot shows a timeline of detected chords around the current position, with labels indicating chord names and colors (green for consonant, red for dissonant) to highlight their harmonic quality. This visualization helps users understand the harmonic structure of the music and track chord changes over time.",
                                    style={
                                        'fontSize': '16px',
                                        'margin': '20px',
                                        'color': '#555'
                                    }
                                ),
                                html.H4(
                                    "What You Will See",
                                    style={
                                        'fontSize': '20px',
                                        'margin': '20px 20px 10px',
                                        'color': '#34495e',
                                        'fontWeight': '500'
                                    }
                                ),
                                html.P(
                                    "A two-part figure: the top part shows the current chord and a chroma bar chart, while the bottom part shows a timeline of chords with labels and colors indicating their type.",
                                    style={
                                        'fontSize': '16px',
                                        'margin': '10px 20px',
                                        'color': '#555'
                                    }
                                ),
                                html.H4(
                                    "Why It Matters",
                                    style={
                                        'fontSize': '20px',
                                        'margin': '20px 20px 10px',
                                        'color': '#34495e',
                                        'fontWeight': '500'
                                    }
                                ),
                                html.P(
                                    "It provides insight into the chord progression, which is fundamental to understanding the harmonic framework of a piece of music. It’s useful for musicians, composers, or analysts studying chord transitions or harmonic complexity.",
                                    style={
                                        'fontSize': '16px',
                                        'margin': '10px 20px',
                                        'color': '#555'
                                    }
                                ),
                                html.H4(
                                    "How to Use",
                                    style={
                                        'fontSize': '20px',
                                        'margin': '20px 20px 10px',
                                        'color': '#34495e',
                                        'fontWeight': '500'
                                    }
                                ),
                                html.P(
                                    "Move the playback slider to see the current chord and its context within the chord progression. Pay attention to the color coding to distinguish between consonant (stable, harmonious) and dissonant (tense, unstable) chords.",
                                    style={
                                        'fontSize': '16px',
                                        'margin': '10px 20px',
                                        'color': '#555'
                                    }
                                ),
                                html.H4(
                                    "Chord Type Explained",
                                    style={
                                        'fontSize': '20px',
                                        'margin': '20px 20px 10px',
                                        'color': '#34495e',
                                        'fontWeight': '500'
                                    }
                                ),
                                html.P(
                                    "1. Major (e.g., Root + Major Third + Perfect Fifth): Bright and stable, harmonious (green)\n"
                                    "2. Minor (e.g., Root + Minor Third + Perfect Fifth): Soft and slightly sad, harmonious (green)\n"
                                    "3. Augmented (e.g., Root + Minor Third + Diminished Fifth): Tense and unstable, dissonant (red)\n"
                                    "4. Diminished (e.g., Root + Major Third + Augmented Fifth): Sharp and unstable, dissonant (red)",
                                    style={
                                        'fontSize': '16px',
                                        'margin': '10px 20px',
                                        'color': '#555',
                                        'whiteSpace': 'pre-line'
                                    }
                                ),                                  
                                dcc.Graph(
                                    id='chord-figure',
                                    style={
                                        'backgroundColor': '#ffffff',
                                        'borderRadius': '6px',
                                        'padding': '10px',
                                        'margin': '20px 0',
                                        'boxShadow': '0 2px 8px rgba(0, 0, 0, 0.05)'
                                    }
                                )
                            ],
                            style={
                                'backgroundColor': '#ffffff',
                                'borderRadius': '0 6px 6px 6px',
                                'padding': '20px',
                                'boxShadow': '0 2px 8px rgba(0, 0, 0, 0.1)'
                            }
                        ),
                        # Real-time Chord Statistics Tab
                        dcc.Tab(
                            label='Real-time Chord Statistics',
                            value='tab-3',
                            children=[
                                html.H3(
                                    "Real-time Chord Statistics",
                                    style={
                                        'fontSize': '25px',
                                        'textAlign': 'center',
                                        'margin': '20px 0',
                                        'color': '#1f2c51',
                                        'fontWeight': '600'
                                    }
                                ),
                                html.P(
                                    "The Real-time Chord Statistics visualization summarizes the distribution of chord types (consonant, dissonant, or none) in the audio up to the current playback position. It presents a bar chart with three categories: Consonant (major/minor chords), Dissonant (augmented/diminished chords), and None (no clear chord detected). The height of each bar indicates the number of frames where each chord type was detected. This visualization provides a high-level overview of the harmonic character of the audio, showing whether the music leans toward stable, harmonious chords or more complex, dissonant ones.",
                                    style={
                                        'fontSize': '16px',
                                        'margin': '20px',
                                        'color': '#555'
                                    }
                                ),
                                html.H4(
                                    "What You Will See",
                                    style={
                                        'fontSize': '20px',
                                        'margin': '20px 20px 10px',
                                        'color': '#34495e',
                                        'fontWeight': '500'
                                    }
                                ),
                                html.P(
                                    "A bar chart with three bars labeled 'Consonant,' 'Dissonant,' and 'None,' colored green, red, and gray, respectively, showing the count of each chord type.",
                                    style={
                                        'fontSize': '16px',
                                        'margin': '10px 20px',
                                        'color': '#555'
                                    }
                                ),
                                html.H4(
                                    "Why It Matters",
                                    style={
                                        'fontSize': '20px',
                                        'margin': '20px 20px 10px',
                                        'color': '#34495e',
                                        'fontWeight': '500'
                                    }
                                ),
                                html.P(
                                    "It offers a quick way to assess the harmonic complexity of the audio, which can be useful for comparing different sections of a song or different tracks. For example, a high 'None' count might indicate atonal or rhythm-heavy sections.",
                                    style={
                                        'fontSize': '16px',
                                        'margin': '10px 20px',
                                        'color': '#555'
                                    }
                                ),
                                html.H4(
                                    "How to Use",
                                    style={
                                        'fontSize': '20px',
                                        'margin': '20px 20px 10px',
                                        'color': '#34495e',
                                        'fontWeight': '500'
                                    }
                                ),
                                html.P(
                                    "As you move the playback slider, the chart updates to reflect the chord statistics up to that point. Use this to compare the harmonic content of different parts of the audio or across different tracks.",
                                    style={
                                        'fontSize': '16px',
                                        'margin': '10px 20px',
                                        'color': '#555'
                                    }
                                ),
                                dcc.Graph(
                                    id='chord-stat-figure',
                                    style={
                                        'backgroundColor': '#ffffff',
                                        'borderRadius': '6px',
                                        'padding': '10px',
                                        'margin': '20px 0',
                                        'boxShadow': '0 2px 8px rgba(0, 0, 0, 0.05)'
                                    }
                                )
                            ],
                            style={
                                'backgroundColor': '#ffffff',
                                'borderRadius': '0 6px 6px 6px',
                                'padding': '20px',
                                'boxShadow': '0 2px 8px rgba(0, 0, 0, 0.1)'
                            }
                        ),
                        # Tempo Waveform Tab
                        dcc.Tab(
                            label='Tempo Waveform',
                            value='tab-4',
                            children=[
                                html.H3(
                                    "Tempo Waveform",
                                    style={
                                        'fontSize': '25px',
                                        'textAlign': 'center',
                                        'margin': '20px 0',
                                        'color': '#1f2c51',
                                        'fontWeight': '600'
                                    }
                                ),
                                html.P(
                                    "The Tempo Waveform visualization displays the onset strength and beat patterns for each track (Vocals, Drums, Bass, Other) over a time window around the current playback position. Each track has a plot showing the onset strength (a measure of sudden changes in audio, like note or drum hits) as a line, with markers indicating strong beats (triangles) and weak beats (circles). A red dashed line marks the current playback position. This visualization helps users understand the rhythmic structure of the audio, including the tempo (beats per minute) and the placement of beats within each track.",
                                    style={
                                        'fontSize': '16px',
                                        'margin': '20px',
                                        'color': '#555'
                                    }
                                ),
                                html.H4(
                                    "What You Will See",
                                    style={
                                        'fontSize': '20px',
                                        'margin': '20px 20px 10px',
                                        'color': '#34495e',
                                        'fontWeight': '500'
                                    }
                                ),
                                html.P(
                                    "Four plots (one per track) in a 2x2 grid, each showing the onset strength waveform, strong and weak beat markers, and a vertical line indicating the current time within a 5-second window.",
                                    style={
                                        'fontSize': '16px',
                                        'margin': '10px 20px',
                                        'color': '#555'
                                    }
                                ),
                                html.H4(
                                    "Why It Matters",
                                    style={
                                        'fontSize': '20px',
                                        'margin': '20px 20px 10px',
                                        'color': '#34495e',
                                        'fontWeight': '500'
                                    }
                                ),
                                html.P(
                                    "It highlights the rhythmic characteristics of each track, making it easier to analyze tempo, beat placement, and rhythmic complexity. For example, the Drums track might show clear, regular beats, while the Vocals track might have more irregular onsets.",
                                    style={
                                        'fontSize': '16px',
                                        'margin': '10px 20px',
                                        'color': '#555'
                                    }
                                ),
                                html.H4(
                                    "How to Use",
                                    style={
                                        'fontSize': '20px',
                                        'margin': '20px 20px 10px',
                                        'color': '#34495e',
                                        'fontWeight': '500'
                                    }
                                ),
                                html.P(
                                    "Move the playback slider to explore the rhythmic patterns. Look for strong beat markers to identify the main pulse of the music and compare how different tracks contribute to the overall rhythm.",
                                    style={
                                        'fontSize': '16px',
                                        'margin': '10px 20px',
                                        'color': '#555'
                                    }
                                ),
                                dcc.Graph(
                                    id='tempo-figure',
                                    style={
                                        'backgroundColor': '#ffffff',
                                        'borderRadius': '6px',
                                        'padding': '10px',
                                        'margin': '20px 0',
                                        'boxShadow': '0 2px 8px rgba(0, 0, 0, 0.05)'
                                    }
                                )
                            ],
                            style={
                                'backgroundColor': '#ffffff',
                                'borderRadius': '0 6px 6px 6px',
                                'padding': '20px',
                                'boxShadow': '0 2px 8px rgba(0, 0, 0, 0.1)'
                            }
                        ),
                        # Piano Roll Tab
                        dcc.Tab(
                            label='Piano Roll',
                            value='tab-5',
                            children=[
                                html.H3(
                                    "Piano Roll",
                                    style={
                                        'fontSize': '25px',
                                        'textAlign': 'center',
                                        'margin': '20px 0',
                                        'color': '#1f2c51',
                                        'fontWeight': '600'
                                    }
                                ),
                                html.P(
                                    "The Piano Roll visualization shows the pitch classes (C, C#, D, etc.) played over time for selected tracks, resembling a traditional piano roll used in music production software. Each track’s notes are represented as colored rectangles, with the x-axis showing time, the y-axis showing pitch classes, and the length of each rectangle indicating note duration. Users can select which tracks to display (Vocals, Drums, Bass, Other) using a dropdown menu. A red dashed line indicates the current playback position. This visualization is ideal for analyzing the melodic and harmonic content of individual tracks or comparing note patterns across tracks.",
                                    style={
                                        'fontSize': '16px',
                                        'margin': '20px',
                                        'color': '#555'
                                    }
                                ),
                                html.H4(
                                    "What You Will See",
                                    style={
                                        'fontSize': '20px',
                                        'margin': '20px 20px 10px',
                                        'color': '#34495e',
                                        'fontWeight': '500'
                                    }
                                ),
                                html.P(
                                    "A plot with colored rectangles for each note, arranged by pitch class and time, with a dropdown to select tracks and a red line showing the current position within a 5-second window.",
                                    style={
                                        'fontSize': '16px',
                                        'margin': '10px 20px',
                                        'color': '#555'
                                    }
                                ),
                                html.H4(
                                    "Why It Matters",
                                    style={
                                        'fontSize': '20px',
                                        'margin': '20px 20px 10px',
                                        'color': '#34495e',
                                        'fontWeight': '500'
                                    }
                                ),
                                html.P(
                                    "It provides a clear, visual representation of the notes played, making it easy to see melodies, harmonies, or rhythmic patterns. It’s particularly useful for musicians or producers analyzing specific parts of a track.",
                                    style={
                                        'fontSize': '16px',
                                        'margin': '10px 20px',
                                        'color': '#555'
                                    }
                                ),
                                html.H4(
                                    "How to Use",
                                    style={
                                        'fontSize': '20px',
                                        'margin': '20px 20px 10px',
                                        'color': '#34495e',
                                        'fontWeight': '500'
                                    }
                                ),
                                html.P(
                                    "Select tracks from the dropdown to focus on specific instruments. Move the playback slider to follow the notes in real-time, and observe how different tracks contribute to the melody or harmony.",
                                    style={
                                        'fontSize': '16px',
                                        'margin': '10px 20px',
                                        'color': '#555'
                                    }
                                ),
                                html.Label(
                                    "Select Tracks to Display:",
                                    style={
                                        'fontSize': '16px',
                                        'margin': '20px 20px 10px',
                                        'color': '#333',
                                        'fontWeight': '500'
                                    }
                                ),
                                dcc.Dropdown(
                                    id='piano-roll-track-selector',
                                    options=[{'label': t, 'value': t} for t in ['Vocals', 'Drums', 'Bass', 'Other']],
                                    multi=True,
                                    value=['Vocals', 'Drums', 'Bass', 'Other'],
                                    style={
                                        'width': '50%',
                                        'margin': '10px 20px',
                                        'borderRadius': '6px',
                                        'border': '1px solid #ccc',
                                        'backgroundColor': '#ffffff',
                                        'fontSize': '14px'
                                    }
                                ),
                                dcc.Graph(
                                    id='piano-roll-figure',
                                    style={
                                        'backgroundColor': '#ffffff',
                                        'borderRadius': '6px',
                                        'padding': '10px',
                                        'margin': '20px 0',
                                        'boxShadow': '0 2px 8px rgba(0, 0, 0, 0.05)'
                                    }
                                )
                            ],
                            style={
                                'backgroundColor': '#ffffff',
                                'borderRadius': '0 6px 6px 6px',
                                'padding': '20px',
                                'boxShadow': '0 2px 8px rgba(0, 0, 0, 0.1)'
                            }
                        ),
                        # Chord Segments by Track Tab
                        dcc.Tab(
                            label='Chord Segments by Track',
                            value='tab-6',
                            children=[
                                html.H3(
                                    "Chord Segments by Track",
                                    style={
                                        'fontSize': '25px',
                                        'textAlign': 'center',
                                        'margin': '20px 0',
                                        'color': '#1f2c51',
                                        'fontWeight': '600'
                                    }
                                ),
                                html.P(
                                    "The Chord Segments by Track visualization allows users to explore and edit detected chord segments for each track (Vocals, Drums, Bass, Other). It includes a dropdown to select a track, a table listing the detected chords with their start times, durations, and pitch shifts, and a sequence table for building a custom chord sequence. Users can modify segment parameters, add segments to a sequence, play individual segments, or save the sequence as an audio file. A chord sequence visualization plots the selected segments as colored rectangles, showing their timing, track, and pitch shift. This tab is designed for interactive harmonic analysis and creative remixing.",
                                    style={
                                        'fontSize': '16px',
                                        'margin': '20px',
                                        'color': '#555'
                                    }
                                ),
                                html.H4(
                                    "What You Will See",
                                    style={
                                        'fontSize': '20px',
                                        'margin': '20px 20px 10px',
                                        'color': '#34495e',
                                        'fontWeight': '500'
                                    }
                                ),
                                html.P(
                                    "A dropdown for track selection, an editable table of chord segments, a sequence table, buttons for adding/removing/playing segments, and a plot showing the chord sequence as rectangles with chord labels and pitch shift information.",
                                    style={
                                        'fontSize': '16px',
                                        'margin': '10px 20px',
                                        'color': '#555'
                                    }
                                ),
                                html.H4(
                                    "Why It Matters",
                                    style={
                                        'fontSize': '20px',
                                        'margin': '20px 20px 10px',
                                        'color': '#34495e',
                                        'fontWeight': '500'
                                    }
                                ),
                                html.P(
                                    "It enables users to dive deep into the harmonic structure of each track, edit chord timings or pitch, and create custom sequences for remixing or analysis. It’s a powerful tool for music producers or educators studying chord progressions.",
                                    style={
                                        'fontSize': '16px',
                                        'margin': '10px 20px',
                                        'color': '#555'
                                    }
                                ),
                                html.H4(
                                    "How to Use",
                                    style={
                                        'fontSize': '20px',
                                        'margin': '20px 20px 10px',
                                        'color': '#34495e',
                                        'fontWeight': '500'
                                    }
                                ),
                                html.P(
                                    "1. Select a track: Select a track (e.g. Vocal or Drums) from the drop-down menu to view a table of its chord segments, including the chord name, start time (seconds), duration (seconds), and pitch change (semitones).\n"
                                    "2. Edit segments: Click a start time, duration, or pitch change cell in the table, enter a new value, and press Enter or click outside the table to save the changes. Note: Chord names are not editable.\n"
                                    "3. Build a sequence: Select one or more rows in the table using the checkboxes and click the Add to Sequence button to add the selected chord segments to a custom sequence. The sequence will be displayed in the table below, with a graphical visualization of the chord timing and pitch changes.\n"
                                    "4. Play and export: Click the Play Sequence button to preview the audio effect of the sequence. When finished, click the Save Sequence button to export the sequence as an audio file (MP3 format) for download.",
                                    style={
                                        'fontSize': '16px',
                                        'margin': '10px 20px',
                                        'color': '#555',
                                        'whiteSpace': 'pre-line'
                                    }
                                ),
                                html.Label(
                                    "Select Track:",
                                    style={
                                        'fontSize': '16px',
                                        'margin': '20px 20px 10px',
                                        'color': '#333',
                                        'fontWeight': '500'
                                    }
                                ),
                                dcc.Dropdown(
                                    id='chord-track-selector',
                                    options=[],
                                    value=None,
                                    style={
                                        'width': '50%',
                                        'margin': '10px 20px',
                                        'borderRadius': '6px',
                                        'border': '1px solid #ccc',
                                        'backgroundColor': '#ffffff',
                                        'fontSize': '14px'
                                    }
                                ),
                                html.Label(
                                    "Chord Segments:",
                                    style={
                                        'fontSize': '16px',
                                        'margin': '20px 20px 10px',
                                        'color': '#333',
                                        'fontWeight': '500'
                                    }
                                ),
                                html.Div(
                                    id='chord-track-table-container',
                                    children=[
                                        dash_table.DataTable(
                                            id='chord-track-table',
                                            columns=[
                                                {"name": "Chord", "id": "chord"},
                                                {"name": "Start (s)", "id": "start", "editable": True},
                                                {"name": "Duration (s)", "id": "duration", "editable": True},
                                                {"name": "Pitch Shift (semitones)", "id": "pitch_shift", "editable": True}
                                            ],
                                            data=[],
                                            row_selectable='multi',
                                            style_table={
                                                'maxHeight': '300px',
                                                'overflowY': 'auto',
                                                'borderRadius': '6px',
                                                'boxShadow': '0 2px 8px rgba(0, 0, 0, 0.05)',
                                                'backgroundColor': '#ffffff'
                                            },
                                            style_cell={
                                                'textAlign': 'center',
                                                'padding': '12px',
                                                'fontSize': '14px',
                                                'color': '#333'
                                            },
                                            style_header={
                                                'fontWeight': 'bold',
                                                'backgroundColor': '#f8f9fa',
                                                'borderBottom': '1px solid #dee2e6',
                                                'color': '#333'
                                            },
                                            style_data_conditional=[
                                                {'if': {'row_index': 'odd'}, 'backgroundColor': '#f9f9f9'},
                                                {
                                                    'if': {'state': 'selected'},
                                                    'backgroundColor': 'rgba(0, 116, 217, 0.3)',
                                                    'border': '1px solid rgb(0, 116, 217)'
                                                }
                                            ]
                                        ),
                                        html.Button(
                                            "Add to Sequence",
                                            id='add-to-sequence-button',
                                            n_clicks=0,
                                            style={
                                                'backgroundColor': '#007bff',
                                                'color': '#ffffff',
                                                'border': 'none',
                                                'borderRadius': '6px',
                                                'padding': '10px 20px',
                                                'fontSize': '14px',
                                                'fontWeight': '500',
                                                'cursor': 'pointer',
                                                'margin': '10px 10px 10px 20px',
                                                'transition': 'background-color 0.3s ease'
                                            }
                                        ),
                                        html.Button(
                                            "Play Selected",
                                            id='play-selected-button',
                                            n_clicks=0,
                                            style={
                                                'backgroundColor': '#007bff',
                                                'color': '#ffffff',
                                                'border': 'none',
                                                'borderRadius': '6px',
                                                'padding': '10px 20px',
                                                'fontSize': '14px',
                                                'fontWeight': '500',
                                                'cursor': 'pointer',
                                                'margin': '10px',
                                                'transition': 'background-color 0.3s ease'
                                            }
                                        ),
                                    ],
                                    style={
                                        'margin': '20px'
                                    }
                                ),
                                html.Label(
                                    "Edited Chord Sequence:",
                                    style={
                                        'fontSize': '16px',
                                        'margin': '20px 20px 10px',
                                        'color': '#333',
                                        'fontWeight': '500'
                                    }
                                ),
                                dash_table.DataTable(
                                    id='chord-sequence-table',
                                    columns=[
                                        {"name": "Chord", "id": "chord"},
                                        {"name": "Start (s)", "id": "start"},
                                        {"name": "Duration (s)", "id": "duration"},
                                        {"name": "Track", "id": "track"},
                                        {"name": "Pitch Shift", "id": "pitch_shift"}
                                    ],
                                    data=[],
                                    row_selectable='multi',
                                    style_table={
                                        'maxHeight': '300px',
                                        'overflowY': 'auto',
                                        'borderRadius': '6px',
                                        'boxShadow': '0 2px 8px rgba(0, 0, 0, 0.05)',
                                        'backgroundColor': '#ffffff'
                                    },
                                    style_cell={
                                        'textAlign': 'center',
                                        'padding': '12px',
                                        'fontSize': '14px',
                                        'color': '#333'
                                    },
                                    style_header={
                                        'fontWeight': 'bold',
                                        'backgroundColor': '#f8f9fa',
                                        'borderBottom': '1px solid #dee2e6',
                                        'color': '#333'
                                    },
                                    style_data_conditional=[
                                        {'if': {'row_index': 'odd'}, 'backgroundColor': '#f9f9f9'},
                                        {
                                            'if': {'state': 'selected'},
                                            'backgroundColor': 'rgba(0, 116, 217, 0.3)',
                                            'border': '1px solid rgb(0, 116, 217)'
                                        }
                                    ]
                                ),
                                html.Div([
                                    html.Button(
                                        "Remove Selected",
                                        id='remove-from-sequence-button',
                                        n_clicks=0,
                                        style={
                                            'backgroundColor': '#007bff',
                                            'color': '#ffffff',
                                            'border': 'none',
                                            'borderRadius': '6px',
                                            'padding': '10px 20px',
                                            'fontSize': '14px',
                                            'fontWeight': '500',
                                            'cursor': 'pointer',
                                            'margin': '10px 10px 10px 20px',
                                            'transition': 'background-color 0.3s ease'
                                        }
                                    ),
                                    html.Button(
                                        "Save Sequence",
                                        id='save-sequence-button',
                                        n_clicks=0,
                                        style={
                                            'backgroundColor': '#007bff',
                                            'color': '#ffffff',
                                            'border': 'none',
                                            'borderRadius': '6px',
                                            'padding': '10px 20px',
                                            'fontSize': '14px',
                                            'fontWeight': '500',
                                            'cursor': 'pointer',
                                            'margin': '10px',
                                            'transition': 'background-color 0.3s ease'
                                        }
                                    ),
                                    html.Button(
                                        "Play Sequence",
                                        id='play-sequence-button',
                                        n_clicks=0,
                                        style={
                                            'backgroundColor': '#007bff',
                                            'color': '#ffffff',
                                            'border': 'none',
                                            'borderRadius': '6px',
                                            'padding': '10px 20px',
                                            'fontSize': '14px',
                                            'fontWeight': '500',
                                            'cursor': 'pointer',
                                            'margin': '10px',
                                            'transition': 'background-color 0.3s ease'
                                        }
                                    )
                                ], style={'margin': '20px'}),
                                dcc.Graph(
                                    id='chord-sequence-visualization',
                                    style={
                                        'backgroundColor': '#ffffff',
                                        'borderRadius': '6px',
                                        'padding': '10px',
                                        'margin': '20px 0',
                                        'boxShadow': '0 2px 8px rgba(0, 0, 0, 0.05)'
                                    }
                                ),
                                html.Label(
                                    "Track Audio Playback:",
                                    style={
                                        'fontSize': '16px',
                                        'margin': '20px 20px 10px',
                                        'color': '#333',
                                        'fontWeight': '500'
                                    }
                                ),
                                html.Div(
                                    id='track-audio-container',
                                    children=[],
                                    style={
                                        'margin': '20px'
                                    }
                                ),
                            ],
                            style={
                                'backgroundColor': '#ffffff',
                                'borderRadius': '0 6px 6px 6px',
                                'padding': '20px',
                                'boxShadow': '0 2px 8px rgba(0, 0, 0, 0.1)'
                            }
                        ),
                    ]
                ),
                # Playback Slider
                dcc.Slider(
                    id='playback-position',
                    min=0,
                    max=1,
                    step=0.001,
                    value=0,
                    marks={0: '0', 1: 'End'},
                    updatemode='drag',
                ),
                # Hidden Components
                dcc.Interval(id='auto-update', interval=200, n_intervals=0, disabled=True),
                dcc.Store(id='tracks-store'),
                dcc.Store(id='features-store'),
                dcc.Store(id='duration-store'),
                dcc.Store(id='_audio-content', data=None),
                dcc.Store(id='play-state', data={'is_playing': False}),
                dcc.Store(id='last-position', data=0),
                dcc.Store(id='tempo-file-store', data=None),
                dcc.Store(id='chord-segments-store', data={}),
                dcc.Store(id='chord-sequence-store', data=[]),
                dcc.Interval(id='_audio-sync-timer', interval=200),
                dcc.Input(id='_current-audio-position', value=0, style={'display': 'none'}),
                dcc.Input(id='_audio-duration', value=0, style={'display': 'none'}),
                dcc.Input(id='_stored-audio-position', value=0, style={'display': 'none'}),
                dcc.Input(id='_n-const-polls', value=0, type='number', style={'display': 'none'}),
                dcc.Store(id='_audio-duration-store', data=None),
                dcc.Store(id='track-audio-store', data={}),
                dcc.Store(id='original-track-audio-store', data={}),
                dcc.Download(id="download-sequence"),
            ], style={
                'fontFamily': '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
                'backgroundColor': '#f5f7fa',
                'color': '#333',
                'padding': '20px',
                'maxWidth': '1200px',
                'margin': '0 auto'
            })
        ])

        self._setup_audio_position_sync()
        self._register_callbacks()

    def _setup_audio_position_sync(self):
        self.app.clientside_callback(
            '''
            function(value) {
                const audio = document.getElementById('_audio-controls');
                if (!audio) return [null, null, null];
                return [audio.currentTime / audio.duration, audio.duration, audio.duration];
            }
            ''',
            Output('_current-audio-position', 'value'),
            Output('_audio-duration', 'value'),
            Output('_audio-duration-store', 'data'),
            Input('_audio-sync-timer', 'n_intervals'),
            prevent_initial_call=True
        )

        @self.app.callback(
            Output('_stored-audio-position', 'value'),
            Output('_audio-sync-timer', 'interval'),
            Output('_n-const-polls', 'value'),
            Input('_current-audio-position', 'value'),
            State('_stored-audio-position', 'value'),
            State('_n-const-polls', 'value'),
            State('_audio-sync-timer', 'interval'),
            prevent_initial_call=True
        )
        def update_audio_position(current_pos, stored_pos, n_const_polls, current_interval):
            sync_interval_ms = 200
            idle_interval_ms = 1000
            n_sync_before_idle = 5
            if current_pos is None or current_pos == stored_pos:
                if n_const_polls > n_sync_before_idle and current_interval != idle_interval_ms:
                    return no_update, idle_interval_ms, n_const_polls + 1
                else:
                    return no_update, no_update, n_const_polls + 1
            else:
                if current_interval != sync_interval_ms:
                    return current_pos, sync_interval_ms, 0
                else:
                    return current_pos, no_update, 0

    def _register_callbacks(self):
        self.app.clientside_callback(
            '''
            let debounceTimeout;
            function debounce(func, wait) {
                return function(...args) {
                    clearTimeout(debounceTimeout);
                    debounceTimeout = setTimeout(() => func.apply(this, args), wait);
                };
            }
            function(slider_position, audio_duration) {
                const audio = document.getElementById('_audio-controls');
                if (!audio || !audio_duration) return window.dash_clientside.no_update;
                debounce(() => { audio.currentTime = slider_position * audio_duration; }, 100)();
                return window.dash_clientside.no_update;
            }
            ''',
            Output('_audio-controls', 'id'),
            Input('playback-position', 'value'),
            State('_audio-duration', 'value'),
            prevent_initial_call=True
        )

        self.app.clientside_callback(
            '''
            function(play_state, position, audio_duration) {
                const audio = document.getElementById('_audio-controls');
                if (!audio || !audio_duration) return window.dash_clientside.no_update;
                if (play_state.is_playing) {
                    if (position >= 0.999) audio.currentTime = 0;
                    audio.currentTime = position * audio_duration;
                    audio.play();
                } else {
                    audio.pause();
                }
                return window.dash_clientside.no_update;
            }
            ''',
            Output('_audio-controls', 'id', allow_duplicate=True),
            Input('play-state', 'data'),
            State('playback-position', 'value'),
            State('_audio-duration', 'value'),
            prevent_initial_call=True
        )

        self.app.clientside_callback(
            '''
            function(current_pos) {
                const audio = document.getElementById('_audio-controls');
                if (!audio) return window.dash_clientside.no_update;
                if (current_pos >= 0.999) {
                    audio.currentTime = 0;
                    audio.pause();
                    return 0;
                }
                return window.dash_clientside.no_update;
            }
            ''',
            Output('playback-position', 'value'),
            Input('_current-audio-position', 'value'),
            prevent_initial_call=True
        )

        @self.app.callback(
            Output('audio-load-status', 'children'),
            Input('_audio-duration-store', 'data'),
            prevent_initial_call=True
        )
        def update_audio_load_status(duration):
            if duration is not None:
                return "Audio loaded."
            return "Loading audio..."

        @self.app.callback(
            [
                Output('upload-status', 'children'),
                Output('_sound-file-display', 'children'),
                Output('_audio-content', 'data'),
                Output('tracks-store', 'data'),
                Output('features-store', 'data'),
                Output('duration-store', 'data'),
                Output('playback-position', 'value', allow_duplicate=True),
                Output('tempo-file-store', 'data'),
                Output('chord-segments-store', 'data'),
                Output('chord-track-selector', 'options'),
                Output('track-audio-store', 'data'),
                Output('track-audio-container', 'children'),
                Output('original-track-audio-store', 'data'),
            ],
            Input('upload-audio', 'contents'),
            State('upload-audio', 'filename'),
            prevent_initial_call=True
        )
        def process_uploaded_file(contents, filename):
            return self.process_uploaded_file(contents, filename)

        @self.app.callback(
            Output('chord-track-table', 'data'),
            Input('chord-track-selector', 'value'),
            State('chord-segments-store', 'data'),
            prevent_initial_call=True
        )
        def update_chord_track_table(selected_track, chord_segments):
            if not selected_track or not chord_segments:
                print(f"update_chord_track_table: No selected_track ({selected_track}) or chord_segments is empty")
                return []

            segments = chord_segments.get(selected_track, [])
            if not segments:
                print(f"No segments found for track: {selected_track}")
                return []

            for segment in segments:
                if 'pitch_shift' not in segment:
                    segment['pitch_shift'] = 0

            table_data = [
                {
                    "chord": segment['chord'],
                    "start": f"{float(segment['start']):.2f}",
                    "duration": f"{float(segment['duration']):.2f}",
                    "pitch_shift": segment['pitch_shift']
                }
                for segment in segments
            ]
            print(f"Updating chord-track-table with {len(table_data)} segments for {selected_track}: {table_data}")
            return table_data

        @self.app.callback(
            Output('track-audio-container', 'children', allow_duplicate=True),
            Input('chord-track-selector', 'value'),
            State('track-audio-store', 'data'),
            prevent_initial_call=True
        )
        def update_track_audio(selected_track, track_audio_data):
            if not track_audio_data:
                return []
            audio_elements = [
                self._track_audio_element(track_name, audio_src)
                for track_name, audio_src in track_audio_data.items()
            ]
            return audio_elements

        @self.app.callback(
            [
                Output('chord-segments-store', 'data', allow_duplicate=True),
                Output('track-audio-store', 'data', allow_duplicate=True),
                Output('upload-status', 'children', allow_duplicate=True),
            ],
            Input('chord-track-table', 'data'),
            State('chord-track-table', 'data_previous'),
            State('chord-track-selector', 'value'),
            State('chord-segments-store', 'data'),
            State('track-audio-store', 'data'),
            State('original-track-audio-store', 'data'),
            State('duration-store', 'data'),
            prevent_initial_call=True
        )
        def update_chord_segments(table_data, prev_table_data, selected_track, chord_segments, track_audio_data, original_track_audio_data, audio_duration):
            if not table_data or not selected_track or not chord_segments or not original_track_audio_data or not audio_duration:
                raise PreventUpdate

            if table_data == prev_table_data:
                raise PreventUpdate

            print(f"Updating segments for {selected_track}. New table data: {table_data}")

            updated_segments = chord_segments.copy()
            segments = updated_segments[selected_track]
            status_message = f"Updated segments for {selected_track}"
            has_error = False

            for i, row_data in enumerate(table_data):
                if i >= len(segments):
                    continue

                try:
                    new_start = float(row_data['start'])
                    if new_start < 0:
                        has_error = True
                        status_message = f"Error: Start time for segment {i} cannot be negative"
                        continue
                    if new_start > audio_duration:
                        has_error = True
                        status_message = f"Error: Start time for segment {i} exceeds audio duration ({audio_duration:.2f}s)"
                        continue
                    if segments[i]['start'] != new_start:
                        segments[i]['start'] = new_start
                        print(f"Updated segment {i} start to {new_start}s")

                    new_duration = float(row_data['duration'])
                    if new_duration <= 0:
                        has_error = True
                        status_message = f"Error: Duration for segment {i} must be positive"
                        continue
                    if new_start + new_duration > audio_duration:
                        has_error = True
                        status_message = f"Error: Segment {i} end time ({new_start + new_duration:.2f}s) exceeds audio duration ({audio_duration:.2f}s)"
                        continue
                    if segments[i]['duration'] != new_duration:
                        segments[i]['duration'] = new_duration
                        print(f"Updated segment {i} duration to {new_duration}s")

                    new_pitch_shift = int(row_data['pitch_shift'])
                    if segments[i].get('pitch_shift', 0) != new_pitch_shift:
                        segments[i]['pitch_shift'] = new_pitch_shift
                        print(f"Updated segment {i} pitch_shift to {new_pitch_shift}")

                except (ValueError, KeyError) as e:
                    has_error = True
                    status_message = f"Error in segment {i}: Invalid input ({str(e)})"
                    print(f"Error parsing segment {i}: {e}")
                    continue

            print(f"Updated chord-segments-store for {selected_track}: {segments}")

            updated_audio_data = track_audio_data.copy()
            original_audio_base64 = original_track_audio_data.get(selected_track, '')
            if original_audio_base64 and not has_error:
                audio_data = base64.b64decode(original_audio_base64.split(',')[1])
                temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                with open(temp_wav.name, 'wb') as f:
                    f.write(audio_data)

                audio_segment = AudioSegment.from_wav(temp_wav.name)
                for segment in segments:
                    pitch_shift = segment.get('pitch_shift', 0)
                    if pitch_shift != 0:
                        start_ms = int(segment['start'] * 1000)
                        duration_ms = int(segment['duration'] * 1000)
                        end_ms = start_ms + duration_ms
                        if end_ms > len(audio_segment):
                            duration_ms = len(audio_segment) - start_ms
                        segment_audio = audio_segment[start_ms:start_ms + duration_ms]
                        shifted_audio = segment_audio._spawn(segment_audio.raw_data, overrides={
                            'frame_rate': int(segment_audio.frame_rate * (2 ** (pitch_shift / 12.0)))
                        })
                        shifted_audio = shifted_audio.set_frame_rate(segment_audio.frame_rate)
                        audio_segment = audio_segment[:start_ms] + shifted_audio + audio_segment[start_ms + duration_ms:]
                        print(f"Applied pitch shift {pitch_shift} to segment at {segment['start']}s for {segment['duration']}s")

                temp_shifted_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                audio_segment.export(temp_shifted_wav.name, format="wav")
                with open(temp_shifted_wav.name, 'rb') as f:
                    updated_audio_data[selected_track] = f"data:audio/wav;base64,{base64.b64encode(f.read()).decode('utf-8')}"

                os.unlink(temp_wav.name)
                os.unlink(temp_shifted_wav.name)

            return (
                updated_segments if not has_error else chord_segments,
                updated_audio_data,
                status_message
            )

        @self.app.callback(
            Output('track-audio-container', 'children', allow_duplicate=True),
            Input('track-audio-store', 'data'),
            prevent_initial_call=True
        )
        def update_track_audio_container(track_audio_data):
            if not track_audio_data:
                return []
            audio_elements = [
                self._track_audio_element(track_name, audio_src)
                for track_name, audio_src in track_audio_data.items()
            ]
            return audio_elements

        self.app.clientside_callback(
            '''
            function(n_clicks, selected_track, selected_rows, chord_segments, track_audio_data) {
                if (!n_clicks || !selected_track || !selected_rows.length || !chord_segments) {
                    console.log("Invalid input for play-selected");
                    return window.dash_clientside.no_update;
                }
                const audio = document.getElementById(`track-audio-${selected_track.toLowerCase()}`);
                if (!audio) {
                    console.log(`Audio element not found for track: ${selected_track}`);
                    return window.dash_clientside.no_update;
                }

                const segments = chord_segments[selected_track];
                if (!segments || selected_rows[0] >= segments.length) {
                    console.log(`No valid segments for row: ${selected_rows[0]}`);
                    return window.dash_clientside.no_update;
                }

                const segment = segments[selected_rows[0]];
                audio.currentTime = parseFloat(segment.start);
                audio.play().then(() => {
                    console.log(`Playing segment at ${segment.start}s for ${segment.duration}s`);
                }).catch(error => {
                    console.error(`Playback error: ${error}`);
                });
                setTimeout(() => {
                    audio.pause();
                    console.log("Segment playback stopped");
                }, parseFloat(segment.duration) * 1000);

                return window.dash_clientside.no_update;
            }
            ''',
            Output('play-selected-button', 'n_clicks'),
            Input('play-selected-button', 'n_clicks'),
            State('chord-track-selector', 'value'),
            State('chord-track-table', 'selected_rows'),
            State('chord-segments-store', 'data'),
            State('track-audio-store', 'data'),
            prevent_initial_call=True
        )

        @self.app.callback(
            [
                Output('chord-sequence-store', 'data'),
                Output('chord-sequence-table', 'data'),
                Output('chord-track-table', 'selected_rows'),
                Output('chord-track-table', 'data', allow_duplicate=True),
            ],
            Input('add-to-sequence-button', 'n_clicks'),
            State('chord-track-selector', 'value'),
            State('chord-track-table', 'selected_rows'),
            State('chord-segments-store', 'data'),
            State('chord-sequence-store', 'data'),
            State('chord-track-table', 'data'),
            prevent_initial_call=True
        )
        def add_to_sequence(n_clicks, selected_track, selected_rows, chord_segments, current_sequence, table_data):
            if not n_clicks or not selected_track or not selected_rows or not chord_segments:
                raise PreventUpdate

            segments = chord_segments.get(selected_track, [])
            if not segments:
                print(f"No segments found for track: {selected_track}")
                raise PreventUpdate

            new_segments = [
                {
                    "chord": segments[row]["chord"],
                    "start": float(segments[row]["start"]),
                    "duration": float(segments[row]["duration"]),
                    "pitch_shift": segments[row].get("pitch_shift", 0),
                    "track": selected_track
                }
                for row in selected_rows if row < len(segments)
            ]
            if not new_segments:
                print(f"No valid segments selected: {selected_rows}")
                raise PreventUpdate

            print(f"Adding to sequence from {selected_track}: {new_segments}")

            updated_sequence = current_sequence + new_segments
            updated_sequence.sort(key=lambda x: float(x['start']))

            sequence_table_data = [
                {
                    "chord": seg["chord"],
                    "start": f"{seg['start']:.2f}",
                    "duration": f"{seg['duration']:.2f}",
                    "track": seg["track"],
                    "pitch_shift": seg.get("pitch_shift", 0)
                }
                for seg in updated_sequence
            ]

            updated_table_data = [
                {
                    "chord": segment["chord"],
                    "start": f"{float(segment['start']):.2f}",
                    "duration": f"{float(segment['duration']):.2f}",
                    "pitch_shift": segment.get("pitch_shift", 0)
                }
                for segment in segments
            ]

            print(f"Updated chord-sequence-table: {sequence_table_data}")
            print(f"Refreshed chord-track-table for {selected_track}: {updated_table_data}")

            return updated_sequence, sequence_table_data, [], updated_table_data

        @self.app.callback(
            Output('chord-sequence-store', 'data', allow_duplicate=True),
            Output('chord-sequence-table', 'data', allow_duplicate=True),
            Input('remove-from-sequence-button', 'n_clicks'),
            State('chord-sequence-table', 'selected_rows'),
            State('chord-sequence-store', 'data'),
            prevent_initial_call=True
        )
        def remove_from_sequence(n_clicks, selected_rows, current_sequence):
            if not n_clicks or not selected_rows or not current_sequence:
                raise PreventUpdate

            updated_sequence = [seg for i, seg in enumerate(current_sequence) if i not in selected_rows]

            updated_sequence.sort(key=lambda x: float(x['start']))

            table_data = [
                {
                    "chord": seg["chord"],
                    "start": f"{float(seg['start']):.2f}",
                    "duration": f"{float(seg['duration']):.2f}",
                    "track": seg["track"],
                    "pitch_shift": seg.get("pitch_shift", 0)
                }
                for seg in updated_sequence
            ]
            print(f"Removed from sequence: {table_data}")
            return updated_sequence, table_data

        @self.app.callback(
            [
                Output('upload-status', 'children', allow_duplicate=True),
                Output('download-sequence', 'data'),
            ],
            Input('save-sequence-button', 'n_clicks'),
            State('chord-sequence-store', 'data'),
            State('track-audio-store', 'data'),
            prevent_initial_call=True
        )
        def save_sequence(n_SB_clicks, chord_sequence, track_audio_data):
            if not n_SB_clicks or not chord_sequence:
                raise PreventUpdate

            segments_by_start = {}
            for segment in chord_sequence:
                start = float(segment['start'])
                if start not in segments_by_start:
                    segments_by_start[start] = []
                segments_by_start[start].append(segment)

            start_times = sorted(segments_by_start.keys())
            combined_audio = AudioSegment.empty()
            last_end_ms = 0

            for start_time in start_times:
                segments = segments_by_start[start_time]
                start_ms = int(start_time * 1000)

                if start_ms > last_end_ms:
                    silence_duration = start_ms - last_end_ms
                    combined_audio += AudioSegment.silent(duration=silence_duration)

                mixed_segment = None
                temp_files = []
                max_duration_ms = 0

                for segment in segments:
                    track_name = segment['track']
                    audio_base64 = track_audio_data.get(track_name, '')
                    if not audio_base64:
                        continue

                    audio_data = base64.b64decode(audio_base64.split(',')[1])
                    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                    with open(temp_wav.name, 'wb') as f:
                        f.write(audio_data)
                    temp_files.append(temp_wav.name)

                    audio_segment = AudioSegment.from_wav(temp_wav.name)
                    duration_ms = int(float(segment['duration']) * 1000)
                    segment_audio = audio_segment[:duration_ms]

                    if mixed_segment is None:
                        mixed_segment = segment_audio
                    else:
                        mixed_segment = mixed_segment.overlay(segment_audio)

                    max_duration_ms = max(max_duration_ms, duration_ms)

                if mixed_segment:
                    combined_audio += mixed_segment
                    last_end_ms = start_ms + max_duration_ms

                for temp_file in temp_files:
                    os.unlink(temp_file)

            sequence_file = "chord_sequence.mp3"
            combined_audio.export(sequence_file, format="mp3")

            return (
                f"Chord sequence saved to {sequence_file}",
                dcc.send_file(sequence_file, filename="chord_sequence.mp3")
            )

        @self.app.callback(
            Output('playback-status', 'children'),
            Input('play-sequence-button', 'n_clicks'),
            State('chord-sequence-table', 'selected_rows'),
            State('chord-sequence-store', 'data'),
            prevent_initial_call=True
        )
        def update_playback_status(n_clicks, selected_rows, chord_sequence):
            if not n_clicks or not chord_sequence:
                return "Ready"
            if not selected_rows:
                return f"Playing entire sequence ({len(chord_sequence)} segments)..."
            return f"Playing {len(selected_rows)} selected segments..."

        self.app.clientside_callback(
            '''
            function(n_clicks, selected_rows, chord_sequence, track_audio_data) {
                if (!n_clicks || !chord_sequence || !track_audio_data) {
                    console.log("Invalid input data");
                    return window.dash_clientside.no_update;
                }

                let segments_to_play = [];
                if (!selected_rows || selected_rows.length === 0) {
                    segments_to_play = chord_sequence;
                    console.log("Playing entire sequence");
                } else {
                    segments_to_play = selected_rows.map(row => chord_sequence[row]);
                    console.log("Playing selected segments");
                }

                const segmentsByStart = {};
                segments_to_play.forEach(segment => {
                    const start = parseFloat(segment.start);
                    if (!segmentsByStart[start]) {
                        segmentsByStart[start] = [];
                    }
                    segmentsByStart[start].push(segment);
                });

                const startTimes = Object.keys(segmentsByStart)
                    .map(time => parseFloat(time))
                    .sort((a, b) => a - b);

                let currentIndex = 0;

                function playNextGroup() {
                    if (currentIndex >= startTimes.length) {
                        console.log("Finished playing sequence");
                        return;
                    }

                    const startTime = startTimes[currentIndex];
                    const segments = segmentsByStart[startTime];
                    const maxDuration = Math.max(...segments.map(seg => parseFloat(seg.duration)));

                    const audioElements = [];
                    segments.forEach(segment => {
                        const trackName = segment.track.toLowerCase();
                        const audio = document.getElementById(`track-audio-${trackName}`);
                        if (audio) {
                            audioElements.push({ audio, segment });
                        } else {
                            console.error(`Audio element not found for track: ${trackName}`);
                        }
                    });

                    audioElements.forEach(({ audio, segment }) => {
                        try {
                            audio.currentTime = parseFloat(segment.start);
                            audio.play().then(() => {
                                console.log(`Playing ${segment.chord} from ${segment.start}s for ${segment.duration}s on ${segment.track}`);
                            }).catch(error => {
                                console.error(`Error playing audio: ${error}`);
                            });
                        } catch (error) {
                            console.error(`Playback error: ${error}`);
                        }
                    });

                    setTimeout(() => {
                        audioElements.forEach(({ audio }) => audio.pause());
                        currentIndex++;
                        playNextGroup();
                    }, maxDuration * 1000);
                }

                playNextGroup();
                return window.dash_clientside.no_update;
            }
            ''',
            Output('play-sequence-button', 'n_clicks'),
            Input('play-sequence-button', 'n_clicks'),
            State('chord-sequence-table', 'selected_rows'),
            State('chord-sequence-store', 'data'),
            State('track-audio-store', 'data'),
            prevent_initial_call=True
        )

        @self.app.callback(
            [Output('play-state', 'data'), Output('auto-update', 'disabled')],
            Input('play-pause-button', 'n_clicks'),
            State('play-state', 'data'),
            State('playback-position', 'value'),
            prevent_initial_call=True
        )
        def toggle_play_state(n_clicks, play_state, position):
            is_playing = play_state['is_playing']
            new_is_playing = not is_playing
            if new_is_playing and position >= 0.999:
                position = 0
            return {'is_playing': new_is_playing}, not new_is_playing

        @self.app.callback(
            [
                Output('chroma-figure', 'figure'),
                Output('waveform-figure', 'figure'),
                Output('chord-figure', 'figure'),
                Output('chord-stat-figure', 'figure'),
                Output('tempo-figure', 'figure'),
                Output('piano-roll-figure', 'figure'),
                Output('last-position', 'data')
            ],
            [
                Input('playback-position', 'value'),
                Input('auto-update', 'n_intervals'),
                Input('_current-audio-position', 'value'),
                Input('piano-roll-track-selector', 'value'),
                Input('visualisation-tabs', 'value')
            ],
            [
                State('tracks-store', 'data'),
                State('features-store', 'data'),
                State('duration-store', 'data'),
                State('play-state', 'data'),
                State('last-position', 'data'),
                State('tempo-file-store', 'data'),
            ]
        )
        def update_visualisation(slider_position, n_intervals, audio_position, selected_tracks, active_tab,
                                tracks_info_data, features_data, duration, play_state, last_position, tempo_file_paths):
            if not tracks_info_data or not features_data or not duration:
                return [go.Figure()] * 6 + [last_position]

            position = audio_position if audio_position is not None else slider_position
            features = np.array(features_data)

            chroma_fig = no_update
            waveform_fig = no_update
            chord_fig = no_update
            chord_stat_fig = no_update
            tempo_fig = no_update
            piano_roll_fig = no_update

            if active_tab == 'tab-0':
                chroma_fig = Chroma_bar_visualiser(features, position)
            elif active_tab == 'tab-1':
                waveform_fig = multi_chroma_visualizer(features=features, position=position, tracks=self.tracks)
            elif active_tab == 'tab-2':
                chord_fig = chord_visualizer(features, position, duration)
            elif active_tab == 'tab-3':
                n_frames = features.shape[1]
                current_idx = position_idx(position, n_frames)
                current_chord_stat = chord_statistics(features[:, :current_idx, :])
                chord_stat_fig = chord_stat_bar_figure(current_chord_stat)
            elif active_tab == 'tab-4':
                tempo_fig = create_tempo_with_beats_figure(tempo_file_paths, position, duration) if tempo_file_paths else go.Figure()
            elif active_tab == 'tab-5':
                piano_roll_fig = piano_roll_visualizer(self.piano_roll_data, position, duration, selected_tracks=selected_tracks)

            return [chroma_fig, waveform_fig, chord_fig, chord_stat_fig, tempo_fig, piano_roll_fig, position]

        @self.app.callback(
            Output('playback-position', 'value', allow_duplicate=True),
            [Input('auto-update', 'n_intervals'), Input('_current-audio-position', 'value')],
            [State('playback-position', 'value'), State('duration-store', 'data'), State('play-state', 'data')],
            prevent_initial_call=True
        )
        def update_playback_position(n_intervals, audio_pos, slider_pos, duration, play_state):
            if not duration:
                raise PreventUpdate
            pos = audio_pos if audio_pos is not None else slider_pos
            if play_state['is_playing'] and n_intervals > 0:
                pos = min(1, pos + (0.2 / duration))
            return pos

        @self.app.callback(
            Output('chord-sequence-visualization', 'figure'),
            Input('chord-sequence-store', 'data'),
            State('duration-store', 'data'),
            prevent_initial_call=True
        )
        def update_chord_sequence_visualization(chord_sequence, duration):
            if not chord_sequence:
                return go.Figure()
            return self.chord_sequence_visualizer(chord_sequence, duration)

def main():
    app = CustomWebApp(verbose=True)
    app.setup_app()
    try:
        app.run(debug=True, port=8080)
    finally:
        app.cleanup()

if __name__ == "__main__":
    main()

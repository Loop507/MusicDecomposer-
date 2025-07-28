import streamlit as st
import numpy as np
import librosa
import librosa.display
import soundfile as sf
import tempfile
import os
import random
from scipy import signal
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import io
import traceback
import gc

# Importa psutil solo se disponibile
try:
    import psutil
except ImportError:
    psutil = None

# ===============================
# 1. PROCESSING IN CHUNKS
# ===============================

def process_audio_in_chunks(audio, sr, processing_func, params, chunk_duration=15.0):
    """Elabora audio lunghi a pezzi per evitare problemi di memoria"""
    chunk_samples = int(chunk_duration * sr)
    if len(audio) <= chunk_samples:
        return processing_func(audio, sr, params)
    
    overlap_samples = int(0.3 * sr)
    chunks_processed = []
    for start in range(0, len(audio), chunk_samples - overlap_samples):
        end = min(start + chunk_samples, len(audio))
        chunk = audio[start:end]
        if chunk.size > 0:
            processed_chunk = processing_func(chunk, sr, params)
            if processed_chunk.size > 0:
                chunks_processed.append(processed_chunk)
            gc.collect()
    if not chunks_processed:
        return np.array([])
    return merge_chunks_with_crossfade(chunks_processed, sr)

def merge_chunks_with_crossfade(chunks, sr, fade_duration=0.1):
    """Unisce chunks con crossfade"""
    if len(chunks) == 1:
        return chunks[0]
    fade_samples = int(fade_duration * sr)
    result = chunks[0].copy()
    for chunk in chunks[1:]:
        if chunk.size == 0:
            continue
        crossfade_len = min(fade_samples, len(result), len(chunk))
        if crossfade_len > 0:
            fade_out = np.linspace(1, 0, crossfade_len)
            fade_in = np.linspace(0, 1, crossfade_len)
            result[-crossfade_len:] *= fade_out
            result[-crossfade_len:] += chunk[:crossfade_len] * fade_in
            if len(chunk) > crossfade_len:
                result = np.concatenate([result, chunk[crossfade_len:]])
        else:
            result = np.concatenate([result, chunk])
    return result

# ===============================
# 2. FUNZIONI OTTIMIZZATE (LEGGI E SICURE)
# ===============================

def optimized_cut_up_sonoro(audio, sr, params):
    fragment_size = params['fragment_size']
    randomness = params.get('cut_randomness', 0.7)
    reassembly = params.get('reassembly_style', 'random')
    if audio.size == 0:
        return np.array([])
    fragment_samples = max(int(fragment_size * sr), 1024)
    num_fragments = len(audio) // fragment_samples
    fragment_indices = [(i * fragment_samples, min((i + 1) * fragment_samples, len(audio))) 
                        for i in range(num_fragments)]
    processed_fragments = []
    for start, end in fragment_indices:
        fragment = audio[start:end].copy()
        if random.random() < randomness:
            variation = random.uniform(0.8, 1.2)
            new_length = int(len(fragment) * variation)
            if 0 < new_length != len(fragment):
                indices = np.linspace(0, len(fragment) - 1, new_length)
                fragment = np.interp(indices, np.arange(len(fragment)), fragment)
        if fragment.size > 0:
            processed_fragments.append(fragment)
    if not processed_fragments:
        return np.array([])
    if reassembly == 'random':
        random.shuffle(processed_fragments)
    elif reassembly == 'reverse':
        processed_fragments = [frag[::-1] for frag in processed_fragments if frag.size > 0]
        processed_fragments.reverse()
    elif reassembly == 'palindrome':
        valid = [f for f in processed_fragments if f.size > 0]
        processed_fragments = valid + [f for f in valid[::-1]]
    elif reassembly == 'spiral':
        new_fragments = []
        start_idx, end_idx = 0, len(processed_fragments) - 1
        while start_idx <= end_idx:
            if start_idx < len(processed_fragments) and processed_fragments[start_idx].size > 0 and len(new_fragments) % 2 == 0:
                new_fragments.append(processed_fragments[start_idx])
                start_idx += 1
            elif end_idx < len(processed_fragments) and processed_fragments[end_idx].size > 0 and len(new_fragments) % 2 != 0:
                new_fragments.append(processed_fragments[end_idx])
                end_idx -= 1
            else:
                if len(new_fragments) % 2 == 0:
                    start_idx += 1
                else:
                    end_idx += 1
        processed_fragments = new_fragments
    total_length = sum(len(f) for f in processed_fragments)
    if total_length == 0:
        return np.array([])
    result = np.empty(total_length, dtype=audio.dtype)
    pos = 0
    for frag in processed_fragments:
        if frag.size > 0:
            result[pos:pos + len(frag)] = frag
            pos += len(frag)
    return result[:pos]

def optimized_musique_concrete(audio, sr, params):
    grain_size = max(params.get('grain_size', 0.1), 0.05)
    texture_density = min(params.get('texture_density', 1.0), 1.5)
    chaos_level = params['chaos_level']
    if audio.size == 0:
        return np.array([])
    grain_samples = max(int(grain_size * sr), 512)
    max_grains = min(300, len(audio) // grain_samples or 1)
    positions = np.linspace(0, len(audio) - grain_samples, max_grains).astype(int)
    grains = []
    for pos in positions:
        grain = audio[pos:pos + grain_samples].copy()
        if grain.size == 0:
            continue
        if random.random() < min(chaos_level / 4.0, 0.3):
            if random.random() < 0.5:
                grain = grain[::-1]
            else:
                grain *= random.uniform(0.5, 1.5)
        if grain.size > 0:
            grains.append(grain)
    if not grains:
        return np.array([])
    max_output_length = int(len(audio) * min(1.5, 1 + texture_density * 0.3))
    if max_output_length <= 0:
        return np.array([])
    result = np.zeros(max_output_length, dtype=audio.dtype)
    num_grains_to_use = min(len(grains), int(len(grains) * texture_density))
    selected_grains = random.sample(grains, num_grains_to_use) if num_grains_to_use > 0 else []
    for grain in selected_grains:
        if grain.size == 0:
            continue
        if grain.size > max_output_length:
            grain = grain[:max_output_length]
        if grain.size >= max_output_length:
            continue
        max_start = max_output_length - grain.size
        if max_start >= 0:
            start_pos = random.randint(0, max_start)
            result[start_pos:start_pos + grain.size] += grain * random.uniform(0.2, 0.8)
    max_val = np.max(np.abs(result))
    if max_val > 0:
        result = result / max_val * 0.8
    return result

def ultra_optimized_random_chaos(audio, sr, params):
    chaos_level = min(params['chaos_level'], 1.5)
    if audio.size == 0:
        return np.array([])
    processed_audio = audio.copy()
    # Reverse
    if random.random() < 0.3 * chaos_level and len(processed_audio) > int(sr * 1):
        length = min(int(sr * 1), len(processed_audio) // 4)
        start = random.randint(0, len(processed_audio) - length)
        processed_audio[start:start + length] = processed_audio[start:start + length][::-1]
    # Volume
    if random.random() < 0.4 * chaos_level:
        num_sections = min(8, max(1, int(chaos_level * 3)))
        section_length = len(processed_audio) // num_sections
        if section_length > 0:
            for i in range(num_sections):
                start = i * section_length
                end = min(start + section_length, len(processed_audio))
                if end > start:
                    processed_audio[start:end] *= random.uniform(0.3, 1.5)
    # Noise
    if random.random() < 0.2 * chaos_level and processed_audio.size > 0:
        noise = np.random.normal(0, min(0.05, chaos_level * 0.02), len(processed_audio))
        processed_audio += noise
    # Shuffle
    if random.random() < 0.5 * chaos_level:
        fragment_length = max(int(sr * 0.5), 1024)
        num_fragments = min(15, len(processed_audio) // fragment_length or 1)
        fragments = [processed_audio[i*fragment_length:(i+1)*fragment_length] 
                     for i in range(num_fragments) if (i+1)*fragment_length <= len(processed_audio)]
        if fragments:
            random.shuffle(fragments)
            processed_audio = np.concatenate(fragments)
    if processed_audio.size > 0:
        max_val = np.max(np.abs(processed_audio))
        if max_val > 0:
            processed_audio = processed_audio / max_val * 0.95
    return processed_audio

def remix_destrutturato(audio, sr, params):
    fragment_size = params['fragment_size']
    beat_preservation = params.get('beat_preservation', 0.4)
    melody_fragmentation = params.get('melody_fragmentation', 1.5)
    if audio.size == 0:
        return np.array([])
    fragment_samples = int(fragment_size * sr)
    if fragment_samples <= 0:
        return audio
    cut_points = []
    if beat_preservation > 0.5:
        try:
            _, beats = librosa.beat.beat_track(y=audio, sr=sr)
            beat_times = librosa.frames_to_time(beats, sr=sr)
            cut_points = [int(t * sr) for t in beat_times if t * sr < len(audio)]
        except:
            pass
    if len(cut_points) == 0:
        num_cuts = int(len(audio) / fragment_samples)
        if num_cuts > 0:
            step = len(audio) // num_cuts
            cut_points = [i * step for i in range(num_cuts)]
    if len(cut_points) == 0:
        return audio
    if 0 not in cut_points:
        cut_points.insert(0, 0)
    if len(audio) not in cut_points:
        cut_points.append(len(audio))
    cut_points = sorted(set(cut_points))
    fragments = []
    for i in range(len(cut_points) - 1):
        start, end = cut_points[i], cut_points[i+1]
        if end <= start:
            continue
        fragment = audio[start:end].copy()
        if fragment.size == 0:
            continue
        if random.random() < melody_fragmentation / 3.0:
            shift = random.uniform(-7, 7)
            fragment = safe_pitch_shift(fragment, sr, shift)
        if fragment.size > 0 and random.random() < melody_fragmentation / 3.0:
            rate = random.uniform(0.7, 1.4)
            fragment = safe_time_stretch(fragment, rate)
        if fragment.size > 0:
            fragments.append(fragment)
    if not fragments:
        return np.array([])
    if beat_preservation > 0.3:
        preserve_count = int(len(fragments) * beat_preservation)
        preserve_indices = random.sample(range(len(fragments)), min(preserve_count, len(fragments)))
        ordered = [np.array([])] * len(fragments)
        for i in range(len(fragments)):
            if i in preserve_indices:
                ordered[i] = fragments[i]
            else:
                try:
                    remaining = [f for j, f in enumerate(fragments) if j not in preserve_indices]
                    random.shuffle(remaining)
                    ordered[i] = remaining.pop()
                except:
                    pass
        fragments = [f for f in ordered if f.size > 0]
    else:
        random.shuffle(fragments)
    if not fragments:
        return np.array([])
    result = fragments[0]
    fade_samples = int(0.05 * sr)
    for frag in fragments[1:]:
        if frag.size == 0:
            continue
        cross = min(fade_samples, len(result), len(frag))
        if cross > 0:
            result[-cross:] *= np.linspace(1, 0, cross)
            result[-cross:] += frag[:cross] * np.linspace(0, 1, cross)
            result = np.concatenate([result, frag[cross:]])
        else:
            result = np.concatenate([result, frag])
    return result

def decostruzione_postmoderna(audio, sr, params):
    irony_level = params.get('irony_level', 0.5)
    context_shift = params.get('context_shift', 0.6)
    fragment_size = params['fragment_size']
    if audio.size == 0:
        return np.array([])
    fragment_samples = int(fragment_size * sr)
    if fragment_samples <= 0:
        return audio
    num_fragments = min(10, len(audio) // fragment_samples or 1)
    fragments = [audio[i*fragment_samples:(i+1)*fragment_samples] for i in range(num_fragments)]
    processed = []
    for frag in fragments:
        if frag.size == 0:
            continue
        if random.random() < irony_level * 0.4:
            if random.random() < 0.

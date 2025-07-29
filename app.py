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

# Configurazione pagina
st.set_page_page_config(
    page_title="MusicDecomposer by loop507",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Titolo principale
st.markdown("""
<div style='text-align: center; padding: 20px;'>
    <h1> MusicDecomposer <span style='font-size:0.6em; color: #666;'>by <span style='font-size:0.8em;'>loop507</span></span></h1>
    <p style='font-size: 1.2em; color: #888;'>Scomponi e Ricomponi Brani in Arte Sonora Sperimentale</p>
    <p style='font-style: italic;'>Inspired by Musique Concrète, Plunderphonics & Cut-up Technique</p>
</div>
""", unsafe_allow_html=True)

# Upload file
uploaded_file = st.file_uploader(
    "Carica il tuo brano da decomporre",
    type=["mp3", "wav", "m4a", "flac", "ogg"],
    help="Supporta MP3, WAV, M4A, FLAC, OGG"
)

# PARAMETRI FISSI PER MASSIMA ELABORAZIONE (NON PIÙ DIPENDENTI DALL'INTENSITÀ)
FIXED_PARAMS = {
    'cut_up_sonoro': {
        'fragment_size': 0.8,
        'chaos_level': 0.9,
        'structure_preservation': 0.2,
        'cut_randomness': 0.8,
        'reassembly_style': 'spiral'
    },
    'remix_destrutturato': {
        'fragment_size': 0.6,
        'chaos_level': 0.8,
        'structure_preservation': 0.3,
        'beat_preservation': 0.3,
        'melody_fragmentation': 1.8
    },
    'musique_concrete': {
        'fragment_size': 0.5,
        'chaos_level': 0.8,
        'structure_preservation': 0.2,
        'grain_size': 0.1,
        'texture_density': 1.5
    },
    'decostruzione_postmoderna': {
        'fragment_size': 1.0,
        'chaos_level': 0.7,
        'structure_preservation': 0.3,
        'irony_level': 0.7,
        'context_shift': 0.8
    },
    'decomposizione_creativa': {
        'fragment_size': 0.8,
        'chaos_level': 0.8,
        'structure_preservation': 0.2,
        'discontinuity': 1.2,
        'emotional_shift': 1.0
    },
    'random_chaos': {
        'fragment_size': 1.0,
        'chaos_level': 1.0,
        'structure_preservation': 0.1
    }
}

def analyze_audio_structure(audio, sr):
    if audio.size == 0:
        return {
            'tempo': 0, 'beats': np.array([]), 'chroma': np.array([]),
            'mfcc': np.array([]), 'spectral_centroids': np.array([]),
            'onset_times': np.array([]), 'onset_frames': np.array([])
        }

    max_analysis_length = min(len(audio), sr * 60)
    analysis_audio = audio[:max_analysis_length]

    tempo = 0
    beats = np.array([])
    try:
        tempo, beats = librosa.beat.beat_track(y=analysis_audio, sr=sr, hop_length=1024)
        if beats.ndim > 1:
            beats = beats.flatten()
    except Exception as e:
        tempo = 0
        beats = np.array([])

    chroma = np.array([])
    mfcc = np.array([])
    spectral_centroids = np.array([])
    onset_frames = np.array([])
    onset_times = np.array([])

    try:
        chroma = librosa.feature.chroma_stft(y=analysis_audio, sr=sr, hop_length=2048)
        mfcc = librosa.feature.mfcc(y=analysis_audio, sr=sr, n_mfcc=13, hop_length=2048)
        spectral_centroids = librosa.feature.spectral_centroid(y=analysis_audio, sr=sr, hop_length=2048)[0]
        onset_frames = librosa.onset.onset_detect(y=analysis_audio, sr=sr, hop_length=1024)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=1024)
    except Exception as e:
        pass # Non blocca l'applicazione se l'estrazione delle feature fallisce parzialmente

    return {
        'tempo': tempo,
        'beats': beats,
        'chroma': chroma,
        'mfcc': mfcc,
        'spectral_centroids': spectral_centroids,
        'onset_times': onset_times,
        'onset_frames': onset_frames
    }

def safe_pitch_shift(audio, sr, n_steps):
    try:
        if audio.size == 0:
            return np.array([])
        
        # librosa.effects.pitch_shift richiede audio abbastanza lungo
        if audio.size < 2048: # O un valore appropriato per hop_length * 2 o più
             return audio # Non shiftare audio troppo corto

        n_steps = np.clip(n_steps, -12, 12)
        hop_length = 512 if len(audio) < sr * 30 else 1024
        
        # Rimosso n_length=len(audio) come richiesto per risolvere l'errore
        result = librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps, hop_length=hop_length)
        return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception as e:
        st.warning(f"Pitch shift fallito: {e}")
        return audio

def safe_time_stretch(audio, rate):
    try:
        if audio.size == 0:
            return np.array([])
        
        # librosa.effects.time_stretch richiede audio abbastanza lungo
        if audio.size < 2048: # O un valore appropriato per hop_length * 2 o più
            return audio # Non stretchare audio troppo corto

        rate = np.clip(rate, 0.25, 4.0)
        hop_length = 512 if len(audio) < 44100 * 30 else 1024
        
        result = librosa.effects.time_stretch(audio, rate=rate, hop_length=hop_length)
        return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception as e:
        st.warning(f"Time stretch fallito: {e}")
        return audio

def cut_up_sonoro(audio, sr, params):
    fragment_size = params['fragment_size']
    randomness = params.get('cut_randomness', 0.7)
    reassembly = params.get('reassembly_style', 'random')

    if audio.size == 0:
        return np.array([])

    fragment_samples = int(fragment_size * sr)
    if fragment_samples <= 0:
        return audio

    num_possible_fragments = len(audio) // fragment_samples
    max_fragments_to_process = min(num_possible_fragments, 500) 

    fragments = []
    if max_fragments_to_process > 0:
        step = max(1, len(audio) // max_fragments_to_process)
    else:
        step = fragment_samples

    for i in range(0, len(audio) - fragment_samples + 1, step):
        if len(fragments) >= max_fragments_to_process:
            break
        
        fragment = audio[i:i + fragment_samples]

        if fragment.size == 0:
            continue

        if random.random() < randomness:
            variation = random.uniform(0.7, 1.3)
            new_size = int(fragment.size * variation)

            if new_size <= 0:
                continue

            if fragment.size > 0:
                if new_size < fragment.size:
                    fragment = fragment[:new_size]
                else:
                    indices = np.linspace(0, fragment.size - 1, new_size)
                    fragment = np.interp(indices, np.arange(fragment.size), fragment)

        if fragment.size > 0:
            fragments.append(fragment)

    if len(fragments) == 0:
        return np.array([])

    if reassembly == 'random':
        random.shuffle(fragments)
    elif reassembly == 'reverse':
        fragments = [frag[::-1] for frag in fragments if frag.size > 0]
        fragments = fragments[::-1]
    elif reassembly == 'palindrome':
        valid_fragments = [frag for frag in fragments if frag.size > 0]
        fragments = valid_fragments + [frag for frag in valid_fragments[::-1]]
    elif reassembly == 'spiral':
        new_fragments = []
        start, end = 0, len(fragments) - 1
        while start <= end and len(new_fragments) < len(fragments):
            if start < len(fragments) and fragments[start].size > 0:
                new_fragments.append(fragments[start])
                start += 1
            if end >= 0 and end != start - 1 and fragments[end].size > 0:
                new_fragments.append(fragments[end])
                end -= 1
        fragments = new_fragments

    if len(fragments) > 0:
        result = np.concatenate(fragments)
    else:
        result = np.array([])

    return result

def remix_destrutturato(audio, sr, params):
    fragment_size = params['fragment_size']
    beat_preservation = params.get('beat_preservation', 0.4)
    melody_fragmentation = params.get('melody_fragmentation', 1.5)

    if audio.size == 0:
        return np.array([])

    structure = analyze_audio_structure(audio, sr)
    beats = structure['beats']

    fragment_samples = int(fragment_size * sr)
    if fragment_samples <= 0:
        return audio

    max_cuts_based_on_length = len(audio) // fragment_samples
    max_cuts = min(max_cuts_based_on_length, 200)

    cut_points = []
    if beat_preservation > 0.5 and beats.size > 0:
        beat_times = librosa.frames_to_time(beats, sr=sr)
        cut_points = [int(t * sr) for t in beat_times[:int(max_cuts)] if t * sr < len(audio)]

    if len(cut_points) == 0 or beat_preservation <= 0.5:
        num_cuts = min(int(max_cuts), int(len(audio) / fragment_samples))
        if num_cuts > 0 and len(audio) > 0:
            if (len(audio) // fragment_samples) > 0:
                cut_points = sorted(random.sample(range(0, len(audio), fragment_samples), min(num_cuts, (len(audio) // fragment_samples))))
            else:
                cut_points = []

    if len(cut_points) == 0:
        return audio

    cut_points = sorted(list(set([0] + cut_points + [len(audio)])))

    fragments = []
    for i in range(len(cut_points) - 1):
        start = cut_points[i]
        end = cut_points[i + 1]
        if end <= start:
            continue
        fragment = audio[start:end]

        if fragment.size > 0:
            if random.random() < melody_fragmentation / 2.0:
                try:
                    shift_steps = random.uniform(-6, 6)
                    fragment = safe_pitch_shift(fragment, sr, shift_steps)
                except Exception:
                    fragment = np.array([])

            if fragment.size > 0 and random.random() < melody_fragmentation / 2.0:
                try:
                    stretch_factor = random.uniform(0.7, 1.5)
                    fragment = safe_time_stretch(fragment, stretch_factor)
                except Exception:
                    fragment = np.array([])

        if fragment.size > 0:
            fragments.append(fragment)

    if len(fragments) == 0:
        return np.array([])

    if beat_preservation > 0.3 and len(fragments) > 1:
        preserve_count = int(len(fragments) * beat_preservation)
        preserve_indices = random.sample(range(len(fragments)), min(preserve_count, len(fragments)))
        
        non_preserved = [fragments[i] for i in range(len(fragments)) if i not in preserve_indices]
        random.shuffle(non_preserved)
        
        result_fragments = []
        non_preserved_iter = iter(non_preserved)
        
        for i in range(len(fragments)):
            if i in preserve_indices:
                result_fragments.append(fragments[i])
            else:
                try:
                    result_fragments.append(next(non_preserved_iter))
                except StopIteration:
                    break
        
        fragments = result_fragments
    else:
        random.shuffle(fragments)

    if len(fragments) == 0:
        return np.array([])

    result = fragments[0]
    fade_samples = int(0.02 * sr)

    for fragment in fragments[1:]:
        if result.size == 0:
            result = fragment
            continue
        if fragment.size == 0:
            continue

        current_fade_samples = min(fade_samples, result.size // 4, fragment.size // 4)
        
        if current_fade_samples > 0:
            result = np.concatenate([result[:-current_fade_samples], fragment])
        else:
            result = np.concatenate([result, fragment])

    return result

def musique_concrete(audio, sr, params):
    grain_size = params.get('grain_size', 0.1)
    texture_density = params.get('texture_density', 1.0)
    chaos_level = params['chaos_level']

    if audio.size == 0:
        return np.array([])

    grain_samples = int(grain_size * sr)
    if grain_samples <= 0:
        return audio

    max_grains_to_extract = 500
    
    max_grains_based_on_length = len(audio) // max(1, grain_samples // 4) 
    max_grains = min(max_grains_based_on_length, max_grains_to_extract) 

    step = max(grain_samples // 4, 1) 
    if max_grains > 0 and len(audio) > 0:
        step = max(grain_samples // 4, len(audio) // max_grains) 
    
    grains = []
    for i in range(0, len(audio) - grain_samples + 1, step):
        if len(grains) >= max_grains:
            break
        
        grain = audio[i:i + grain_samples]

        if grain.size == 0:
            continue

        try:
            window = signal.windows.hann(len(grain))
            grain = grain * window
        except Exception:
            pass

        if random.random() < chaos_level / 2.0:
            grain = grain[::-1]

        if grain.size > 0 and random.random() < chaos_level / 2.0:
            try:
                shift = random.uniform(-12, 12)
                grain = safe_pitch_shift(grain, sr, shift)
            except Exception:
                grain = np.array([])

        if grain.size > 0 and random.random() < chaos_level / 2.0:
            try:
                stretch = random.uniform(0.25, 4.0)
                grain = safe_time_stretch(grain, stretch)
            except Exception:
                grain = np.array([])

        if grain.size > 0:
            grains.append(grain)

    if len(grains) == 0:
        return np.array([])

    max_output_grains = 250
    num_grains_output = min(int(len(grains) * texture_density), max_output_grains) 
    
    if num_grains_output <= 0:
        return np.array([])

    if num_grains_output < len(grains):
        grains = random.sample(grains, num_grains_output)

    max_length_samples = min(int(len(audio) * 1.5), sr * 120)
    result = np.zeros(max_length_samples, dtype=audio.dtype)

    for grain in grains:
        if grain.size == 0:
            continue

        if grain.size < max_length_samples:
            max_start_pos = max_length_samples - grain.size
            if max_start_pos < 0:
                grain = grain[:max_length_samples]
                max_start_pos = 0
            
            start_pos = random.randint(0, max(0, max_start_pos))
            end_pos = start_pos + grain.size
            
            grain_to_add = grain[:min(grain.size, max_length_samples - start_pos)]
            
            if grain_to_add.size > 0:
                result[start_pos : start_pos + grain_to_add.size] += grain_to_add * random.uniform(0.5, 0.8)

    if result.size > 0:
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        max_val = np.max(np.abs(result))
        if max_val > 0:
            result = result / max_val * 0.95

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

    max_fragments_based_on_length = len(audio) // fragment_samples
    max_fragments = min(max_fragments_based_on_length, 100) 
    
    fragments = []
    fragment_types = []

    step = max(fragment_samples, len(audio) // max_fragments) if max_fragments > 0 else fragment_samples
    
    for i in range(0, len(audio) - fragment_samples + 1, step):
        if len(fragments) >= max_fragments:
            break
            
        fragment = audio[i:i + fragment_samples]
        if fragment.size == 0:
            continue

        energy = np.sqrt(np.mean(fragment**2))
        
        mean_sample_energy = 0
        if len(audio) > 0 and fragment_samples > 0: 
            sample_count = min(10, (len(audio) // fragment_samples) if fragment_samples > 0 else 1)
            if sample_count > 0:
                sample_indices = np.linspace(0, len(audio) - fragment_samples, sample_count, dtype=int)
                sample_energies = [np.sqrt(np.mean(audio[j:j+fragment_samples]**2)) for j in sample_indices if audio[j:j+fragment_samples].size > 0]
                mean_sample_energy = np.mean(sample_energies) if sample_energies else 0
        
        fragment_type = 'important' if energy > mean_sample_energy else 'random'
        
        fragments.append(fragment)
        fragment_types.append(fragment_type)

    if len(fragments) == 0:
        return audio

    processed_fragments = []
    
    for fragment, frag_type in zip(fragments, fragment_types):
        if fragment.size == 0:
            continue

        processed_frag = fragment.copy()

        if frag_type == 'important' and random.random() < irony_level * 0.5: 
            transform_choice = random.choice(['reverse', 'volume_down', 'fade_out', 'pitch_shift_subtle'])
            
            try:
                if transform_choice == 'reverse':
                    processed_frag = processed_frag[::-1]
                elif transform_choice == 'volume_down':
                    processed_frag = processed_frag * 0.2
                elif transform_choice == 'fade_out':
                    fade = np.linspace(1, 0.1, len(processed_frag))
                    processed_frag = processed_frag * fade
                elif transform_choice == 'pitch_shift_subtle':
                    processed_frag = safe_pitch_shift(processed_frag, sr, random.uniform(-2, 2))
            except Exception:
                processed_frag = fragment

        if processed_frag.size > 0 and random.random() < context_shift * 0.4: 
            effect_choice = random.choice(['fade_in', 'light_noise', 'time_stretch_subtle'])
            
            try:
                if effect_choice == 'fade_in':
                    fade = np.linspace(0.1, 1, len(processed_frag))
                    processed_frag = processed_frag * fade
                elif effect_choice == 'light_noise':
                    noise = np.random.normal(0, 0.01, len(processed_frag)) 
                    processed_frag = processed_frag + noise
                elif effect_choice == 'time_stretch_subtle':
                    processed_frag = safe_time_stretch(processed_frag, random.uniform(0.8, 1.2))
            except Exception:
                processed_frag = fragment

        if processed_frag.size > 0:
            processed_fragments.append(processed_frag)

    if len(processed_fragments) == 0:
        return audio

    random.shuffle(processed_fragments)
    final_fragments = processed_fragments

    if len(final_fragments) == 0:
        return audio
    
    if len(final_fragments) == 1:
        return final_fragments[0]

    result = final_fragments[0].copy()
    
    for next_fragment in final_fragments[1:]:
        if next_fragment.size == 0:
            continue

        crossfade_samples_desired = int(0.02 * sr) 
        current_crossfade_len = min(crossfade_samples_desired, result.size, next_fragment.size)
        
        if current_crossfade_len > 0:
            result_overlap = result[-current_crossfade_len:]
            fade_out = np.linspace(1, 0, current_crossfade_len)
            result[-current_crossfade_len:] = result_overlap * fade_out
            
            next_fragment_overlap = next_fragment[:current_crossfade_len]
            fade_in = np.linspace(0, 1, current_crossfade_len)
            next_fragment_faded_in = next_fragment_overlap * fade_in
            
            combined_overlap = result[-current_crossfade_len:] + next_fragment_faded_in
            
            result = np.concatenate([result[:-current_crossfade_len], combined_overlap, next_fragment[current_crossfade_len:]])
        else:
            result = np.concatenate([result, next_fragment])

    if len(result) > 0:
        max_val = np.max(np.abs(result))
        if max_val > 1.0:
            result = result / max_val * 0.95

    return result

def decomposizione_creativa(audio, sr, params):
    discontinuity = params.get('discontinuity', 1.0)
    emotional_shift = params.get('emotional_shift', 0.8)
    fragment_size = params['fragment_size']
    chaos_level = params['chaos_level']

    if audio.size == 0:
        return np.array([])

    max_analysis_length = min(len(audio), sr * 60)
    analysis_audio = audio[:max_analysis_length]
    
    try:
        onset_frames = librosa.onset.onset_detect(y=analysis_audio, sr=sr, hop_length=1024)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=1024)
    except Exception:
        onset_times = np.array([])

    fragment_samples = int(fragment_size * sr)
    if fragment_samples <= 0:
        return audio

    max_fragments_based_on_length = len(audio) // fragment_samples
    max_fragments = min(max_fragments_based_on_length, 150) 
    
    cut_points = sorted(list(set([0] + 
                                 [int(t * sr) for t in onset_times[:int(max_fragments * 0.5)] if int(t * sr) < len(audio)] + 
                                 [i * fragment_samples for i in range(0, max_fragments) if i * fragment_samples < len(audio)]))) 
    
    cut_points = [p for p in cut_points if p < len(audio)]
    if len(audio) not in cut_points:
        cut_points.append(len(audio))
    cut_points = sorted(list(set(cut_points)))

    processed_fragments = []
    for i in range(len(cut_points) - 1):
        start_sample = cut_points[i]
        end_sample = cut_points[i+1]
        
        if end_sample <= start_sample:
            continue

        fragment = audio[start_sample:end_sample].copy()
        if fragment.size == 0:
            continue

        if random.random() < discontinuity * 0.15: 
            if random.random() < 0.5:
                fragment = np.zeros_like(fragment) * random.uniform(0.05, 0.2) 
            else:
                continue 

        if random.random() < emotional_shift * 0.3: 
            if random.random() < 0.5:
                shift_steps = random.uniform(-12 * emotional_shift, 12 * emotional_shift) 
                fragment = safe_pitch_shift(fragment, sr, shift_steps)
            else:
                stretch_rate = random.uniform(1 - 0.4 * emotional_shift, 1 + 0.4 * emotional_shift) 
                fragment = safe_time_stretch(fragment, stretch_rate)
        
        if fragment.size > 0 and random.random() < chaos_level * 0.1: 
            fragment = fragment[::-1] 

        if fragment.size > 0:
            processed_fragments.append(fragment)

    if not processed_fragments:
        return np.array([])

    final_audio = np.concatenate(processed_fragments)

    if final_audio.size > 0:
        max_val = np.max(np.abs(final_audio))
        if max_val > 0:
            final_audio = final_audio / max_val * 0.95
    return final_audio

def random_chaos(audio, sr, params):
    """Random Chaos ottimizzato per performance e maggiore impatto"""
    chaos_level = params['chaos_level']

    if audio.size == 0:
        return np.array([])

    processed_audio = audio.copy()

    # Pitch Shift
    if random.random() < 0.6 * chaos_level:
        shift_steps = random.uniform(-12, 12) 
        processed_audio = safe_pitch_shift(processed_audio, sr, shift_steps)
        # Check if pitch shift resulted in empty array (e.g. if audio was too short)
        if processed_audio.size == 0: 
            return np.array([])

    # Time Stretch
    if random.random() < 0.6 * chaos_level:
        stretch_rate = random.uniform(0.25, 4.0) 
        processed_audio = safe_time_stretch(processed_audio, stretch_rate)
        # Check if time stretch resulted in empty array
        if processed_audio.size == 0: 
            return np.array([])

    # Inversione casuale
    if random.random() < 0.5 * chaos_level:
        if processed_audio.size > sr * 2 and random.random() < 0.8:
            segment_len = int(random.uniform(sr * 1.5, sr * 7)) 
            start_idx = random.randint(0, max(0, processed_audio.size - segment_len))
            end_idx = min(processed_audio.size, start_idx + segment_len)
            if end_idx > start_idx:
                processed_audio[start_idx:end_idx] = processed_audio[start_idx:end_idx][::-1]
        else:
            processed_audio = processed_audio[::-1]

    # Rumore
    if random.random() < 0.5 * chaos_level:
        if processed_audio.size > 0:
            noise_amplitude = random.uniform(0.02, 0.2) * chaos_level 
            noise = np.random.normal(0, noise_amplitude, processed_audio.size)
            processed_audio = processed_audio + noise

    # Frammentazione e riassemblaggio caotico
    if random.random() < 0.7 * chaos_level:
        if processed_audio.size > 0:
            max_fragment_len = int(sr * 3) 
            min_fragment_len = int(sr * 0.1) 
            chaos_fragments = []
            current_pos = 0
            max_fragments_in_chaos = 100 
            
            while current_pos < processed_audio.size and len(chaos_fragments) < max_fragments_in_chaos:
                frag_len = min(random.randint(min_fragment_len, max_fragment_len), 
                              processed_audio.size - current_pos)
                if frag_len <= 0: 
                    break
                fragment = processed_audio[current_pos : current_pos + frag_len]
                
                if random.random() < 0.3 * chaos_level:
                    fragment = fragment[::-1]
                if random.random() < 0.2 * chaos_level:
                    fragment = safe_pitch_shift(fragment, sr, random.uniform(-4, 4))
                
                chaos_fragments.append(fragment)
                current_pos += frag_len + int(sr * random.uniform(0, 0.5))

            if chaos_fragments:
                random.shuffle(chaos_fragments)
                result_chaos = chaos_fragments[0]
                for next_frag_chaos in chaos_fragments[1:]:
                    if result_chaos.size == 0:
                        result_chaos = next_frag_chaos
                        continue
                    if next_frag_chaos.size == 0:
                        continue
                    
                    crossfade_len_chaos = int(sr * random.uniform(0.005, 0.03)) 
                    crossfade_len_chaos = min(crossfade_len_chaos, result_chaos.size // 2, next_frag_chaos.size // 2)
                    
                    if crossfade_len_chaos > 0:
                        result_chaos = np.concatenate([result_chaos[:-crossfade_len_chaos], next_frag_chaos])
                    else:
                        result_chaos = np.concatenate([result_chaos, next_frag_chaos])
                processed_audio = result_chaos

    if processed_audio.size > 0:
        processed_audio = np.nan_to_num(processed_audio, nan=0.0, posinf=0.0, neginf=0.0)
        max_val = np.max(np.abs(processed_audio))
        if max_val > 0:
            processed_audio = processed_audio / max_val * 0.95
    
    return processed_audio

def apply_heavy_loop_decomposition(audio_segment, sr):
    """
    Applica una decomposizione molto aggressiva e casuale a un segmento audio.
    Questo è il nuovo "stile" per la creazione dei loop.
    """
    if audio_segment.size == 0:
        return np.array([])

    processed_segment = audio_segment.copy()

    # Livello di caos generale per questa decomposizione
    chaos_intensity = 1.0 # Sempre al massimo per "molto decomposto"

    # 1. Pitch Shift casuale e significativo
    if processed_segment.size > 0:
        shift_steps = random.uniform(-10, 10)
        processed_segment = safe_pitch_shift(processed_segment, sr, shift_steps)
        if processed_segment.size == 0: return np.array([])

    # 2. Time Stretch casuale e significativo
    if processed_segment.size > 0:
        stretch_rate = random.uniform(0.5, 2.0)
        processed_segment = safe_time_stretch(processed_segment, stretch_rate)
        if processed_segment.size == 0: return np.array([])

    # 3. Frammentazione e riassemblaggio iper-caotico
    # Assicurati che il segmento sia abbastanza lungo per la frammentazione
    min_segment_for_fragmentation = sr * 0.5 # Es. almeno 0.5 secondi
    if processed_segment.size > min_segment_for_fragmentation: 
        min_frag_len = int(sr * 0.05) # Frammenti molto piccoli
        max_frag_len = int(sr * 0.5)  # Frammenti di media dimensione
        
        fragments = []
        current_pos = 0
        max_fragments_to_cut = 50 # Limite per performance

        while current_pos < processed_segment.size and len(fragments) < max_fragments_to_cut:
            # Assicurati che frag_len non superi la dimensione rimanente
            frag_len = random.randint(min_frag_len, max_frag_len)
            if current_pos + frag_len > processed_segment.size:
                frag_len = processed_segment.size - current_pos
            
            if frag_len <= 0:
                break # Nessun frammento valido rimanente
            
            fragment = processed_segment[current_pos : current_pos + frag_len].copy()

            # Mini-manipolazioni sui singoli frammenti
            if fragment.size > 0:
                if random.random() < 0.3: # Inverti alcuni frammenti
                    fragment = fragment[::-1]
                if random.random() < 0.2: # Pitch shift leggero su alcuni frammenti
                    fragment = safe_pitch_shift(fragment, sr, random.uniform(-3, 3))
                if random.random() < 0.2: # Time stretch leggero su alcuni frammenti
                    fragment = safe_time_stretch(fragment, random.uniform(0.8, 1.2))
                
                if fragment.size > 0: # Aggiungi solo frammenti validi
                    fragments.append(fragment)
            
            current_pos += frag_len + int(sr * random.uniform(0.01, 0.1)) # Piccole pause/salti

        if fragments:
            random.shuffle(fragments) # Riassemblaggio completamente casuale
            processed_segment = np.concatenate(fragments)
        else:
            # Se la frammentazione non ha prodotto frammenti validi, non modificare il segmento da questo passaggio
            pass 
            
    # 4. Aggiunta di rumore impulsivo/glitch
    if processed_segment.size > 0 and random.random() < 0.7:
        num_impulses = random.randint(1, 5)
        for _ in range(num_impulses):
            if processed_segment.size > 0:
                impulse_pos = random.randint(0, processed_segment.size - 1)
                impulse_len = int(sr * random.uniform(0.001, 0.01))
                end_impulse = min(processed_segment.size, impulse_pos + impulse_len)
                if end_impulse > impulse_pos:
                    processed_segment[impulse_pos:end_impulse] += np.random.normal(0, random.uniform(0.1, 0.3), end_impulse - impulse_len) # Fixed len

    # 5. Normalizzazione finale
    if processed_segment.size > 0:
        processed_segment = np.nan_to_num(processed_segment, nan=0.0, posinf=0.0, neginf=0.0)
        max_val = np.max(np.abs(processed_segment))
        if max_val > 0:
            processed_segment = processed_segment / max_val * 0.9 # Normalizza un po' più basso per evitare clipping nel loop
        else: # Se max_val è 0 (audio silenzioso), assicurati che non sia NaN o Inf
            processed_segment = np.zeros_like(processed_segment)

    return processed_segment

def create_loop(original_audio, sr, loop_duration_sec, num_repetitions=3):
    """
    Crea un loop dall'audio originale. Il segmento di loop viene prima decomposto in modo massivo.
    """
    if original_audio.size == 0:
        st.error("❌ Impossibile creare il loop: l'audio originale è vuoto.")
        return np.array([])

    loop_segment_length_samples = int(loop_duration_sec * sr)

    if loop_segment_length_samples <= 0:
        st.error("❌ Durata del loop non valida.")
        return np.array([])

    if len(original_audio) < loop_segment_length_samples:
        st.error(f"❌ Audio originale troppo breve ({len(original_audio)/sr:.2f}s) per creare un loop di {loop_duration_sec:.2f}s.")
        return np.array([])
    
    # Seleziona il segmento iniziale dall'audio ORIGINALE della durata esatta desiderata
    segment_to_decompose = original_audio[:loop_segment_length_samples].copy()
    
    if segment_to_decompose.size == 0:
        st.error("❌ Il segmento selezionato per il loop è vuoto dopo il taglio iniziale.")
        return np.array([])

    # 2. Applica la decomposizione "molto decomposto" al segmento
    st.info(f"🌀 Applicando una decomposizione 'random chaos' al segmento di {loop_duration_sec:.2f} secondi...")
    decomposed_loop_segment = apply_heavy_loop_decomposition(segment_to_decompose, sr)
    
    if decomposed_loop_segment.size == 0:
        st.error("❌ La decomposizione del segmento di loop ha prodotto un audio vuoto o non riproducibile. Prova con una durata del loop diversa o un brano originale più lungo.")
        return np.array([])

    # 3. Assicurati che il segmento decomposto abbia la lunghezza esatta del loop desiderato
    if len(decomposed_loop_segment) > loop_segment_length_samples:
        decomposed_loop_segment = decomposed_loop_segment[:loop_segment_length_samples]
    elif len(decomposed_loop_segment) < loop_segment_length_samples:
        padding = np.zeros(loop_segment_length_samples - len(decomposed_loop_segment))
        decomposed_loop_segment = np.concatenate((decomposed_loop_segment, padding))

    # 4. Ripeti il segmento decomposto
    looped_audio = np.tile(decomposed_loop_segment, num_repetitions)

    st.success(f"✅ Loop generato! Segmento di {loop_duration_sec:.2f} secondi (dall'originale e decomposto in modo casuale), ripetuto {num_repetitions} volte. Durata totale loop: {len(looped_audio)/sr:.2f} secondi.")
    return looped_audio


# Logica principale per processare l'audio
if uploaded_file is not None:
    target_sr = 22050
    # Rimossa la sezione "⚙️ Impostazioni Caricamento Audio" dalla sidebar
    
    tmp_file_path = None # Inizializza per garantire che sia definito anche in caso di errore
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        audio, sr = librosa.load(tmp_file_path, sr=target_sr, duration=300)

        # Salva l'audio originale caricato in session_state per il loop
        st.session_state['original_audio'] = audio
        st.session_state['sr'] = sr

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Durata Caricata", f"{len(audio)/sr:.2f} sec")
        with col2:
            st.metric("Sample Rate Elaborazione", f"{sr} Hz")
        with col3:
            st.metric("Canali", "Mono")

        st.subheader("Audio Originale (Caricato)")
        # Rimosso sample_rate=sr
        st.audio(uploaded_file, format='audio/wav')

        st.markdown("---")
        st.subheader("🎛️ Seleziona il Metodo di Decomposizione per Massima Elaborazione")
        
        method_options = [
            "cut_up_sonoro",
            "remix_destrutturato", 
            "musique_concrete",
            "decostruzione_postmoderna",
            "decomposizione_creativa",
            "random_chaos"
        ]
        
        method_labels = {
            "cut_up_sonoro": "🎭 Cut-up Sonoro",
            "remix_destrutturato": "🔄 Remix Destrutturato",
            "musique_concrete": "🎵 Musique Concrète",
            "decostruzione_postmoderna": "🏛️ Decostruzione Postmoderna",
            "decomposizione_creativa": "🎨 Decomposizione Creativa",
            "random_chaos": "🌪️ Random Chaos"
        }
        
        selected_method = st.selectbox(
            "Scegli il Metodo:",
            method_options,
            format_func=lambda x: method_labels[x],
            help="Tutti i metodi ora applicano un'elaborazione approfondita."
        )

        if st.button("🎭 SCOMPONI E RICOMPONI (Massima Elaborazione)", type="primary", use_container_width=True):
            with st.spinner(f"Applicando {method_labels[selected_method]} con la massima elaborazione..."):
                
                params = FIXED_PARAMS[selected_method].copy()
                processed_audio = np.array([])
                
                if selected_method == "cut_up_sonoro":
                    processed_audio = cut_up_sonoro(audio, sr, params)
                elif selected_method == "remix_destrutturato":
                    processed_audio = remix_destrutturato(audio, sr, params)
                elif selected_method == "musique_concrete":
                    processed_audio = musique_concrete(audio, sr, params)
                elif selected_method == "decostruzione_postmoderna":
                    processed_audio = decostruzione_postmoderna(audio, sr, params)
                elif selected_method == "decomposizione_creativa":
                    processed_audio = decomposizione_creativa(audio, sr, params)
                elif selected_method == "random_chaos":
                    processed_audio = random_chaos(audio, sr, params)

                if processed_audio.size == 0:
                    st.error("❌ Elaborazione fallita - audio risultante vuoto. Prova un metodo diverso o un file audio differente.")
                else:
                    st.session_state['processed_audio_main'] = processed_audio # Rinominato per chiarezza
                    st.session_state['current_download_type'] = 'main_decomposed' # Imposta tipo download
                    
                    st.success("✅ Decomposizione completata!")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        new_duration = len(processed_audio)/sr
                        original_duration = len(audio)/sr
                        st.metric("Nuova Durata", f"{new_duration:.2f} sec", 
                                f"{(new_duration - original_duration):.2f} sec")
                    with col2:
                        original_rms = np.sqrt(np.mean(audio**2))
                        processed_rms = np.sqrt(np.mean(processed_audio**2))
                        st.metric("RMS Energy", f"{processed_rms:.4f}", 
                                f"{(processed_rms - original_rms):.4f}")
                    with col3:
                        min_len = min(len(processed_audio), len(audio))
                        if min_len > 0:
                            spec_orig = np.abs(np.fft.fft(audio[:min_len]))
                            spec_proc = np.abs(np.fft.fft(processed_audio[:min_len]))
                            spectral_diff = np.mean(np.abs(spec_proc - spec_orig))
                        else:
                            spectral_diff = 0.0
                        st.metric("Variazione Spettrale", f"{spectral_diff:.2e}")

                    st.subheader("Audio Decomposto e Ricomposto")
                    
                    # Salva l'audio processato in un file temporaneo per riproduzione e download
                    processed_tmp_path_for_playback = ""
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file_playback:
                        sf.write(tmp_file_playback.name, processed_audio, sr)
                        processed_tmp_path_for_playback = tmp_file_playback.name
                    
                    # Rimosso sample_rate=sr
                    st.audio(processed_tmp_path_for_playback, format='audio/wav') 
                    os.unlink(processed_tmp_path_for_playback) # Elimina il file temporaneo dopo la riproduzione
                    
        
        # === SEZIONE LOOP (mostrata solo se original_audio esiste) ===
        if 'original_audio' in st.session_state and st.session_state['original_audio'].size > 0:
            original_audio = st.session_state['original_audio']
            sr = st.session_state['sr']
            
            # Spostato tutta la sezione loop nella sidebar
            with st.sidebar:
                st.markdown("---")
                st.subheader("🔁 Crea un Loop Decomposto dall'Audio Originale")
                st.info("Questo loop sarà generato con una decomposizione caotica e imprevedibile, direttamente dal tuo brano originale.")

                loop_duration_option = st.radio(
                    "Durata del segmento per il Loop:",
                    (2.0, 4.0, 8.0, 15.0), # Opzioni in secondi
                    format_func=lambda x: f"{x:.1f} secondi",
                    key="loop_duration_radio"
                )
                num_repetitions = st.slider("Numero di ripetizioni del loop", 1, 10, 3, key="num_repetitions_slider_loop")

                looped_result = np.array([]) # Inizializza qui
                
                if st.button(f"🌀 Genera Loop Decomposto di {loop_duration_option:.1f} secondi", use_container_width=True, key="generate_loop_button"):
                    with st.spinner(f"Generando loop decomposto di {loop_duration_option:.1f} secondi dal brano originale..."):
                        looped_result = create_loop(original_audio, sr, loop_duration_option, num_repetitions)
                        if looped_result.size > 0:
                            st.success("Loop decomposto generato con successo!")
                            st.audio(looped_result, format='audio/wav', sample_rate=sr) # Qui sample_rate è ok perché è un numpy array
                            st.session_state['current_download_audio'] = looped_result # Salva per il download
                            st.session_state['current_download_type'] = 'loop'
                        else:
                            st.error("Impossibile generare il loop. Controlla i messaggi di avviso sopra.")
            
            # --- Sezione Download (rimane nella colonna principale, ma usa lo stato di sessione) ---
            final_audio_for_download = np.array([])
            filename = "decomposed_audio.wav"

            if 'current_download_type' in st.session_state:
                if st.session_state['current_download_type'] == 'main_decomposed' and 'processed_audio_main' in st.session_state:
                    final_audio_for_download = st.session_state['processed_audio_main']
                    method_names = {
                        "cut_up_sonoro": "CutUp",
                        "remix_destrutturato": "RemixDestrutturato", 
                        "musique_concrete": "MusiqueConcrete",
                        "decostruzione_postmoderna": "DecostruzionePostmoderna",
                        "decomposizione_creativa": "DecomposizioneCreativa",
                        "random_chaos": "RandomChaos"
                    }
                    current_selected_method = selected_method 
                    filename = f"{uploaded_file.name.split('.')[0]}_{method_names[current_selected_method]}_MAX.wav"
                elif st.session_state['current_download_type'] == 'loop' and 'current_download_audio' in st.session_state:
                    final_audio_for_download = st.session_state['current_download_audio']
                    filename = f"{uploaded_file.name.split('.')[0]}_LoopDecomposed_{loop_duration_option:.1f}s_x{num_repetitions}.wav"
            elif 'processed_audio_main' in st.session_state: # Default to main if no loop was generated
                 final_audio_for_download = st.session_state['processed_audio_main']
                 method_names = {
                    "cut_up_sonoro": "CutUp",
                    "remix_destrutturato": "RemixDestrutturato", 
                    "musique_concrete": "MusiqueConcrete",
                    "decostruzione_postmoderna": "DecostruzionePostmoderna",
                    "decomposizione_creativa": "DecomposizioneCreativa",
                    "random_chaos": "RandomChaos"
                }
                 current_selected_method = selected_method 
                 filename = f"{uploaded_file.name.split('.')[0]}_{method_names[current_selected_method]}_MAX.wav"


            if final_audio_for_download.size > 0:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as final_tmp_file:
                    sf.write(final_tmp_file.name, final_audio_for_download, sr)
                    final_download_path = final_tmp_file.name

                st.markdown("---")
                st.subheader("💾 Scarica il Tuo Audio Elaborato")
                with open(final_download_path, 'rb') as audio_file:
                    audio_bytes = audio_file.read()
                    st.download_button(
                        label="💾 Scarica Audio Elaborato",
                        data=audio_bytes,
                        file_name=filename,
                        mime="audio/wav",
                        use_container_width=True
                    )
                os.unlink(final_download_path) 
            
            # --- Fine Sezione Download ---


            # --- Sintesi Artistica e Grafici (usano l'audio scaricato più di recente) ---
            audio_for_charts = final_audio_for_download if final_audio_for_download.size > 0 else (st.session_state.get('processed_audio_main', np.array([])) if 'processed_audio_main' in st.session_state else np.array([]))
            
            if audio_for_charts.size > 0:
                st.markdown("---")
                st.subheader("Sintesi Artistica della Decomposizione")

                if 'current_download_type' in st.session_state and st.session_state['current_download_type'] == 'loop':
                    st.markdown("""
                    Il **Loop Decomposto** è stato creato applicando un processo di trasformazione molto aggressivo e casuale direttamente a un segmento dell'audio originale. Questo produce un frammento sonoro intrinsecamente caotico e imprevedibile, perfetto per texture ritmiche astratte o elementi drone.
                    """)
                else: # Mostra la descrizione del metodo principale
                    technique_descriptions = {
                        "cut_up_sonoro": """
                        Il metodo **"Cut-up Sonoro"** si inspira a una tecnica letteraria dove il testo viene frammentato e riassemblato. Il brano viene diviso in sezioni, che vengono poi **tagliate e riassemblate in un ordine casuale o predefinito** (come inversione o palindromo). Questo crea un effetto di collage sonoro, dove il significato originale è destrutturato per rivelare nuove connessioni e pattern imprevedibili.
                        """,
                        "remix_destrutturato": """
                        Il **"Remix Destrutturato"** mira a mantenere alcuni elementi riconoscibili del brano originale (come battiti o frammenti melodici), ma li **ricontestualizza in un nuovo arrangiamento**. Vengono applicate manipolazioni come pitch shift e time stretch ai frammenti, che poi vengono riorganizzati per creare un'esperienza d'ascolto familiare ma sorprendentemente nuova.
                        """,
                        "musique_concrete": """
                        La **"Musique Concrète"** si basa sui principi di manipolazione sonora. Questo metodo si concentra sulla **manipolazione di "grani" sonori** (piccolissimi frammenti dell'audio) attraverso tecniche come la sintesi granulare, l'inversione e il pitch/time shift. Il risultato è una texture sonora astratta che esplora le proprietà intrinseche del suono.
                        """,
                        "decostruzione_postmoderna": """
                        La **"Decostruzione Postmoderna"** applica un approccio critico al brano, **decostruendone il significato musicale originale** attraverso l'uso di ironia e spostamenti di contesto. Frammenti "importanti" vengono manipolati in modi inaspettati, e vengono introdotti elementi di rottura per provocare una riflessione critica sull'opera.
                        """,
                        "decomposizione_creativa": """
                        La **"Decomposizione Creativa"** si focalizza sulla creazione di **discontinuità e "shift emotivi"** intensi. Utilizzando l'analisi degli onset, il brano viene frammentato dinamicamente. I frammenti vengono trasformati con variazioni pronunciate di pitch e tempo per generare un'esperienza sonora ricca di espressività.
                        """,
                        "random_chaos": """
                        Il metodo **"Random Chaos"** è progettato per produrre **risultati altamente imprevedibili e sperimentali**. Ogni esecuzione è unica. Vengono applicate operazioni casuali come pitch shift e time stretch, inversioni casuali, aggiunta di rumore e frammentazione per esplorare i limiti della manipolazione audio.
                        """
                    }
                    st.markdown(f"""
                    Con il metodo **"{method_labels[selected_method]}"** applicato con la **massima elaborazione**, il brano originale è stato trasformato in un'opera sonora unica e profonda.
                    """)
                    st.markdown("---")
                    st.markdown("**Descrizione della Tecnica Applicata:**")
                    st.markdown(technique_descriptions[selected_method])

                st.markdown("---")
                st.markdown("### Riepilogo dei Cambiamenti Quantitativi:")

                analysis_text = f"""
                * **Durata:** L'audio originale (fino a 5 minuti) è stato trasformato in **{len(audio_for_charts)/sr:.2f} secondi** ({'allungamento' if len(audio_for_charts)/sr > (len(audio) / sr) else 'accorciamento'} di **{abs((len(audio_for_charts)/sr) - (len(audio) / sr)):.2f} secondi**).
                * **Sample Rate Elaborazione:** Il brano è stato elaborato a **{sr} Hz**.
                * **Energia RMS:** Variazione di **{np.sqrt(np.mean(audio_for_charts**2)) - np.sqrt(np.mean(audio**2)):.4f}** - il suono risultante è {'più forte' if np.sqrt(np.mean(audio_for_charts**2)) > np.sqrt(np.mean(audio**2)) else 'più debole' if np.sqrt(np.mean(audio_for_charts**2)) < np.sqrt(np.mean(audio**2)) else 'simile'} in volume medio.
                * **Variazione Spettrale:** **{np.mean(np.abs(np.abs(np.fft.fft(audio_for_charts[:min(len(audio_for_charts), len(audio))])) - np.abs(np.fft.fft(audio[:min(len(audio_for_charts), len(audio))])))):.2e}** - quantifica il cambiamento del "colore" e distribuzione delle frequenze rispetto all'originale.
                
                **Parametri Applicati** (Massima Elaborazione):
                """
                st.markdown(analysis_text)
                
                if 'current_download_type' in st.session_state and st.session_state['current_download_type'] == 'loop':
                    st.markdown(f"* Durata Loop: {loop_duration_option:.1f}s")
                    st.markdown(f"* Ripetizioni: {num_repetitions}")
                    st.markdown("* Stile Decomposizione Loop: Molto Decomposto (Random Chaos)")
                else:
                    params_to_display = FIXED_PARAMS[selected_method].copy()
                    for param_name, param_value in params_to_display.items():
                        if param_name == 'fragment_size':
                            st.markdown(f"* Dimensione Frammenti: {param_value}s")
                        elif param_name == 'chaos_level':
                            st.markdown(f"* Livello Chaos: {param_value}")
                        elif param_name == 'structure_preservation':
                            st.markdown(f"* Conservazione Struttura: {param_value}")
                        elif param_name == 'cut_randomness':
                            st.markdown(f"* Casualità Tagli: {param_value}")
                        elif param_name == 'reassembly_style':
                            st.markdown(f"* Stile Riassemblaggio: {param_value}")
                        elif param_name == 'beat_preservation':
                            st.markdown(f"* Conservazione Battito: {param_value*100:.0f}%")
                        elif param_name == 'melody_fragmentation':
                            st.markdown(f"* Frammentazione Melodica: {param_value}")
                        elif param_name == 'grain_size':
                            st.markdown(f"* Dimensione Grana: {param_value}s")
                        elif param_name == 'texture_density':
                            st.markdown(f"* Densità Texture: {param_value}")
                        elif param_name == 'irony_level':
                            st.markdown(f"* Livello Ironia: {param_value}")
                        elif param_name == 'context_shift':
                            st.markdown(f"* Spostamento Contesto: {param_value}")
                        elif param_name == 'discontinuity':
                            st.markdown(f"* Discontinuità: {param_value}")
                        elif param_name == 'emotional_shift':
                            st.markdown(f"* Shift Emotivo: {param_value}")


                with st.expander("📊 Confronto Forme d'Onda"):
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
                    
                    if audio.size > 0:
                        time_orig = np.linspace(0, len(audio)/sr, len(audio))
                        ax1.plot(time_orig, audio, color='blue', alpha=0.7)
                    ax1.set_title("Forma d'Onda Originale")
                    ax1.set_xlabel("Tempo (sec)")
                    ax1.set_ylabel("Ampiezza")
                    ax1.grid(True, alpha=0.3)
                    
                    if audio_for_charts.size > 0: 
                        time_proc = np.linspace(0, len(audio_for_charts)/sr, len(audio_for_charts))
                        ax2.plot(time_proc, audio_for_charts, color='red', alpha=0.7)
                    ax2.set_title(f"Forma d'Onda Elaborata ({'Loop Decomposto' if 'current_download_type' in st.session_state and st.session_state['current_download_type'] == 'loop' else method_labels[selected_method]})")
                    ax2.set_xlabel("Tempo (sec)")
                    ax2.set_ylabel("Ampiezza")
                    ax2.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)

                with st.expander("🎼 Analisi Spettrale"):
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                    
                    if audio.size > sr * 2:
                        D_orig = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
                        librosa.display.specshow(D_orig, y_axis='hz', x_axis='time', sr=sr, ax=ax1)
                        ax1.set_title("Spettrogramma Originale")
                        ax1.set_xlabel("Tempo (sec)")
                        ax1.set_ylabel("Frequenza (Hz)")
                    else:
                        ax1.set_title("Spettrogramma Originale (Audio troppo corto per analisi)")
                    
                    if audio_for_charts.size > sr * 2: 
                        D_proc = librosa.amplitude_to_db(np.abs(librosa.stft(audio_for_charts)), ref=np.max)
                        librosa.display.specshow(D_proc, y_axis='hz', x_axis='time', sr=sr, ax=ax2)
                        ax2.set_title("Spettrogramma Elaborato")
                        ax2.set_xlabel("Tempo (sec)")
                        ax2.set_ylabel("Frequenza (Hz)")
                    else:
                        ax2.set_title("Spettrogramma Elaborato (Audio troppo corto per analisi)")
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                
        # Pulisci il file temporaneo dell'audio caricato all'inizio
        try:
            if tmp_file_path and os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
        except Exception as e:
            st.error(f"Errore durante la pulizia del file originale: {e}")

    except Exception as e:
        st.error(f"❌ Errore nel processamento principale: {str(e)}")
        st.error(f"Dettagli: {traceback.format_exc()}")
        if 'tmp_file_path' in locals() and tmp_file_path and os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)


else:
    st.info("👆 Carica un file audio per iniziare la decomposizione")
    
    with st.expander("📖 Come usare MusicDecomposer"):
        st.markdown("""
        ### Modalità di utilizzo:

        1.  **Carica il tuo file audio** (MP3, WAV, M4A, FLAC, OGG).
        2.  L'audio verrà automaticamente processato a **22050 Hz** per ottimizzare le performance.
        3.  **SCOMPONI E RICOMPONI**:
            * **Seleziona il metodo di decomposizione** che preferisci (es. "Cut-up Sonoro"). Tutti i metodi applicheranno la **massima elaborazione**.
            * **Clicca "SCOMPONI E RICOMPONI"** per generare l'arte sonora.
        4.  **CREA LOOP DECOMPOSTO** (ora nella sidebar):
            * Indipendentemente dalla decomposizione principale, puoi creare un **loop molto decomposto** direttamente da un segmento del tuo brano originale.
            * **Scegli la durata in secondi** del segmento che vuoi looppare (es. 2, 4, 8, 15 secondi).
            * **Scegli il numero di ripetizioni**.
            * **Clicca "Genera Loop Decomposto"**. Il segmento verrà manipolato in modo caotico e poi ripetuto.

        ### Metodi disponibili (Massima Elaborazione):
        -   **🎭 Cut-up Sonoro**: Collage sonoro con frammenti riassemblati in modo intenso.
        -   **🔄 Remix Destrutturato**: Remix creativo che spinge i limiti della riorganizzazione.  
        -   **🎵 Musique Concrète**: Manipolazione granulare profonda per texture astratte complesse.
        -   **🏛️ Decostruzione Postmoderna**: Approccio critico radicale con ironia e rotture marcate.
        -   **🎨 Decomposizione Creativa**: Focus su discontinuità estreme e shift emotivi pronunciati.
        -   **🌪️ Random Chaos**: Trasformazioni altamente imprevedibili e massimali.
        
        ### Loop Decomposto:
        -   Una nuova funzionalità che applica un insieme di trasformazioni casuali e aggressive (pitch shift, time stretch, inversione, frammentazione, rumore) a un piccolo segmento del tuo audio originale, poi lo ripete per creare un loop molto caotico e imprevedibile, perfetto per ambientazioni estreme o glitch sonori.

        ### Ottimizzazioni:
        -   ⚡ Tutti i metodi sono ora ottimizzati per file fino a 5 minuti con elaborazione approfondita.
        -   🎛️ Parametri preimpostati per garantire la massima intensità per ogni stile.
        -   🚀 Performance bilanciate per un'esperienza d'uso stabile.
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><em>MusicDecomposer by loop507</em></p>
    <p>Trasforma la tua musica in arte sonora sperimentale</p>
    <p style='font-size: 0.8em;'>Inspired by: Musique Concrète • Plunderphonics • Cut-up Technique • Postmodern Deconstruction</p>
</div>
""", unsafe_allow_html=True)

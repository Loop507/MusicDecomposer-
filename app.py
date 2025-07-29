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
st.set_page_config(
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

# PARAMETRI PREIMPOSTATI PER OGNI INTENSITÀ
PRESET_PARAMS = {
    'soft': {
        'cut_up_sonoro': {
            'fragment_size': 2.0,
            'chaos_level': 0.3,
            'structure_preservation': 0.7,
            'cut_randomness': 0.4,
            'reassembly_style': 'random'
        },
        'remix_destrutturato': {
            'fragment_size': 1.5,
            'chaos_level': 0.3,
            'structure_preservation': 0.8,
            'beat_preservation': 0.7,
            'melody_fragmentation': 0.8
        },
        'musique_concrete': {
            'fragment_size': 1.0,
            'chaos_level': 0.2,
            'structure_preservation': 0.6,
            'grain_size': 0.2,
            'texture_density': 0.8
        },
        'decostruzione_postmoderna': {
            'fragment_size': 2.0,
            'chaos_level': 0.3,
            'structure_preservation': 0.7,
            'irony_level': 0.3,
            'context_shift': 0.4
        },
        'decomposizione_creativa': {
            'fragment_size': 1.5,
            'chaos_level': 0.3,
            'structure_preservation': 0.6,
            'discontinuity': 0.5,
            'emotional_shift': 0.4
        },
        'random_chaos': {
            'fragment_size': 2.0,
            'chaos_level': 0.4,
            'structure_preservation': 0.5
        }
    },
    'medio': {
        'cut_up_sonoro': {
            'fragment_size': 1.2,
            'chaos_level': 0.6,
            'structure_preservation': 0.4,
            'cut_randomness': 0.6,
            'reassembly_style': 'reverse'
        },
        'remix_destrutturato': {
            'fragment_size': 1.0,
            'chaos_level': 0.6,
            'structure_preservation': 0.5,
            'beat_preservation': 0.5,
            'melody_fragmentation': 1.2
        },
        'musique_concrete': {
            'fragment_size': 0.8,
            'chaos_level': 0.5,
            'structure_preservation': 0.4,
            'grain_size': 0.15,
            'texture_density': 1.2
        },
        'decostruzione_postmoderna': {
            'fragment_size': 1.5,
            'chaos_level': 0.5,
            'structure_preservation': 0.5,
            'irony_level': 0.5,
            'context_shift': 0.6
        },
        'decomposizione_creativa': {
            'fragment_size': 1.0,
            'chaos_level': 0.6,
            'structure_preservation': 0.4,
            'discontinuity': 0.8,
            'emotional_shift': 0.7
        },
        'random_chaos': {
            'fragment_size': 1.5,
            'chaos_level': 0.7,
            'structure_preservation': 0.3
        }
    },
    'hard': {
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
}

def analyze_audio_structure(audio, sr):
    """Analizza la struttura del brano per identificare elementi musicali - Versione ottimizzata"""
    if audio.size == 0:
        return {
            'tempo': 0, 'beats': np.array([]), 'chroma': np.array([]),
            'mfcc': np.array([]), 'spectral_centroids': np.array([]),
            'onset_times': np.array([]), 'onset_frames': np.array([])
        }

    # Per file lunghi, usa solo una porzione per l'analisi
    max_analysis_length = min(len(audio), sr * 60)  # Massimo 60 secondi per analisi
    analysis_audio = audio[:max_analysis_length]

    tempo = 0
    beats = np.array([])
    try:
        tempo, beats = librosa.beat.beat_track(y=analysis_audio, sr=sr, hop_length=1024)
        if beats.ndim > 1:
            beats = beats.flatten()
    except Exception as e:
        st.warning(f"Warning: Could not track beats, {e}")
        tempo = 0
        beats = np.array([])

    # Analisi semplificata per le altre feature
    chroma = np.array([])
    mfcc = np.array([])
    spectral_centroids = np.array([])
    onset_frames = np.array([])
    onset_times = np.array([])

    try:
        # Usa hop_length più grande per velocizzare
        chroma = librosa.feature.chroma_stft(y=analysis_audio, sr=sr, hop_length=2048)
        mfcc = librosa.feature.mfcc(y=analysis_audio, sr=sr, n_mfcc=13, hop_length=2048)
        spectral_centroids = librosa.feature.spectral_centroid(y=analysis_audio, sr=sr, hop_length=2048)[0]
        onset_frames = librosa.onset.onset_detect(y=analysis_audio, sr=sr, hop_length=1024)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=1024)
    except Exception as e:
        st.warning(f"Warning: Feature extraction partially failed: {e}")

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
    """Pitch shift sicuro con gestione errori - Versione ottimizzata"""
    try:
        if audio.size == 0:
            return np.array([])
        
        # Limita il pitch shift per evitare artefatti estremi
        n_steps = np.clip(n_steps, -12, 12)
        
        # Per audio molto lunghi, usa hop_length più grande
        hop_length = 512 if len(audio) < sr * 30 else 1024
        
        result = librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps, hop_length=hop_length)
        return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception as e:
        st.warning(f"Pitch shift fallito: {e}")
        return audio

def safe_time_stretch(audio, rate):
    """Time stretch sicuro con gestione errori - Versione ottimizzata"""
    try:
        if audio.size == 0:
            return np.array([])
        
        # Limita il rate per evitare artefatti estremi
        rate = np.clip(rate, 0.25, 4.0)
        
        # Per audio molto lunghi, usa hop_length più grande
        hop_length = 512 if len(audio) < 44100 * 30 else 1024
        
        result = librosa.effects.time_stretch(audio, rate=rate, hop_length=hop_length)
        return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception as e:
        st.warning(f"Time stretch fallito: {e}")
        return audio

def cut_up_sonoro(audio, sr, params):
    """Implementa la tecnica cut-up applicata all'audio - Versione ottimizzata"""
    fragment_size = params['fragment_size']
    randomness = params.get('cut_randomness', 0.7)
    reassembly = params.get('reassembly_style', 'random')

    if audio.size == 0:
        return np.array([])

    fragment_samples = int(fragment_size * sr)
    if fragment_samples <= 0:
        return audio

    # Limita il numero di frammenti per file lunghi
    max_fragments = min(50, len(audio) // fragment_samples)
    
    fragments = []
    step = max(fragment_samples, len(audio) // max_fragments) if max_fragments > 0 else fragment_samples
    
    for i in range(0, min(len(audio) - fragment_samples + 1, max_fragments * step), step):
        fragment = audio[i:i + fragment_samples]

        if fragment.size == 0:
            continue

        if random.random() < randomness:
            variation = random.uniform(0.7, 1.3)  # Variazione più conservativa
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

    # Applica lo stile di riassemblaggio
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
    """Remix che mantiene elementi riconoscibili - Versione ottimizzata"""
    fragment_size = params['fragment_size']
    beat_preservation = params.get('beat_preservation', 0.4)
    melody_fragmentation = params.get('melody_fragmentation', 1.5)

    if audio.size == 0:
        return np.array([])

    # Analisi semplificata per file lunghi
    structure = analyze_audio_structure(audio, sr)
    beats = structure['beats']

    fragment_samples = int(fragment_size * sr)
    if fragment_samples <= 0:
        return audio

    # Limita il numero di tagli
    max_cuts = min(30, len(audio) / fragment_samples) # Cambiato in divisione
    
    cut_points = []
    if beat_preservation > 0.5 and beats.size > 0:
        beat_times = librosa.frames_to_time(beats, sr=sr)
        cut_points = [int(t * sr) for t in beat_times[:int(max_cuts)] if t * sr < len(audio)] # Aggiunto int() per max_cuts

    if len(cut_points) == 0 or beat_preservation <= 0.5:
        num_cuts = min(int(max_cuts), int(len(audio) / fragment_samples))
        if num_cuts > 0 and len(audio) > 0:
            cut_points = sorted(random.sample(range(0, len(audio), fragment_samples), min(num_cuts, len(range(0, len(audio), fragment_samples)))))

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
            # Applica trasformazioni più conservative
            if random.random() < melody_fragmentation / 4.0:  # Ridotta probabilità
                try:
                    shift_steps = random.uniform(-4, 4)  # Range ridotto
                    fragment = safe_pitch_shift(fragment, sr, shift_steps)
                except Exception:
                    fragment = np.array([])

            if fragment.size > 0 and random.random() < melody_fragmentation / 4.0:
                try:
                    stretch_factor = random.uniform(0.8, 1.2)  # Range ridotto
                    fragment = safe_time_stretch(fragment, stretch_factor)
                except Exception:
                    fragment = np.array([])

        if fragment.size > 0:
            fragments.append(fragment)

    if len(fragments) == 0:
        return np.array([])

    # Riassemblaggio semplificato
    if beat_preservation > 0.3 and len(fragments) > 1:
        preserve_count = int(len(fragments) * beat_preservation)
        preserve_indices = random.sample(range(len(fragments)), min(preserve_count, len(fragments)))
        
        # Rimescola solo i frammenti non preservati
        non_preserved = [fragments[i] for i in range(len(fragments)) if i not in preserve_indices]
        random.shuffle(non_preserved)
        
        # Ricostruisci mantenendo l'ordine per quelli preservati
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

    # Concatenazione con crossfade semplificato
    if len(fragments) == 0:
        return np.array([])

    result = fragments[0]
    fade_samples = int(0.02 * sr)  # Fade più corto

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
    """Applica tecniche di musique concrète - Versione ottimizzata"""
    grain_size = params.get('grain_size', 0.1)
    texture_density = params.get('texture_density', 1.0)
    chaos_level = params['chaos_level']

    if audio.size == 0:
        return np.array([])

    grain_samples = int(grain_size * sr)
    if grain_samples <= 0:
        return audio

    # Limita il numero di grani per file lunghi
    max_grains = min(100, len(audio) // (grain_samples // 4))
    step = max(grain_samples // 2, len(audio) // max_grains) if max_grains > 0 else grain_samples // 2
    
    grains = []
    for i in range(0, min(len(audio) - grain_samples + 1, max_grains * step), step):
        grain = audio[i:i + grain_samples]

        if grain.size == 0:
            continue

        # Applicazione finestra semplificata
        try:
            window = signal.windows.hann(len(grain))  # Hann invece di Gaussian (più veloce)
            grain = grain * window
        except Exception:
            pass

        # Trasformazioni più conservative
        if random.random() < chaos_level / 4.0:  # Ridotta probabilità
            grain = grain[::-1]

        if grain.size > 0 and random.random() < chaos_level / 4.0:
            try:
                shift = random.uniform(-6, 6)  # Range ridotto
                grain = safe_pitch_shift(grain, sr, shift)
            except Exception:
                grain = np.array([])

        if grain.size > 0 and random.random() < chaos_level / 4.0:
            try:
                stretch = random.uniform(0.5, 2.0)  # Range ridotto
                grain = safe_time_stretch(grain, stretch)
            except Exception:
                grain = np.array([])

        if grain.size > 0:
            grains.append(grain)

    if len(grains) == 0:
        return np.array([])

    # Seleziona un numero gestibile di grani
    num_grains_output = min(int(len(grains) * texture_density), 80)
    if num_grains_output <= 0:
        return np.array([])

    if num_grains_output < len(grains):
        grains = random.sample(grains, num_grains_output)

    # Crea il risultato finale
    max_length = min(int(len(audio) * (1 + texture_density * 0.3)), len(audio) * 2)  # Limita lunghezza
    result = np.zeros(max_length)

    for grain in grains:
        if grain.size == 0:
            continue

        if grain.size < max_length:
            start_pos = random.randint(0, max_length - grain.size)
            end_pos = start_pos + grain.size
            result[start_pos:end_pos] += grain * random.uniform(0.5, 0.8)

    # Normalizzazione
    if result.size > 0:
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        max_val = np.max(np.abs(result))
        if max_val > 0:
            result = result / max_val * 0.8

    return result

def decostruzione_postmoderna(audio, sr, params):
    """Decostruzione postmoderna ottimizzata per performance"""
    irony_level = params.get('irony_level', 0.5)
    context_shift = params.get('context_shift', 0.6)
    fragment_size = params['fragment_size']

    if audio.size == 0:
        return np.array([])

    fragment_samples = int(fragment_size * sr)
    if fragment_samples <= 0:
        return audio

    # Numero massimo di frammenti per evitare sovraccarico
    max_fragments = min(20, len(audio) // fragment_samples)
    
    fragments = []
    fragment_types = []

    # Identifica alcuni punti importanti (semplificato)
    step = len(audio) // max_fragments if max_fragments > 0 else fragment_samples
    
    for i in range(0, len(audio) - fragment_samples + 1, step):
        if len(fragments) >= max_fragments:
            break
            
        fragment = audio[i:i + fragment_samples]
        if fragment.size > 0:
            # Calcola energia semplice
            energy = np.sqrt(np.mean(fragment**2))
            # Utilizza una media per la soglia basata su un piccolo campione dell'audio per evitare errori con audio vuoto
            if len(audio) > fragment_samples * 10: # Assicurati che ci sia abbastanza audio per il campione
                sample_energies = [np.sqrt(np.mean(audio[j:j+fragment_samples]**2)) 
                                   for j in range(0, len(audio) - fragment_samples + 1, len(audio) // 10)]
                mean_sample_energy = np.mean(sample_energies) if sample_energies else 0
            else:
                mean_sample_energy = 0 # Nessuna media se l'audio è troppo corto
            
            fragment_type = 'important' if energy > mean_sample_energy else 'random'
            
            fragments.append(fragment)
            fragment_types.append(fragment_type)

    if len(fragments) == 0:
        return audio

    # Processa frammenti con trasformazioni leggere
    processed_fragments = []
    
    for fragment, frag_type in zip(fragments, fragment_types):
        if fragment.size == 0:
            continue

        processed_frag = fragment.copy()

        # Trasformazioni ironiche semplificate
        if frag_type == 'important' and random.random() < irony_level * 0.3:
            transform_choice = random.choice(['reverse', 'volume_down', 'fade_out'])
            
            try:
                if transform_choice == 'reverse':
                    processed_frag = processed_frag[::-1]
                elif transform_choice == 'volume_down':
                    processed_frag = processed_frag * 0.4
                elif transform_choice == 'fade_out':
                    fade = np.linspace(1, 0.3, len(processed_frag))
                    processed_frag = processed_frag * fade
            except Exception:
                processed_frag = fragment

        # Context shift leggero
        if processed_frag.size > 0 and random.random() < context_shift * 0.2:
            effect_choice = random.choice(['fade_in', 'light_noise'])
            
            try:
                if effect_choice == 'fade_in':
                    fade = np.linspace(0.4, 1, len(processed_frag))
                    processed_frag = processed_frag * fade
                elif effect_choice == 'light_noise':
                    noise = np.random.normal(0, 0.005, len(processed_frag))
                    processed_frag = processed_frag + noise
            except Exception:
                processed_frag = fragment

        if processed_frag.size > 0:
            processed_fragments.append(processed_frag)

    if len(processed_fragments) == 0:
        return audio

    # Riassemblaggio semplificato
    random.shuffle(processed_fragments)
    final_fragments = processed_fragments[:min(15, len(processed_fragments))]

    if len(final_fragments) == 0:
        return audio
    
    if len(final_fragments) == 1:
        return final_fragments[0]

    # Concatenazione con crossfade leggero
    result = final_fragments[0].copy()
    
    for next_fragment in final_fragments[1:]:
        crossfade_samples = min(128, len(result) // 8, len(next_fragment) // 8)
        
        if crossfade_samples > 0:
            fade_out = np.linspace(1, 0, crossfade_samples)
            fade_in = np.linspace(0, 1, crossfade_samples)
            
            result[-crossfade_samples:] *= fade_out
            result[-crossfade_samples:] += next_fragment[:crossfade_samples] * fade_in
            
            if len(next_fragment) > crossfade_samples:
                result = np.concatenate([result, next_fragment[crossfade_samples:]])
        else:
            result = np.concatenate([result, next_fragment])

    # Normalizzazione finale
    if len(result) > 0:
        max_val = np.max(np.abs(result))
        if max_val > 1.0:
            result = result / max_val * 0.95

    return result

def decomposizione_creativa(audio, sr, params):
    """Decomposizione Creativa ottimizzata"""
    discontinuity = params.get('discontinuity', 1.0)
    emotional_shift = params.get('emotional_shift', 0.8)
    fragment_size = params['fragment_size']
    chaos_level = params['chaos_level']

    if audio.size == 0:
        return np.array([])

    # Limita l'analisi degli onset per file lunghi
    max_analysis_length = min(len(audio), sr * 60)
    analysis_audio = audio[:max_analysis_length]
    
    try:
        onset_frames = librosa.onset.onset_detect(y=analysis_audio, sr=sr, hop_length=1024)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=1024)
    except Exception:
        onset_times = np.array([])

    # Crea punti di taglio limitati
    fragment_samples = int(fragment_size * sr)
    max_fragments = min(25, len(audio) // fragment_samples)
    
    cut_points = sorted(list(set([0] + [int(t * sr) for t in onset_times[:max_fragments//2]] + 
                                [i * fragment_samples for i in range(0, max_fragments)])))
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

        # Discontinuità semplificata
        if random.random() < discontinuity * 0.08:  # Probabilità ridotta
            if random.random() < 0.5:
                fragment = np.zeros_like(fragment) * 0.1  # Silenzio parziale invece di completo
            else:
                continue

        # Shift emotivi più conservativi
        if random.random() < emotional_shift * 0.2:
            if random.random() < 0.5:
                shift_steps = random.uniform(-6 * emotional_shift, 6 * emotional_shift)
                fragment = safe_pitch_shift(fragment, sr, shift_steps)
            else:
                stretch_rate = random.uniform(1 - 0.2 * emotional_shift, 1 + 0.2 * emotional_shift)
                fragment = safe_time_stretch(fragment, stretch_rate)
        
        # Caos leggero
        if fragment.size > 0 and random.random() < chaos_level * 0.05:
            fragment = fragment[::-1]

        if fragment.size > 0:
            processed_fragments.append(fragment)

    if not processed_fragments:
        return np.array([])

    # Concatena i frammenti
    final_audio = np.concatenate(processed_fragments)

    # Normalizzazione finale
    if final_audio.size > 0:
        max_val = np.max(np.abs(final_audio))
        if max_val > 0:
            final_audio = final_audio / max_val * 0.95
    return final_audio

def random_chaos(audio, sr, params):
    """Random Chaos ottimizzato per performance"""
    chaos_level = params['chaos_level']

    if audio.size == 0:
        return np.array([])

    processed_audio = audio.copy()

    # Operazioni casuali più conservative
    # Pitch Shift moderato
    if random.random() < 0.3 * chaos_level:  # Probabilità ridotta
        shift_steps = random.uniform(-12, 12)  # Range ridotto
        processed_audio = safe_pitch_shift(processed_audio, sr, shift_steps)
        if processed_audio.size == 0: 
            return np.array([])

    # Time Stretch moderato
    if random.random() < 0.3 * chaos_level:
        stretch_rate = random.uniform(0.25, 4.0)  # Range ridotto
        processed_audio = safe_time_stretch(processed_audio, stretch_rate)
        if processed_audio.size == 0: 
            return np.array([])

    # Inversione casuale
    if random.random() < 0.2 * chaos_level:
        if processed_audio.size > sr * 2 and random.random() < 0.5:
            start_idx = random.randint(0, max(0, processed_audio.size - int(sr * 2)))
            end_idx = min(processed_audio.size, start_idx + int(sr * 2))
            if end_idx > start_idx:
                processed_audio[start_idx:end_idx] = processed_audio[start_idx:end_idx][::-1]
        else:
            processed_audio = processed_audio[::-1]

    # Rumore leggero
    if random.random() < 0.2 * chaos_level:
        if processed_audio.size > 0:
            noise_amplitude = random.uniform(0.005, 0.05) * chaos_level
            noise = np.random.normal(0, noise_amplitude, processed_audio.size)
            processed_audio = processed_audio + noise

    # Frammentazione semplificata
    if random.random() < 0.3 * chaos_level:
        if processed_audio.size > 0:
            max_fragment_len = int(sr * 2)  # Massimo 2 secondi per frammento
            chaos_fragments = []
            current_pos = 0
            max_fragments = 20  # Limita numero frammenti
            
            while current_pos < processed_audio.size and len(chaos_fragments) < max_fragments:
                frag_len = min(random.randint(int(sr * 0.1), max_fragment_len), 
                              processed_audio.size - current_pos)
                if frag_len <= 0: 
                    break
                chaos_fragments.append(processed_audio[current_pos : current_pos + frag_len])
                current_pos += frag_len + int(sr * random.uniform(0, 0.1))

            if chaos_fragments:
                random.shuffle(chaos_fragments)
                processed_audio = np.concatenate(chaos_fragments)

    # Normalizzazione finale
    if processed_audio.size > 0:
        processed_audio = np.nan_to_num(processed_audio, nan=0.0, posinf=0.0, neginf=0.0)
        max_val = np.max(np.abs(processed_audio))
        if max_val > 0:
            processed_audio = processed_audio / max_val * 0.95
    
    return processed_audio

# Logica principale per processare l'audio
if uploaded_file is not None:
    # Aggiungi un selettore per il sample rate target nella sidebar o in una colonna
    st.sidebar.subheader("⚙️ Impostazioni Caricamento Audio")
    target_sr_options = {
        "Auto (Originale)": None,
        "44100 Hz (CD Quality)": 44100,
        "22050 Hz (Radio Quality)": 22050,
        "16000 Hz (Voice Quality)": 16000,
        "8000 Hz (Telephone Quality)": 8000
    }
    
    selected_target_sr_label = st.sidebar.selectbox(
        "Seleziona il Sample Rate per l'elaborazione:",
        list(target_sr_options.keys()),
        index=2, # Preseleziona 22050 Hz come default più leggero
        help="Un sample rate più basso riduce il consumo di memoria e CPU, ma può ridurre la qualità audio."
    )
    
    target_sr = target_sr_options[selected_target_sr_label]

    try:
        # Carica il file audio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        # Leggi l'audio con limitazione per file molto lunghi e con target_sr
        audio, sr = librosa.load(tmp_file_path, sr=target_sr, duration=300)  # Massimo 5 minuti
        
        # Mostra informazioni del file originale
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Durata", f"{len(audio)/sr:.2f} sec")
        with col2:
            st.metric("Sample Rate", f"{sr} Hz")
        with col3:
            st.metric("Canali", "Mono")

        # Player per l'audio originale
        st.subheader("Audio Originale")
        st.audio(uploaded_file, format='audio/wav')

        # INTERFACCIA CON 3 PULSANTI PREIMPOSTATI
        st.markdown("---")
        st.subheader("🎛️ Scegli l'Intensità della Decomposizione")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🟢 SOFT", type="secondary", use_container_width=True):
                st.session_state.intensity = 'soft'
        
        with col2:
            if st.button("🟡 MEDIO", type="secondary", use_container_width=True):
                st.session_state.intensity = 'medio'
        
        with col3:
            if st.button("🔴 HARD", type="primary", use_container_width=True):
                st.session_state.intensity = 'hard'

        # Mostra intensità selezionata
        if hasattr(st.session_state, 'intensity'):
            intensity_labels = {'soft': '🟢 SOFT', 'medio': '🟡 MEDIO', 'hard': '🔴 HARD'}
            st.info(f"Intensità selezionata: **{intensity_labels[st.session_state.intensity]}**")
            
            # Selector per il metodo
            st.markdown("---")
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
                "Seleziona il Metodo di Decomposizione:",
                method_options,
                format_func=lambda x: method_labels[x]
            )

            # Pulsante per processare
            if st.button("🎭 SCOMPONI E RICOMPONI", type="primary", use_container_width=True):
                with st.spinner(f"Applicando {method_labels[selected_method]} con intensità {intensity_labels[st.session_state.intensity]}..."):
                    
                    # Ottieni parametri preimpostati
                    params = PRESET_PARAMS[st.session_state.intensity][selected_method].copy()
                    
                    # Applica il metodo selezionato
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

                    # Verifica risultato
                    if processed_audio.size == 0:
                        st.error("❌ Elaborazione fallita - audio risultante vuoto. Prova un metodo diverso.")
                    else:
                        # Salva l'audio processato
                        processed_tmp_path = ""
                        try:
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as processed_tmp:
                                sf.write(processed_tmp.name, processed_audio, sr)
                                processed_tmp_path = processed_tmp.name

                            # Mostra risultati
                            st.success("✅ Decomposizione completata!")
                            
                            # Metriche dell'audio processato
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

                            # Player per l'audio processato
                            st.subheader("Audio Decomposto e Ricomposto")
                            
                            with open(processed_tmp_path, 'rb') as audio_file:
                                audio_bytes = audio_file.read()
                                st.audio(audio_bytes, format='audio/wav')

                            # Pulsante download
                            method_names = {
                                "cut_up_sonoro": "CutUp",
                                "remix_destrutturato": "RemixDestrutturato", 
                                "musique_concrete": "MusiqueConcrete",
                                "decostruzione_postmoderna": "DecostruzionePostmoderna",
                                "decomposizione_creativa": "DecomposizioneCreativa",
                                "random_chaos": "RandomChaos"
                            }
                            
                            filename = f"{uploaded_file.name.split('.')[0]}_{method_names[selected_method]}_{st.session_state.intensity}.wav"
                            
                            st.download_button(
                                label="💾 Scarica Audio Decomposto",
                                data=audio_bytes,
                                file_name=filename,
                                mime="audio/wav",
                                use_container_width=True
                            )

                            # SINTESI ARTISTICA DELLA DECOMPOSIZIONE
                            technique_descriptions = {
                                "cut_up_sonoro": """
                                Il metodo **"Cut-up Sonoro"** si ispira a una tecnica letteraria dove il testo viene frammentato e riassemblato. Il brano viene diviso in sezioni, che vengono poi **tagliate e riassemblate in un ordine casuale o predefinito** (come inversione o palindromo). Questo crea un effetto di collage sonoro, dove il significato originale è destrutturato per rivelare nuove connessioni e pattern imprevedibili.
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
                            
                            # Sintesi Artistica Dinamica
                            st.subheader("Sintesi Artistica della Decomposizione")
                            
                            if selected_method == "cut_up_sonoro":
                                st.markdown(f"""
                                Con il metodo del **"Cut-up Sonoro"** applicato con intensità **{st.session_state.intensity.upper()}**, il brano originale è stato smembrato e ricombinato in un collage sonoro decostruito. Frammenti di {params['fragment_size']:.1f} secondi sono stati riorganizzati con casualità {params['cut_randomness']:.1f} e stile '{params['reassembly_style']}', creando **inaspettate giustapposizioni** che sfidano la percezione tradizionale.
                                """)
                            elif selected_method == "remix_destrutturato":
                                st.markdown(f"""
                                Attraverso il **"Remix Destrutturato"** con intensità **{st.session_state.intensity.upper()}**, l'essenza del brano è stata catturata e rielaborata. Con conservazione del battito al {params['beat_preservation']*100:.0f}% e frammentazione melodica {params['melody_fragmentation']:.1f}, gli elementi sonori sono stati **ricollocati in un paesaggio acustico reinventato** che è familiare ed estraneo insieme.
                                """)
                            elif selected_method == "musique_concrete":
                                st.markdown(f"""
                                Con la **"Musique Concrète"** in modalità **{st.session_state.intensity.upper()}**, il brano è stato ridotto a grani sonori di {params['grain_size']:.2f} secondi, manipolati e ricombinati con densità {params['texture_density']:.1f}. Il risultato è un'**opera sonora astratta** che esplora le qualità timbriche intrinseche oltre l'organizzazione musicale originale.
                                """)
                            elif selected_method == "decostruzione_postmoderna":
                                st.markdown(f"""
                                La **"Decostruzione Postmoderna"** in versione **{st.session_state.intensity.upper()}** ha applicato un filtro concettuale con ironia {params['irony_level']:.1f} e shift di contesto {params['context_shift']:.1f}. Elementi riconoscibili sono stati trattati in modo inaspettato, provocando una **riflessione critica sull'opera** e trasformando il familiare in qualcosa di destabilizzante ma affascinante.
                                """)
                            elif selected_method == "decomposizione_creativa":
                                st.markdown(f"""
                                La **"Decomposizione Creativa"** con intensità **{st.session_state.intensity.upper()}** ha frammentato il brano sui punti di attacco, generando discontinuità {params['discontinuity']:.1f} e shift emotivi {params['emotional_shift']:.1f}. Il risultato è un'esperienza sonora **ricca di colpi di scena** e improvvisi cambi di umore emotivo.
                                """)
                            elif selected_method == "random_chaos":
                                st.markdown(f"""
                                Il **"Random Chaos"** in modalità **{st.session_state.intensity.upper()}** ha spinto il brano nei suoi limiti estremi con chaos level {params['chaos_level']:.1f}. **Trasformazioni radicali e casuali** hanno creato un'esplorazione sonora che sfugge a qualsiasi classificazione, un viaggio in un paesaggio acustico alieno dove l'originale è appena un eco lontano.
                                """)

                            st.markdown("---")
                            st.markdown("**Descrizione della Tecnica Applicata:**")
                            st.markdown(technique_descriptions[selected_method])

                            st.markdown("---")
                            st.markdown("### Riepilogo dei Cambiamenti Quantitativi:")

                            analysis_text = f"""
                            * **Durata:** L'audio originale di **{uploaded_file.size / (sr * 2):.2f} secondi** è stato trasformato in **{new_duration:.2f} secondi** ({'allungamento' if new_duration > (uploaded_file.size / (sr * 2)) else 'accorciamento'} di **{abs(new_duration - (uploaded_file.size / (sr * 2))):.2f} secondi**).
                            * **Sample Rate Elaborazione:** Il brano è stato elaborato a **{sr} Hz**. Un sample rate più basso riduce il consumo di memoria e CPU.
                            * **Energia RMS:** Variazione di **{processed_rms - original_rms:.4f}** - il suono risultante è {'più forte' if processed_rms > original_rms else 'più debole' if processed_rms < original_rms else 'simile'} in volume medio.
                            * **Variazione Spettrale:** **{spectral_diff:.2e}** - quantifica il cambiamento del "colore" e distribuzione delle frequenze rispetto all'originale.
                            
                            **Parametri Applicati** (Intensità {st.session_state.intensity.upper()}):
                            """
                            st.markdown(analysis_text)
                            # Stampa i parametri specifici del metodo
                            for param_name, param_value in params.items():
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


                            # Visualizzazioni
                            with st.expander("📊 Confronto Forme d'Onda"):
                                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
                                
                                # Assicurati che audio non sia vuoto prima di plottare
                                if audio.size > 0:
                                    time_orig = np.linspace(0, len(audio)/sr, len(audio))
                                    ax1.plot(time_orig, audio, color='blue', alpha=0.7)
                                ax1.set_title("Forma d'Onda Originale")
                                ax1.set_xlabel("Tempo (sec)")
                                ax1.set_ylabel("Ampiezza")
                                ax1.grid(True, alpha=0.3)
                                
                                if processed_audio.size > 0:
                                    time_proc = np.linspace(0, len(processed_audio)/sr, len(processed_audio))
                                    ax2.plot(time_proc, processed_audio, color='red', alpha=0.7)
                                ax2.set_title(f"Forma d'Onda Decomposta ({method_labels[selected_method]} - {st.session_state.intensity.upper()})")
                                ax2.set_xlabel("Tempo (sec)")
                                ax2.set_ylabel("Ampiezza")
                                ax2.grid(True, alpha=0.3)
                                
                                plt.tight_layout()
                                st.pyplot(fig)

                            with st.expander("🎼 Analisi Spettrale"):
                                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                                
                                # Aggiungi controlli per evitare errori su audio vuoto o troppo corto per STFT
                                if audio.size > sr * 2: # Richiede almeno 2 secondi di audio per una STFT significativa
                                    D_orig = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
                                    librosa.display.specshow(D_orig, y_axis='hz', x_axis='time', sr=sr, ax=ax1)
                                    ax1.set_title("Spettrogramma Originale")
                                    ax1.set_xlabel("Tempo (sec)")
                                    ax1.set_ylabel("Frequenza (Hz)")
                                else:
                                    ax1.set_title("Spettrogramma Originale (Audio troppo corto per analisi)")
                                
                                if processed_audio.size > sr * 2:
                                    D_proc = librosa.amplitude_to_db(np.abs(librosa.stft(processed_audio)), ref=np.max)
                                    librosa.display.specshow(D_proc, y_axis='hz', x_axis='time', sr=sr, ax=ax2)
                                    ax2.set_title("Spettrogramma Decomposto")
                                    ax2.set_xlabel("Tempo (sec)")
                                    ax2.set_ylabel("Frequenza (Hz)")
                                else:
                                    ax2.set_title("Spettrogramma Decomposto (Audio troppo corto per analisi)")
                                
                                plt.tight_layout()
                                st.pyplot(fig)
                                
                        finally:
                            # Cleanup
                            try:
                                if os.path.exists(tmp_file_path):
                                    os.unlink(tmp_file_path)
                                if processed_tmp_path and os.path.exists(processed_tmp_path):
                                    os.unlink(processed_tmp_path)
                            except Exception as e:
                                st.error(f"Errore durante la pulizia: {e}")
        else:
            st.info("👆 Seleziona prima l'intensità della decomposizione (SOFT, MEDIO, HARD)")

    except Exception as e:
        st.error(f"❌ Errore nel processamento: {str(e)}")
        st.error(f"Dettagli: {traceback.format_exc()}")

else:
    # Messaggio quando non c'è file caricato
    st.info("👆 Carica un file audio per iniziare la decomposizione")
    
    # Istruzioni d'uso
    with st.expander("📖 Come usare MusicDecomposer"):
        st.markdown("""
        ### Modalità di utilizzo semplificata:

        1.  **Carica il tuo file audio** (MP3, WAV, M4A, FLAC, OGG)
        2.  **Imposta il Sample Rate per l'elaborazione** (nella sidebar a sinistra):
            * Un sample rate più basso (es. 22050 Hz o 16000 Hz) ridurrà notevolmente il consumo di risorse.
        3.  **Scegli l'intensità:**
            * 🟢 **SOFT**: Trasformazioni delicate, conserva molto dell'originale
            * 🟡 **MEDIO**: Trasformazioni bilanciate, mix di conservazione e creatività  
            * 🔴 **HARD**: Trasformazioni intense, risultati più estremi e sperimentali
        4.  **Seleziona il metodo di decomposizione**
        5.  **Clicca "SCOMPONI E RICOMPONI"**

        ### Metodi disponibili:
        - **🎭 Cut-up Sonoro**: Collage sonoro con frammenti riassemblati
        - **🔄 Remix Destrutturato**: Remix creativo che mantiene elementi riconoscibili  
        - **🎵 Musique Concrète**: Manipolazione granulare per texture astratte
        - **🏛️ Decostruzione Postmoderna**: Approccio critico con ironia e rotture
        - **🎨 Decomposizione Creativa**: Focus su discontinuità e shift emotivi
        - **🌪️ Random Chaos**: Trasformazioni imprevedibili e sperimentali

        ### Ottimizzazioni:
        - ⚡ Tutti i metodi sono ottimizzati per file fino a 5 minuti
        - 🎛️ Parametri preimpostati per ogni intensità
        - 🚀 Performance ottimizzate per evitare crash
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

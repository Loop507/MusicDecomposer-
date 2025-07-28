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
import traceback # Importa per i dettagli degli errori

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

# Sidebar per parametri
with st.sidebar:
    st.header("Controlli Decomposizione")

    decomposition_method = st.selectbox(
        "Metodo di Decomposizione",
        [
            "cut_up_sonoro",
            "remix_destrutturato",
            "musique_concrete",
            "decostruzione_postmoderna",
            "decomposizione_creativa",
            "random_chaos"
        ],
        format_func=lambda x: {
            "cut_up_sonoro": "Cut-up Sonoro",
            "remix_destrutturato": "Remix Destrutturato",
            "musique_concrete": "Musique Concrète",
            "decostruzione_postmoderna": "Decostruzione Postmoderna",
            "decomposizione_creativa": "Decomposizione Creativa",
            "random_chaos": "Random Chaos"
        }[x]
    )

    st.markdown("---")

    # Parametri generali - più leggeri per decostruzione postmoderna
    if decomposition_method == "decostruzione_postmoderna":
        fragment_size = st.slider("Dimensione Frammenti (sec)", 0.5, 3.0, 1.5, 0.1)  # Frammenti più grandi
        chaos_level = st.slider("Livello di Chaos", 0.1, 1.5, 0.8, 0.1)  # Meno chaos
        structure_preservation = st.slider("Conservazione Struttura", 0.2, 0.8, 0.5, 0.1)  # Range più ristretto
    else:
        fragment_size = st.slider("Dimensione Frammenti (sec)", 0.1, 5.0, 1.0, 0.1)
        chaos_level = st.slider("Livello di Chaos", 0.1, 3.0, 1.0, 0.1)
        structure_preservation = st.slider("Conservazione Struttura", 0.0, 1.0, 0.3, 0.1)

    st.markdown("---")

    # Parametri specifici per metodo
    cut_randomness = None
    reassembly_style = None
    beat_preservation = None
    melody_fragmentation = None
    grain_size = None
    texture_density = None
    irony_level = None
    context_shift = None
    discontinuity = None
    emotional_shift = None

    if decomposition_method == "cut_up_sonoro":
        st.subheader("Cut-up Parameters")
        cut_randomness = st.slider("Casualità Tagli", 0.1, 1.0, 0.7, 0.1)
        reassembly_style = st.selectbox("Stile Riassemblaggio",
                                         ["random", "reverse", "palindrome", "spiral"])


    elif decomposition_method == "remix_destrutturato":
        st.subheader("Remix Parameters")
        beat_preservation = st.slider("Conserva Ritmo", 0.0, 1.0, 0.4, 0.1)
        melody_fragmentation = st.slider("Frammentazione Melodia", 0.1, 3.0, 1.5, 0.1)

    elif decomposition_method == "musique_concrete":
        st.subheader("Concrete Parameters")
        grain_size = st.slider("Dimensione Grani", 0.01, 0.5, 0.1, 0.01)
        texture_density = st.slider("Densità Texture", 0.1, 3.0, 1.0, 0.1)

    elif decomposition_method == "decostruzione_postmoderna":
        st.subheader("Postmodern Parameters (Ottimizzati)")
        irony_level = st.slider("Livello Ironia", 0.1, 1.0, 0.5, 0.1)  # Range ridotto
        context_shift = st.slider("Shift di Contesto", 0.1, 1.0, 0.6, 0.1)  # Range ridotto
        st.info("🔧 Parametri ottimizzati per prestazioni migliori")

    elif decomposition_method == "decomposizione_creativa":
        st.subheader("Creative Parameters")
        discontinuity = st.slider("Discontinuità", 0.1, 2.0, 1.0, 0.1)
        emotional_shift = st.slider("Shift Emotivo", 0.1, 2.0, 0.8, 0.1)

# Upload file
uploaded_file = st.file_uploader(
    "Carica il tuo brano da decomporre",
    type=["mp3", "wav", "m4a", "flac", "ogg"],
    help="Supporta MP3, WAV, M4A, FLAC, OGG"
)

def analyze_audio_structure(audio, sr):
    """Analizza la struttura del brano per identificare elementi musicali"""
    if audio.size == 0:
        return {
            'tempo': 0, 'beats': np.array([]), 'chroma': np.array([]),
            'mfcc': np.array([]), 'spectral_centroids': np.array([]),
            'onset_times': np.array([]), 'onset_frames': np.array([])
        }

    tempo = 0
    beats = np.array([])
    try:
        tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)
        if beats.ndim > 1:
            beats = beats.flatten()
    except Exception as e:
        st.warning(f"Warning: Could not track beats, {e}. Trace: {traceback.format_exc()}")
        tempo = 0
        beats = np.array([])

    chroma = np.array([])
    try:
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    except Exception as e:
        st.warning(f"Warning: Could not extract chroma features, {e}. Trace: {traceback.format_exc()}")

    mfcc = np.array([])
    try:
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    except Exception as e:
        st.warning(f"Warning: Could not extract MFCC features, {e}. Trace: {traceback.format_exc()}")

    spectral_centroids = np.array([])
    try:
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
    except Exception as e:
        st.warning(f"Warning: Could not extract spectral centroids, {e}. Trace: {traceback.format_exc()}")

    onset_frames = np.array([])
    try:
        onset_frames = librosa.onset.onset_detect(y=audio, sr=sr)
    except Exception as e:
        st.warning(f"Warning: Could not detect onsets, {e}. Trace: {traceback.format_exc()}")

    onset_times = np.array([])
    try:
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    except Exception as e:
        st.warning(f"Warning: Could not convert onset frames to times, {e}. Trace: {traceback.format_exc()}")

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
    """Pitch shift sicuro con gestione errori"""
    try:
        if audio.size == 0:
            return np.array([])
        result = librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
        return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception as e:
        st.warning(f"Pitch shift fallito: {e}")
        return audio


def safe_time_stretch(audio, rate):
    """Time stretch sicuro con gestione errori"""
    try:
        if audio.size == 0:
            return np.array([])
        result = librosa.effects.time_stretch(audio, rate=rate)
        return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception as e:
        st.warning(f"Time stretch fallito: {e}")
        return audio

def cut_up_sonoro(audio, sr, params):
    """Implementa la tecnica cut-up applicata all'audio"""
    fragment_size = params['fragment_size']
    randomness = params.get('cut_randomness', 0.7)
    reassembly = params.get('reassembly_style', 'random')

    if audio.size == 0:
        return np.array([])

    fragment_samples = int(fragment_size * sr)
    if fragment_samples <= 0:
        return audio # O np.array([]) se si preferisce restituire vuoto per dimensioni invalide

    fragments = []
    # Assicurati che il range sia valido
    for i in range(0, len(audio) - fragment_samples + 1, fragment_samples):
        fragment = audio[i:i + fragment_samples]

        if fragment.size == 0:
            continue

        if random.random() < randomness:
            variation = random.uniform(0.5, 1.5)
            new_size = int(fragment.size * variation)

            if new_size <= 0:
                continue

            if fragment.size > 0: # Solo se c'è qualcosa da manipolare
                if new_size < fragment.size:
                    fragment = fragment[:new_size]
                else:
                    indices = np.linspace(0, fragment.size - 1, new_size)
                    fragment = np.interp(indices, np.arange(fragment.size), fragment)
            else: # Se il frammento è vuoto, rimane vuoto
                fragment = np.array([])

        if fragment.size > 0:
            fragments.append(fragment)

    if len(fragments) == 0:
        return np.array([])

    if reassembly == 'random':
        random.shuffle(fragments)
    elif reassembly == 'reverse':
        fragments = [frag[::-1] for frag in fragments if frag.size > 0]
        fragments = fragments[::-1] # Reverse dell'ordine dei frammenti

    elif reassembly == 'palindrome':
        # Assicurati che i frammenti siano validi prima di aggiungerli alla lista del palindromo
        valid_fragments = [frag for frag in fragments if frag.size > 0]
        fragments = valid_fragments + [frag for frag in valid_fragments[::-1]]
    elif reassembly == 'spiral':
        new_fragments = []
        start, end = 0, len(fragments) - 1
        while start <= end:
            # CORREZIONE PRINCIPALE: controllo di bounds e .size
            if start < len(fragments) and fragments[start].size > 0 and len(new_fragments) % 2 == 0:
                new_fragments.append(fragments[start])
                start += 1
            elif end < len(fragments) and fragments[end].size > 0 and len(new_fragments) % 2 != 0:
                new_fragments.append(fragments[end])
                end -= 1
            else: # Se il frammento è vuoto, sposta il puntatore senza aggiungere
                if len(new_fragments) % 2 == 0:
                    start += 1
                else:
                    end -= 1
        fragments = new_fragments

    if len(fragments) > 0:
        result = np.concatenate(fragments)
    else:
        result = np.array([])

    return result

def remix_destrutturato(audio, sr, params):
    """Remix che mantiene elementi riconoscibili ma li ricontestualizza"""
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

    cut_points = []
    if beat_preservation > 0.5 and beats.size > 0:
        beat_times = librosa.frames_to_time(beats, sr=sr)
        cut_points = [int(t * sr) for t in beat_times if t * sr < len(audio)]

    if len(cut_points) == 0 or (beat_preservation <= 0.5):
        num_cuts = int(len(audio) / fragment_samples)
        if num_cuts == 0 and len(audio) > 0:
            num_cuts = 1
        if len(audio) > 0 and num_cuts > 0:
            # Ensure start point is valid for random.sample
            valid_range_len = len(range(0, len(audio)))
            if valid_range_len > 0:
                cut_points = sorted(random.sample(range(0, len(audio)), min(num_cuts, valid_range_len)))
            else:
                return np.array([])
        else:
            return np.array([])

    cut_points = sorted(list(set(cut_points)))
    if len(cut_points) == 0:
        return np.array([])

    if 0 not in cut_points:
        cut_points.insert(0, 0)
    if len(audio) not in cut_points:
        cut_points.append(len(audio))
    cut_points = sorted(list(set(cut_points)))

    fragments = []
    for i in range(len(cut_points) - 1):
        start = cut_points[i]
        end = cut_points[i + 1]
        if end <= start:
            continue
        fragment = audio[start:end]

        if fragment.size > 0:
            if random.random() < melody_fragmentation / 3.0:
                try:
                    shift_steps = random.uniform(-7, 7)
                    fragment = safe_pitch_shift(fragment, sr, shift_steps)
                except Exception as e:
                    st.warning(f"Pitch shift failed for fragment in remix_destrutturato: {e}. Trace: {traceback.format_exc()}")
                    fragment = np.array([]) # Marca il frammento come vuoto in caso di errore

            if fragment.size > 0 and random.random() < melody_fragmentation / 3.0: # Aggiunto controllo size
                try:
                    stretch_factor = random.uniform(0.7, 1.4)
                    fragment = safe_time_stretch(fragment, stretch_factor)
                except Exception as e:
                    st.warning(f"Time stretch failed for fragment in remix_destrutturato: {e}. Trace: {traceback.format_exc()}")
                    fragment = np.array([]) # Marca il frammento come vuoto in caso di errore

        if fragment.size > 0:
            fragments.append(fragment)

    if len(fragments) == 0:
        return np.array([])

    if beat_preservation > 0.3:
        preserve_count = int(len(fragments) * beat_preservation)
        if len(fragments) > 0: # Ensure fragments is not empty for random.sample
            preserve_indices = random.sample(range(len(fragments)), preserve_count)
        else:
            preserve_indices = []

        ordered_fragments_temp = [np.array([])] * len(fragments) # Inizializza con array vuoti
        preserved_map = {idx: frag for idx, frag in enumerate(fragments) if idx in preserve_indices}

        remaining_fragments_list = [fragments[i] for i in range(len(fragments)) if i not in preserve_indices]
        random.shuffle(remaining_fragments_list)
        remaining_fragments_iter = iter(remaining_fragments_list)


        for i in range(len(fragments)):
            if i in preserve_indices:
                ordered_fragments_temp[i] = preserved_map[i]
            else:
                try:
                    ordered_fragments_temp[i] = next(remaining_fragments_iter)
                except StopIteration:
                    ordered_fragments_temp[i] = np.array([])

        fragments = [f for f in ordered_fragments_temp if f.size > 0] # Filtra frammenti null o vuoti

    else:
        random.shuffle(fragments)

    if len(fragments) == 0:
        return np.array([])

    result = fragments[0]
    fade_samples = int(0.05 * sr)

    for fragment in fragments[1:]:
        if result.size == 0:
            result = fragment
            continue

        if fragment.size == 0:
            continue

        current_fade_samples = min(fade_samples, result.size, fragment.size)

        if current_fade_samples > 0:
            fade_out = np.linspace(1, 0, current_fade_samples)
            fade_in = np.linspace(0, 1, current_fade_samples)

            overlap_result = result[-current_fade_samples:]
            overlap_fragment = fragment[:current_fade_samples]

            overlapped_section = overlap_result * fade_out + overlap_fragment * fade_in

            result = np.concatenate([result[:-current_fade_samples], overlapped_section, fragment[current_fade_samples:]])
        else:
            result = np.concatenate([result, fragment])

    return result

def musique_concrete(audio, sr, params):
    """Applica tecniche di musique concrète: granular synthesis e manipolazioni concrete"""
    grain_size = params.get('grain_size', 0.1)
    texture_density = params.get('texture_density', 1.0)
    chaos_level = params['chaos_level']

    if audio.size == 0:
        return np.array([])

    grain_samples = int(grain_size * sr)
    if grain_samples <= 0:
        return audio

    grains = []
    # Assicurati che l'iterazione sia valida
    step = grain_samples // 2 if grain_samples // 2 > 0 else 1
    if len(audio) < grain_samples: # Se audio è più corto di un grano, non può creare grani
        if audio.size > 0: # Se audio non è vuoto ma è corto, trattalo come un unico grano
            grains.append(audio)
        else:
            return np.array([])
    else:
        for i in range(0, len(audio) - grain_samples + 1, step):
            grain = audio[i:i + grain_samples]

            if grain.size == 0:
                continue

            try:
                window = signal.windows.gaussian(len(grain), std=len(grain)/6)
                grain = grain * window
            except Exception as e:
                st.warning(f"Gaussian window failed for grain in musique_concrete: {e}. Trace: {traceback.format_exc()}")
                grain = np.array([])

            if grain.size == 0:
                continue

            if random.random() < chaos_level / 3.0:
                grain = grain[::-1]

            if grain.size > 0 and random.random() < chaos_level / 3.0: # Aggiunto controllo size
                try:
                    shift = random.uniform(-12, 12)
                    grain = safe_pitch_shift(grain, sr, shift)
                except Exception as e:
                    st.warning(f"Pitch shift failed for grain in musique_concrete: {e}. Trace: {traceback.format_exc()}")
                    grain = np.array([])

            if grain.size > 0 and random.random() < chaos_level / 3.0: # Aggiunto controllo size
                try:
                    stretch = random.uniform(0.25, 4.0)
                    grain = safe_time_stretch(grain, stretch)
                except Exception as e:
                    st.warning(f"Time stretch failed for grain in musique_concrete: {e}. Trace: {traceback.format_exc()}")
                    grain = np.array([])

            if grain.size > 0:
                grains.append(grain)

    if len(grains) == 0:
        return np.array([])

    num_grains_output = int(len(grains) * texture_density)
    if num_grains_output <= 0:
        return np.array([])

    if num_grains_output > len(grains):
        extra_grains = random.choices(grains, k=num_grains_output - len(grains))
        grains.extend(extra_grains)
    else:
        if len(grains) > 0: # Ensure grains is not empty for random.sample
            grains = random.sample(grains, num_grains_output)
        else:
            return np.array([])


    max_length = int(len(audio) * (1 + texture_density * 0.5))
    if max_length <= 0:
        return np.array([])

    result = np.zeros(max_length)

    for grain in grains:
        if grain.size == 0:
            continue

        if grain.size < max_length:
            start_pos = random.randint(0, max_length - grain.size)
            end_pos = start_pos + grain.size
            if end_pos > max_length:
                grain = grain[:max_length - start_pos]
                end_pos = max_length
            result[start_pos:end_pos] += grain * random.uniform(0.3, 1.0)
        else:
            # If grain is longer than max_length, it needs to be clipped
            if max_length > 0:
                start_pos = random.randint(0, max_length // 2) if max_length // 2 >= 0 else 0
                clip_length = min(grain.size, max_length - start_pos)
                if clip_length > 0:
                    result[start_pos : start_pos + clip_length] += grain[:clip_length] * random.uniform(0.3, 1.0)
            # If max_length is 0, nothing can be added, result remains zeros

    if result.size > 0:
        # Pulisci da eventuali NaN o Inf prima della normalizzazione
        result[np.isnan(result)] = 0
        result[np.isinf(result)] = 0
        if np.max(np.abs(result)) > 0:
            result = result / np.max(np.abs(result)) * 0.8
        else:
            result = np.array([]) # Se l'audio è tutto zero, consideralo vuoto
    else:
        result = np.array([])

    return result
def decostruzione_postmoderna(audio, sr, params):
    """Decostruzione postmoderna ottimizzata e più leggera"""
    irony_level = params.get('irony_level', 0.5)  # Valore più basso di default
    context_shift = params.get('context_shift', 0.6)  # Valore più basso di default
    fragment_size = params['fragment_size']

    if audio.size == 0:
        return np.array([])

    fragment_samples = int(fragment_size * sr)
    if fragment_samples <= 0:
        return audio

    try:
        # Versione semplificata dell'analisi dell'energia
        # Usiamo solo RMS semplice, no frame analysis complessa
        window_size = min(1024, len(audio) // 10)  # Finestra più piccola
        if window_size <= 0:
            window_size = min(512, len(audio))
        
        if window_size > len(audio):
            # Audio troppo corto, trattalo come un singolo frammento
            fragments = [audio]
            fragment_types = ['important']
        else:
            # Calcola energia in modo semplificato
            num_windows = len(audio) // window_size
            energy_values = []
            
            for i in range(num_windows):
                start_idx = i * window_size
                end_idx = min(start_idx + window_size, len(audio))
                window_audio = audio[start_idx:end_idx]
                
                if window_audio.size > 0:
                    energy = np.sqrt(np.mean(window_audio**2))
                    energy_values.append((energy, start_idx))

            # Identifica solo alcuni punti ad alta energia (max 8 per limitare complessità)
            max_important_fragments = min(8, len(energy_values))
            if energy_values:
                energy_values.sort(key=lambda x: x[0], reverse=True)
                important_starts = [start for _, start in energy_values[:max_important_fragments]]
            else:
                important_starts = []

            # Crea frammenti
            fragments = []
            fragment_types = []

            # Frammenti importanti (limitati)
            for start_sample in important_starts:
                end_sample = min(start_sample + fragment_samples, len(audio))
                if end_sample > start_sample:
                    fragment = audio[start_sample:end_sample]
                    if fragment.size > 0:
                        fragments.append(fragment)
                        fragment_types.append('important')

            # Aggiungi solo alcuni frammenti casuali (max 6)
            max_random_fragments = min(6, max(1, int(len(fragments) * 0.5)))
            for _ in range(max_random_fragments):
                if len(audio) >= fragment_samples:
                    max_start = len(audio) - fragment_samples
                    if max_start > 0:
                        start = random.randint(0, max_start)
                        fragment = audio[start:start + fragment_samples]
                    else:
                        fragment = audio.copy()
                else:
                    fragment = audio.copy()
                
                if fragment.size > 0:
                    fragments.append(fragment)
                    fragment_types.append('random')

        if len(fragments) == 0:
            return audio

        # Processa frammenti con trasformazioni più leggere
        processed_fragments = []
        
        for i, (fragment, frag_type) in enumerate(zip(fragments, fragment_types)):
            if fragment.size == 0:
                continue

            processed_frag = fragment.copy()

            # Trasformazioni ironiche semplificate (meno probabilità, meno aggressive)
            if frag_type == 'important' and random.random() < irony_level * 0.4:  # Ridotta probabilità
                transform_choice = random.choice([
                    'reverse',      # Inversione temporale (veloce)
                    'volume_down',  # Riduzione volume (veloce)
                    'fade_out'      # Fade out (veloce)
                ])
                
                try:
                    if transform_choice == 'reverse':
                        processed_frag = processed_frag[::-1]
                    elif transform_choice == 'volume_down':
                        processed_frag = processed_frag * 0.3
                    elif transform_choice == 'fade_out':
                        fade = np.linspace(1, 0.2, len(processed_frag))
                        processed_frag = processed_frag * fade
                        
                except Exception as e:
                    st.warning(f"Trasformazione ironica fallita: {e}")
                    processed_frag = fragment

            # Context shift più leggero
            if processed_frag.size > 0 and random.random() < context_shift * 0.3:  # Ridotta probabilità
                effect_choice = random.choice([
                    'fade_in',    # Fade in (veloce)
                    'light_noise' # Rumore leggero (veloce)
                ])
                
                try:
                    if effect_choice == 'fade_in':
                        fade = np.linspace(0.3, 1, len(processed_frag))
                        processed_frag = processed_frag * fade
                    elif effect_choice == 'light_noise':
                        noise_level = 0.01  # Molto leggero
                        noise = np.random.normal(0, noise_level, len(processed_frag))
                        processed_frag = processed_frag + noise
                        
                except Exception as e:
                    st.warning(f"Context shift fallito: {e}")
                    processed_frag = fragment

            if processed_frag.size > 0:
                processed_fragments.append(processed_frag)

        if len(processed_fragments) == 0:
            return audio

        # Riassemblaggio semplificato - senza sorting complesso per energia
        # Semplicemente mescola e concatena
        random.shuffle(processed_fragments)
        
        # Limita il numero di frammenti per evitare audio troppo lunghi
        max_final_fragments = min(12, len(processed_fragments))
        final_fragments = processed_fragments[:max_final_fragments]

        # Concatenazione con crossfade più semplice
        if len(final_fragments) == 0:
            return audio
       # Concatenazione con crossfade più semplice
        if len(final_fragments) == 0:
            return audio
        
        if len(final_fragments) == 1:
            return final_fragments[0]
        
        # Concatena i frammenti con crossfade leggero
        result = final_fragments[0].copy()
        
        for i in range(1, len(final_fragments)):
            next_fragment = final_fragments[i]
            
            # Crossfade semplice solo se entrambi i frammenti hanno dimensione sufficiente
            crossfade_samples = min(256, len(result) // 4, len(next_fragment) // 4)
            
            if crossfade_samples > 0 and len(result) >= crossfade_samples and len(next_fragment) >= crossfade_samples:
                # Applica crossfade
                fade_out = np.linspace(1, 0, crossfade_samples)
                fade_in = np.linspace(0, 1, crossfade_samples)
                
                # Modifica la coda del risultato esistente
                result[-crossfade_samples:] *= fade_out
                
                # Prepara l'inizio del prossimo frammento
                next_start = next_fragment[:crossfade_samples] * fade_in
                
                # Sovrapponi le parti con crossfade
                result[-crossfade_samples:] += next_start
                
                # Aggiungi il resto del frammento
                if len(next_fragment) > crossfade_samples:
                    result = np.concatenate([result, next_fragment[crossfade_samples:]])
            else:
                # Concatenazione semplice senza crossfade
                result = np.concatenate([result, next_fragment])
        
        # Normalizzazione finale per evitare clipping
        if len(result) > 0:
            max_val = np.max(np.abs(result))
            if max_val > 1.0:
                result = result / max_val * 0.95
        
        return result
        
    except Exception as e:
        st.warning(f"Errore nella decostruzione postmoderna: {e}")
        return audio

def decomposizione_creativa(audio, sr, params):
    """
    Decomposizione Creativa:
    Focus su discontinuità e shift emotivi, trasformazioni basate sull'analisi degli onset.
    Genera variazioni espressive intense.
    """
    discontinuity = params.get('discontinuity', 1.0)
    emotional_shift = params.get('emotional_shift', 0.8)
    fragment_size = params['fragment_size']
    chaos_level = params['chaos_level']

    if audio.size == 0:
        return np.array([])

    # Analisi degli onset per identificare i punti di "rottura" o importanza
    try:
        onset_frames = librosa.onset.onset_detect(y=audio, sr=sr, units='frames')
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    except Exception as e:
        st.warning(f"Errore nel rilevamento degli onset per decomposizione_creativa: {e}")
        onset_times = np.array([]) # Fallback a array vuoto

    processed_fragments = []
    current_time = 0.0

    # Determina i punti di taglio basati su onset e fragment_size
    cut_points = sorted(list(set([0] + [int(t * sr) for t in onset_times] + [int(current_time + fragment_size * sr * random.uniform(0.5, 1.5)) for _ in range(int(len(audio) / (fragment_size * sr * 2)))])))
    cut_points = [p for p in cut_points if p < len(audio)]
    if len(audio) not in cut_points:
        cut_points.append(len(audio))
    cut_points = sorted(list(set(cut_points)))

    for i in range(len(cut_points) - 1):
        start_sample = cut_points[i]
        end_sample = cut_points[i+1]
        
        if end_sample <= start_sample:
            continue

        fragment = audio[start_sample:end_sample].copy()
        if fragment.size == 0:
            continue

        # Applica discontinuità: salta o silenzia frammenti casualmente
        if random.random() < discontinuity * 0.1: # Bassa probabilità di silenziare o saltare
            if random.random() < 0.5: # 50% di probabilità di silenziare
                fragment = np.zeros_like(fragment)
            else: # 50% di probabilità di saltare (non aggiungerlo ai processed_fragments)
                continue

        # Applica shift emotivi: variazioni di pitch e tempo più pronunciate
        if random.random() < emotional_shift * 0.4: # Probabilità di applicare shift emotivo
            if random.random() < 0.5: # Pitch shift
                shift_steps = random.uniform(-12 * emotional_shift, 12 * emotional_shift)
                fragment = safe_pitch_shift(fragment, sr, shift_steps)
            else: # Time stretch
                stretch_rate = random.uniform(1 - 0.5 * emotional_shift, 1 + 0.5 * emotional_shift)
                fragment = safe_time_stretch(fragment, stretch_rate)
        
        # Aggiungi un tocco di caos generale
        if fragment.size > 0 and random.random() < chaos_level * 0.1:
            fragment = fragment[::-1] # Inversione leggera

        if fragment.size > 0:
            processed_fragments.append(fragment)

    if not processed_fragments:
        return np.array([])

    # Riassembla i frammenti
    final_audio = np.concatenate(processed_fragments)

    # Normalizzazione finale
    if final_audio.size > 0:
        max_val = np.max(np.abs(final_audio))
        if max_val > 0:
            final_audio = final_audio / max_val * 0.95
    return final_audio

def random_chaos(audio, sr, params):
    """
    Random Chaos: Ogni esecuzione è completamente diversa.
    Operazioni casuali estreme, risultati imprevedibili e sperimentali.
    """
    chaos_level = params['chaos_level']
    # fragment_size non viene usato direttamente in questa implementazione estrema
    # structure_preservation non viene usato, dato che è "chaos"

    if audio.size == 0:
        return np.array([])

    processed_audio = audio.copy()
    current_sr = sr # Mantieni traccia del sample rate che può cambiare con resampling

    # Operazioni casuali con probabilità basata sul chaos_level
    # Pitch Shift estremo
    if random.random() < 0.5 * chaos_level:
        shift_steps = random.uniform(-36, 36) # Fino a 3 ottave su/giù
        processed_audio = safe_pitch_shift(processed_audio, current_sr, shift_steps)
        if processed_audio.size == 0: return np.array([]) # Esci se diventa vuoto

    # Time Stretch estremo
    if random.random() < 0.5 * chaos_level:
        stretch_rate = random.uniform(0.05, 20.0) # Estrema compressione o dilatazione
        processed_audio = safe_time_stretch(processed_audio, stretch_rate)
        if processed_audio.size == 0: return np.array([]) # Esci se diventa vuoto

    # Inversione casuale (totale o di sezioni)
    if random.random() < 0.3 * chaos_level:
        if processed_audio.size > current_sr * 2 and random.random() < 0.5: # Inverti solo una sezione se abbastanza lunga
            start_idx = random.randint(0, max(0, processed_audio.size - int(current_sr * random.uniform(0.5, 5.0))))
            end_idx = min(processed_audio.size, start_idx + int(current_sr * random.uniform(0.5, 5.0)))
            if end_idx > start_idx:
                processed_audio[start_idx:end_idx] = processed_audio[start_idx:end_idx][::-1]
        else: # Inverti tutto
            processed_audio = processed_audio[::-1]
        if processed_audio.size == 0: return np.array([])

    # Aggiunta di rumore bianco o impulso casuale
    if random.random() < 0.4 * chaos_level:
        if processed_audio.size > 0:
            noise_amplitude = random.uniform(0.01, 0.2) * chaos_level
            noise = np.random.normal(0, noise_amplitude, processed_audio.size)
            processed_audio = processed_audio + noise
        if processed_audio.size == 0: return np.array([])

    # Resampling casuale (cambia il "colore" del suono)
    if random.random() < 0.3 * chaos_level:
        new_sr_factor = random.uniform(0.2, 5.0) # Cambia il sample rate fino a 5x
        new_sr = int(current_sr * new_sr_factor)
        if new_sr > 0 and processed_audio.size > 0:
            try:
                processed_audio = librosa.resample(y=processed_audio, orig_sr=current_sr, target_sr=new_sr)
                current_sr = new_sr # Aggiorna il sample rate corrente
            except Exception as e:
                st.warning(f"Errore nel resampling casuale: {e}")
                # Continua con l'audio non risamplingato
        if processed_audio.size == 0: return np.array([])

    # Frammentazione e rimescolamento casuale estremo
    if random.random() < 0.6 * chaos_level:
        if processed_audio.size > 0:
            # Crea frammenti casuali di dimensioni variabili
            chaos_fragments = []
            max_fragment_len = int(current_sr * random.uniform(0.1, 5.0)) # Frammenti da 0.1 a 5 sec
            current_pos = 0
            while current_pos < processed_audio.size:
                frag_len = min(random.randint(int(current_sr * 0.05), max_fragment_len), processed_audio.size - current_pos)
                if frag_len <= 0: break
                chaos_fragments.append(processed_audio[current_pos : current_pos + frag_len])
                current_pos += frag_len + int(current_sr * random.uniform(0, 0.5 * chaos_level)) # Aggiungi pause casuali

            random.shuffle(chaos_fragments) # Rimescola completamente
            if chaos_fragments:
                processed_audio = np.concatenate(chaos_fragments)
            else:
                processed_audio = np.array([])
        if processed_audio.size == 0: return np.array([])


    # Normalizzazione finale per evitare clipping
    if processed_audio.size > 0:
        max_val = np.max(np.abs(processed_audio))
        if max_val > 0: # Evita divisione per zero
            processed_audio = processed_audio / max_val * 0.95 # Normalizza al 95% del massimo
    return processed_audio

# Logica principale per processare l'audio
if uploaded_file is not None:
    try:
        # Carica il file audio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        # Leggi l'audio
        audio, sr = librosa.load(tmp_file_path, sr=None) 
        
        # Mostra informazioni del file originale
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Durata", f"{len(audio)/sr:.2f} sec")
        with col2:
            st.metric("Sample Rate", f"{sr} Hz")
        with col3:
            st.metric("Canali", "Mono")

        # Player per l'audio originale (rimossa emoji)
        st.subheader("Audio Originale")
        st.audio(uploaded_file, format='audio/wav')

        # Pulsante per processare
        if st.button("🎭 SCOMPONI E RICOMPONI", type="primary", use_container_width=True):
            with st.spinner(f"Applicando {decomposition_method}..."):
                
                # Prepara parametri
                params = {
                    'fragment_size': fragment_size,
                    'chaos_level': chaos_level,
                    'structure_preservation': structure_preservation
                }

                # Aggiungi parametri specifici per metodo
                if decomposition_method == "cut_up_sonoro":
                    params.update({
                        'cut_randomness': cut_randomness,
                        'reassembly_style': reassembly_style
                    })
                elif decomposition_method == "remix_destrutturato":
                    params.update({
                        'beat_preservation': beat_preservation,
                        'melody_fragmentation': melody_fragmentation
                    })
                elif decomposition_method == "musique_concrete":
                    params.update({
                        'grain_size': grain_size,
                        'texture_density': texture_density
                    })
                elif decomposition_method == "decostruzione_postmoderna":
                    params.update({
                        'irony_level': irony_level,
                        'context_shift': context_shift
                    })
                elif decomposition_method == "decomposizione_creativa":
                    params.update({
                        'discontinuity': discontinuity,
                        'emotional_shift': emotional_shift
                    })

                # Applica il metodo di decomposizione selezionato
                processed_audio = np.array([]) # Inizializza per sicurezza
                if decomposition_method == "cut_up_sonoro":
                    processed_audio = cut_up_sonoro(audio, sr, params)
                elif decomposition_method == "remix_destrutturato":
                    processed_audio = remix_destrutturato(audio, sr, params)
                elif decomposition_method == "musique_concrete":
                    processed_audio = musique_concrete(audio, sr, params)
                elif decomposition_method == "decostruzione_postmoderna":
                    processed_audio = decostruzione_postmoderna(audio, sr, params)
                elif decomposition_method == "decomposizione_creativa":
                    processed_audio = decomposizione_creativa(audio, sr, params)
                elif decomposition_method == "random_chaos":
                    processed_audio = random_chaos(audio, sr, params)

                # Verifica che l'elaborazione sia andata a buon fine
                if processed_audio.size == 0:
                    st.error("❌ Elaborazione fallita - audio risultante vuoto. Prova a modificare i parametri o a usare un file diverso.")
                else:
                    # Salva l'audio processato
                    processed_tmp_path = "" # Inizializza fuori dal blocco try per essere accessibile dal finally
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
                            # Assicurati che le lunghezze siano compatibili per FFT, altrimenti calcola separatamente
                            min_len = min(len(processed_audio), len(audio))
                            if min_len > 0:
                                # Calcola la differenza media dello spettro di potenza
                                spec_orig = np.abs(np.fft.fft(audio[:min_len]))
                                spec_proc = np.abs(np.fft.fft(processed_audio[:min_len]))
                                spectral_diff = np.mean(np.abs(spec_proc - spec_orig))
                            else:
                                spectral_diff = 0.0
                            st.metric("Variazione Spettrale", f"{spectral_diff:.2e}") # Rinominato per chiarezza

                        # Player per l'audio processato (rimossa emoji)
                        st.subheader("Audio Decomposto e Ricomposto")
                        
                        # Leggi il file processato per il player
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
                        
                        filename = f"{uploaded_file.name.split('.')[0]}_{method_names[decomposition_method]}.wav"
                        
                        st.download_button(
                            label="💾 Scarica Audio Decomposto",
                            data=audio_bytes,
                            file_name=filename,
                            mime="audio/wav",
                            use_container_width=True
                        )

                        # Mappa delle descrizioni delle tecniche
                        technique_descriptions = {
                            "cut_up_sonoro": """
                            Il metodo **"Cut-up Sonoro"** si ispira a una tecnica letteraria dove il testo viene frammentato e riassemblato. Il brano viene diviso in sezioni, che vengono poi **tagliate e riassemblate in un ordine casuale o predefinito** (come inversione o palindromo). Questo crea un effetto di collage sonoro, dove il significato originale è destrutturato per rivelare nuove connessioni e pattern imprevedibili. La musica diventa una forma di testo decostruito.
                            """,
                            "remix_destrutturato": """
                            Il **"Remix Destrutturato"** mira a mantenere alcuni elementi riconoscibili del brano originale (come battiti o frammenti melodici), ma li **ricontestualizza in un nuovo arrangiamento**. Vengono applicate manipolazioni come pitch shift e time stretch ai frammenti, che poi vengono riorganizzati per creare un'esperienza d'ascolto che è sia familiare che sorprendentemente nuova, quasi una reinterpretazione.
                            """,
                            "musique_concrete": """
                            La **"Musique Concrète"** si basa sui principi di manipolazione sonora. Questo metodo si concentra sulla **manipolazione di "grani" sonori** (piccolissimi frammenti dell'audio) attraverso tecniche come la sintesi granulare, l'inversione e il pitch/time shift. Il risultato è una texture sonora astratta, spesso non riconoscibile come musica nel senso tradizionale, che esplora le proprietà intrinseche del suono.
                            """,
                            "decostruzione_postmoderna": """
                            La **"Decostruzione Postmoderna"** applica un approccio critico al brano, **decostruendone il significato musicale originale** attraverso l'uso di ironia e spostamenti di contesto. Frammenti "importanti" vengono manipolati in modi inaspettati (es. volume ridotto, inversione), e vengono introdotti elementi di rottura o rumore. L'obiettivo è provocare una riflessione critica sull'opera e sulla sua percezione.
                            """,
                            "decomposizione_creativa": """
                            La **"Decomposizione Creativa"** si focalizza sulla creazione di **discontinuità e "shift emotivi"** intensi. Utilizzando l'analisi degli onset (punti di attacco del suono), il brano viene frammentato in modo dinamico. I frammenti vengono poi trasformati con variazioni pronunciate di pitch e tempo, e alcuni possono essere silenziati o saltati per generare un'esperienza sonora ricca di espressività e rotture inattese.
                            """,
                            "random_chaos": """
                            Il metodo **"Random Chaos"** è progettato per produrre **risultati altamente imprevedibili e sperimentali**. Ogni esecuzione è unica. Vengono applicate operazioni casuali ed estreme come pitch shift e time stretch massivi, inversioni casuali di sezioni, aggiunta di rumore e resampling. Questo metodo esplora i limiti della manipolazione audio, portando a trasformazioni radicali e spesso disorientanti.
                            """
                        }
                        
                        selected_method_description = technique_descriptions.get(decomposition_method, "Nessuna descrizione disponibile per questo metodo.")

                        # --- Nuova Sezione: Sintesi Artistica della Decomposizione ---
                        st.subheader("Sintesi Artistica della Decomposizione")
                        
                        artistic_summary = ""
                        # Generazione del riassunto artistico in base al metodo
                        if decomposition_method == "cut_up_sonoro":
                            artistic_summary = f"""
                            Con il metodo del **"Cut-up Sonoro"**, il brano originale è stato smembrato e ricombinato, trasformandosi in un'opera di arte sonora ispirata a tecniche di collage e frammentazione. Ogni frammento, lungo circa {fragment_size:.1f} secondi, è stato trattato come un elemento in una composizione decostruita.
                            Il livello di casualità dei tagli ({cut_randomness:.1f}) e lo stile di riassemblaggio ('{reassembly_style}') hanno permesso di **dislocare il significato musicale** originale, creando inaspettate giustapposizioni e ritmi frammentati. Il risultato è un collage sonoro che sfida la percezione tradizionale, invitando l'ascoltatore a trovare nuove narrazioni all'interno della frammentazione.
                            """
                        elif decomposition_method == "remix_destrutturato":
                            artistic_summary = f"""
                            Attraverso il **"Remix Destrutturato"**, l'essenza del brano originale è stata catturata e rielaborata in una forma nuova e sorprendente. Pur mantenendo una certa fedeltà al ritmo (conservazione del battito del {beat_preservation*100:.0f}%), la melodia è stata frammentata e manipolata.
                            I frammenti, di circa {fragment_size:.1f} secondi, hanno subito alterazioni di pitch e tempo (frammentazione melodia: {melody_fragmentation:.1f}), ricollocando gli elementi sonori in un **paesaggio acustico reinventato**. Questo remix non è una semplice variazione, ma una vera e propria decostruzione che riassembla gli ingredienti in un'esperienza d'ascolto che è al contempo familiare ed estranea.
                            """
                        elif decomposition_method == "musique_concrete":
                            artistic_summary = f"""
                            Con la tecnica della **"Musique Concrète"**, il brano è stato ridotto ai suoi "grani" sonori più elementari (dimensione dei grani: {grain_size:.2f} secondi). Questi micro-frammenti sono stati manipolati, rovesciati, stirati e compressi, per poi essere ricombinati con una densità ({texture_density:.1f}) che crea una nuova tessitura.
                            Il risultato è un'opera sonora astratta che **esplora le qualità timbriche intrinseche del suono**, al di là della sua organizzazione musicale originale. L'ascolto si trasforma in un viaggio attraverso paesaggi sonori inusuali, dove il timbro e la consistenza diventano i veri protagonisti.
                            """
                        elif decomposition_method == "decostruzione_postmoderna":
                            artistic_summary = f"""
                            La **"Decostruzione Postmoderna"** ha applicato un filtro concettuale al brano, interrogandone il significato e la percezione. Con frammenti di {fragment_size:.1f} secondi, abbiamo esplorato l'ironia ({irony_level:.1f}) e gli spostamenti di contesto ({context_shift:.1f}).
                            Elementi riconoscibili sono stati trattati in modo inaspettato (es. alterazioni di volume o inversioni rapide), e sono stati introdotti sottili elementi di rottura. L'obiettivo non è solo trasformare il suono, ma anche provocare una **riflessione critica sull'opera e sulla sua fruizione**, trasformando il familiare in qualcosa di leggermente destabilizzante ma affascinante.
                            """
                        elif decomposition_method == "decomposizione_creativa":
                            artistic_summary = f"""
                            Attraverso la **"Decomposizione Creativa"**, il brano è stato frammentato in base ai suoi punti di attacco (onset), consentendo interventi mirati che generano forti discontinuità ({discontinuity:.1f}) e shift emotivi ({emotional_shift:.1f}).
                            I frammenti, di circa {fragment_size:.1f} secondi, sono stati soggetti a intense variazioni di pitch e tempo, e alcuni sono stati deliberatamente silenziati o saltati. Il risultato è un'esperienza sonora **ricca di colpi di scena e improvvisi cambi di umore**, un flusso e riflusso di stati emotivi e tensioni acustiche.
                            """
                        elif decomposition_method == "random_chaos":
                            artistic_summary = f"""
                            Il **"Random Chaos"** ha spinto il brano originale nei suoi limiti estremi, creando un'opera unica e imprevedibile. Con un livello di caos di {chaos_level:.1f}, l'audio è stato sottoposto a **trasformazioni radicali e casuali**, come pitch shift e time stretch estremi, inversioni di sezioni e l'introduzione di rumori.
                            Il risultato è un'esplorazione sonora che sfugge a qualsiasi classificazione, un viaggio in un **paesaggio acustico alieno** dove l'originale è appena un eco lontano, e ogni ascolto rivela nuove, sorprendenti anomalie.
                            """
                        st.markdown(artistic_summary)

                        st.markdown("---")
                        st.markdown(f"**Descrizione della Tecnica Applicata:**") # Titolo più specifico per la spiegazione della tecnica
                        st.markdown(selected_method_description) # Descrizione della tecnica (già esistente)

                        st.markdown("---")
                        st.markdown("### Riepilogo dei Cambiamenti Quantitativi:") # Titolo più specifico

                        analysis_text = f"""
                        * **Durata:** L'audio originale, di **{original_duration:.2f} secondi**, è stato trasformato in un brano di **{new_duration:.2f} secondi**. Questo indica un {'allungamento' if new_duration > original_duration else 'accorciamento'} di **{abs(new_duration - original_duration):.2f} secondi**.
                        * **Energia RMS (Volume Percepito):** Il livello di energia RMS (Root Mean Square), che è un indicatore del volume percepito, ha avuto una variazione di **{processed_rms - original_rms:.4f}**. Questo significa che il suono risultante è generalmente {'più forte' if processed_rms > original_rms else 'più debole' if processed_rms < original_rms else 'simile'} in termini di volume medio.
                        * **Variazione Spettrale:** La variazione spettrale di **{spectral_diff:.2e}** quantifica quanto è cambiato il "colore" o la distribuzione delle frequenze rispetto all'originale. Un valore più alto indica una trasformazione più significativa del timbro e della texture sonora.

                        Questi cambiamenti riflettono l'impatto dei parametri scelti (`Dimensione Frammenti: {fragment_size}s`, `Livello di Chaos: {chaos_level}`, `Conservazione Struttura: {structure_preservation}`).
                        """
                        st.markdown(analysis_text)


                        # Visualizzazione forme d'onda (opzionale)
                        with st.expander("📊 Confronto Forme d'Onda"):
                            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
                            
                            # Originale
                            time_orig = np.linspace(0, len(audio)/sr, len(audio))
                            ax1.plot(time_orig, audio, color='blue', alpha=0.7)
                            ax1.set_title("Forma d'Onda Originale")
                            ax1.set_xlabel("Tempo (sec)")
                            ax1.set_ylabel("Ampiezza")
                            ax1.grid(True, alpha=0.3)
                            
                            # Processato
                            time_proc = np.linspace(0, len(processed_audio)/sr, len(processed_audio))
                            ax2.plot(time_proc, processed_audio, color='red', alpha=0.7)
                            ax2.set_title(f"Forma d'Onda Decomposta ({method_names[decomposition_method]})")
                            ax2.set_xlabel("Tempo (sec)")
                            ax2.set_ylabel("Ampiezza")
                            ax2.grid(True, alpha=0.3)
                            
                            plt.tight_layout()
                            st.pyplot(fig)

                        # Analisi spettrale (opzionale)
                        with st.expander("🎼 Analisi Spettrale"):
                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                            
                            # Spettrogramma originale
                            D_orig = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
                            librosa.display.specshow(D_orig, y_axis='hz', x_axis='time', sr=sr, ax=ax1)
                            ax1.set_title("Spettrogramma Originale")
                            ax1.set_xlabel("Tempo (sec)")
                            ax1.set_ylabel("Frequenza (Hz)")
                            
                            # Spettrogramma processato
                            D_proc = librosa.amplitude_to_db(np.abs(librosa.stft(processed_audio)), ref=np.max)
                            librosa.display.specshow(D_proc, y_axis='hz', x_axis='time', sr=sr, ax=ax2)
                            ax2.set_title("Spettrogramma Decomposto")
                            ax2.set_xlabel("Tempo (sec)")
                            ax2.set_ylabel("Frequenza (Hz)")
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                    finally:
                        # Cleanup files temporanei
                        try:
                            if os.path.exists(tmp_file_path):
                                os.unlink(tmp_file_path)
                            if processed_tmp_path and os.path.exists(processed_tmp_path):
                                os.unlink(processed_tmp_path)
                        except Exception as e:
                            st.error(f"Errore durante la pulizia dei file temporanei: {e}")

    except Exception as e:
        st.error(f"❌ Errore nel processamento: {str(e)}")
        st.error(f"Dettagli: {traceback.format_exc()}")

else:
    # Messaggio quando non c'è file caricato
    st.info("👆 Carica un file audio per iniziare la decomposizione")
    
    # Istruzioni d'uso
    with st.expander("📖 Come usare MusicDecomposer"):
        st.markdown("""
        ### Metodi di Decomposizione:

        **Cut-up Sonoro**
        - Ispirati a una tecnica letteraria di taglio e riassemblaggio
        - Taglia l'audio in frammenti e li riassembla casualmente
        - Ottimo per creare collage sonori sperimentali

        **Remix Destrutturato** - Mantiene elementi riconoscibili ma li ricontestualizza
        - Preserva parzialmente il ritmo originale
        - Ideale per remix creativi e riarrangiamenti

        **Musique Concrète**
        - Basato sui principi di manipolazione sonora
        - Utilizza granular synthesis e manipolazioni concrete
        - Perfetto per texture sonore astratte

        **Decostruzione Postmoderna**
        - Decostruisce il significato musicale originale
        - Applica ironia e spostamenti di contesto
        - Crea riflessioni critiche sull'opera originale

        **Decomposizione Creativa**
        - Focus su discontinuità e shift emotivi
        - Trasformazioni basate sull'analisi degli onset
        - Genera variazioni espressive intense

        **Random Chaos**
        - Ogni esecuzione è completamente diversa
        - Operazioni casuali estreme
        - Risultati imprevedibili e sperimentali

        ### Parametri:
        - **Dimensione Frammenti**: Quanto grandi sono i pezzi tagliati
        - **Livello di Chaos**: Intensità delle trasformazioni
        - **Conservazione Struttura**: Quanto mantenere dell'originale
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

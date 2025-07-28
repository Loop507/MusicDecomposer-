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
            "cut_up_sonoro": "Cut-up Sonoro (Burroughs)",
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
    """Implementa la tecnica cut-up di Burroughs applicata all'audio"""
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

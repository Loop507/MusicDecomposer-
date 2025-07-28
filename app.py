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
    st.header(" Controlli Decomposizione")

    decomposition_method = st.selectbox(
        " Metodo di Decomposizione",
        [
            "cut_up_sonoro",
            "remix_destrutturato",
            "musique_concrete",
            "decostruzione_postmoderna",
            "decomposizione_creativa",
            "random_chaos"
        ],
        format_func=lambda x: {
            "cut_up_sonoro": " Cut-up Sonoro (Burroughs)",
            "remix_destrutturato": " Remix Destrutturato",
            "musique_concrete": " Musique Concrète",
            "decostruzione_postmoderna": " Decostruzione Postmoderna",
            "decomposizione_creativa": " Decomposizione Creativa",
            "random_chaos": " Random Chaos"
        }[x]
    )

    st.markdown("---")

    # Parametri generali
    fragment_size = st.slider(" Dimensione Frammenti (sec)", 0.1, 5.0, 1.0, 0.1)
    chaos_level = st.slider(" Livello di Chaos", 0.1, 3.0, 1.0, 0.1)
    structure_preservation = st.slider(" Conservazione Struttura", 0.0, 1.0, 0.3, 0.1)

    st.markdown("---")

    # Parametri specifici per metodo
    if decomposition_method == "cut_up_sonoro":
        st.subheader(" Cut-up Parameters")
        cut_randomness = st.slider(" Casualità Tagli", 0.1, 1.0, 0.7, 0.1)
        reassembly_style = st.selectbox(" Stile Riassemblaggio",
                                       ["random", "reverse", "palindrome", "spiral"])


    elif decomposition_method == "remix_destrutturato":
        st.subheader(" Remix Parameters")
        beat_preservation = st.slider(" Conserva Ritmo", 0.0, 1.0, 0.4, 0.1)
        melody_fragmentation = st.slider(" Frammentazione Melodia", 0.1, 3.0, 1.5, 0.1)

    elif decomposition_method == "musique_concrete":
        st.subheader(" Concrete Parameters")
        grain_size = st.slider(" Dimensione Grani", 0.01, 0.5, 0.1, 0.01)
        texture_density = st.slider(" Densità Texture", 0.1, 3.0, 1.0, 0.1)

    elif decomposition_method == "decostruzione_postmoderna":
        st.subheader(" Postmodern Parameters")
        irony_level = st.slider(" Livello Ironia", 0.1, 2.0, 1.0, 0.1)
        context_shift = st.slider(" Shift di Contesto", 0.1, 2.0, 1.2, 0.1)

    elif decomposition_method == "decomposizione_creativa":
        st.subheader(" Creative Parameters")
        discontinuity = st.slider(" Discontinuità", 0.1, 2.0, 1.0, 0.1)
        emotional_shift = st.slider(" Shift Emotivo", 0.1, 2.0, 0.8, 0.1)

# Upload file
uploaded_file = st.file_uploader(
    " Carica il tuo brano da decomporre",
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
                    fragment = librosa.effects.pitch_shift(fragment, sr=sr, n_steps=shift_steps)
                except Exception as e:
                    st.warning(f"Pitch shift failed for fragment in remix_destrutturato: {e}. Trace: {traceback.format_exc()}")
                    fragment = np.array([]) # Marca il frammento come vuoto in caso di errore

            if fragment.size > 0 and random.random() < melody_fragmentation / 3.0: # Aggiunto controllo size
                try:
                    stretch_factor = random.uniform(0.7, 1.4)
                    fragment = librosa.effects.time_stretch(fragment, rate=stretch_factor)
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
                    grain = librosa.effects.pitch_shift(grain, sr=sr, n_steps=shift)
                except Exception as e:
                    st.warning(f"Pitch shift failed for grain in musique_concrete: {e}. Trace: {traceback.format_exc()}")
                    grain = np.array([])

            if grain.size > 0 and random.random() < chaos_level / 3.0: # Aggiunto controllo size
                try:
                    stretch = random.uniform(0.25, 4.0)
                    grain = librosa.effects.time_stretch(grain, rate=stretch)
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
    """Decostruzione ironica e postmoderna del brano"""
    irony_level = params.get('irony_level', 1.0)
    context_shift = params.get('context_shift', 1.2)
    fragment_size = params['fragment_size']

    if audio.size == 0:
        return np.array([])

    fragment_samples = int(fragment_size * sr)
    if fragment_samples <= 0:
        return audio

    hop_length = 512
    energy = np.array([])
    if len(audio) >= hop_length:
        try:
            energy = librosa.feature.rms(y=audio, hop_length=hop_length)[0]
        except Exception as e:
            st.warning(f"RMS calculation failed in decostruzione_postmoderna: {e}. Trace: {traceback.format_exc()}")
            energy = np.array([])

    important_frames = np.array([])
    if energy.size > 0:
        energy_threshold = np.percentile(energy, 70)
        important_frames = np.where(energy > energy_threshold)[0]

    important_times = np.array([])
    if important_frames.size > 0:
        important_times = librosa.frames_to_time(important_frames, sr=sr, hop_length=hop_length)

    fragments = []
    fragment_types = []

    for t in important_times:
        start_sample = int(t * sr)
        end_sample = min(start_sample + fragment_samples, len(audio))

        if end_sample > start_sample:
            fragment = audio[start_sample:end_sample]
            if fragment.size > 0:
                fragments.append(fragment)
                fragment_types.append('important')

    num_random = int(len(fragments) * 0.5)
    for _ in range(num_random):
        if len(audio) < fragment_samples:
            break
        start = random.randint(0, len(audio) - fragment_samples)
        fragment = audio[start:start + fragment_samples]
        if fragment.size > 0:
            fragments.append(fragment)
            fragment_types.append('random')

    if len(fragments) == 0:
        return np.array([])

    processed_fragments = []
    for i, (fragment, frag_type) in enumerate(zip(fragments, fragment_types)):
        if fragment.size == 0:
            continue

        if frag_type == 'important' and random.random() < irony_level / 2.0:
            ironic_transforms = [
                lambda x: x[::-1] if x.size > 0 else np.array([]),
                lambda x: librosa.effects.pitch_shift(x, sr=sr, n_steps=-12) if x.size > 0 else np.array([]),
                lambda x: librosa.effects.time_stretch(x, rate=0.25) if x.size > 0 else np.array([]),
                lambda x: x * 0.1 if x.size > 0 else np.array([]),
                # FIX: Limita la dimensione massima del tile per evitare OOM e gestisce frammenti piccoli
                lambda x: np.tile(x[:max(1, min(len(x) // 4, 1000))] if len(x) > 0 else np.array([]), 4) if len(x) > 0 else np.array([]),

            ]
            transform = random.choice(ironic_transforms)
            try:
                transformed_fragment = transform(fragment)
                if transformed_fragment.size > 0:
                    fragment = transformed_fragment
                else:
                    st.warning(f"Ironic transform produced empty fragment. Keeping original. Trace: {traceback.format_exc()}")
            except Exception as e:
                st.warning(f"Ironic transform failed for fragment: {e}. Keeping original. Trace: {traceback.format_exc()}")
                # Fallback to original fragment if error, or empty if original was empty
                fragment = fragment if fragment.size > 0 else np.array([])


        if fragment.size > 0 and random.random() < context_shift / 2.0: # Aggiunto controllo size
            context_effects = [
                lambda x: librosa.effects.pitch_shift(x, sr=sr, n_steps=random.uniform(-7, 7)) if x.size > 0 else np.array([]),
                lambda x: x * np.linspace(0, 1, len(x)) if len(x) > 0 else np.array([]),
                lambda x: x * np.linspace(1, 0, len(x)) if len(x) > 0 else np.array([]),
                lambda x: x + np.random.normal(0, 0.05, len(x)) if len(x) > 0 else np.array([]),
            ]
            effect = random.choice(context_effects)
            try:
                effected_fragment = effect(fragment)
                if effected_fragment.size > 0:
                    fragment = effected_fragment
                else:
                    st.warning(f"Context shift effect produced empty fragment. Keeping original. Trace: {traceback.format_exc()}")
            except Exception as e:
                st.warning(f"Context shift effect failed for fragment: {e}. Trace: {traceback.format_exc()}")
                fragment = fragment if fragment.size > 0 else np.array([])


        if fragment.size > 0:
            processed_fragments.append(fragment)

    if len(processed_fragments) == 0:
        return np.array([])

    fragment_energies = [np.mean(np.abs(frag)) for frag in processed_fragments if frag.size > 0]
    processed_fragments_filtered = [frag for frag in processed_fragments if frag.size > 0]

    if len(processed_fragments_filtered) == 0:
        return np.array([])

    sorted_indices = np.argsort(fragment_energies)

    result_order_indices = []

    high_energy = sorted_indices[-len(sorted_indices)//2:] if sorted_indices.size > 0 else np.array([])
    low_energy = sorted_indices[:len(sorted_indices)//2] if sorted_indices.size > 0 else np.array([])

    min_len_energy = min(len(high_energy), len(low_energy)) if len(high_energy) > 0 and len(low_energy) > 0 else 0

    if min_len_energy == 0 and sorted_indices.size > 0:
        result_order_indices.extend(list(sorted_indices))
    else:
        for i in range(min_len_energy):
            if high_energy.size > i and random.random() < 0.6:
                result_order_indices.append(high_energy[i])
            if low_energy.size > i:
                result_order_indices.append(low_energy[i])
            if high_energy.size > (len(high_energy) - (i+1)) and random.random() < 0.4:
                result_order_indices.append(high_energy[-(i+1)])

    result_fragments_final = [processed_fragments_filtered[i] for i in result_order_indices if i < len(processed_fragments_filtered)]

    if len(result_fragments_final) > 0:
        result = np.concatenate(result_fragments_final)
    else:
        result = np.array([])

    return result

def decomposizione_creativa(audio, sr, params):
    """Decomposizione che enfatizza discontinuità e nuove connessioni emotive"""
    discontinuity = params.get('discontinuity', 1.0)
    emotional_shift = params.get('emotional_shift', 0.8)
    fragment_size = params['fragment_size']

    if audio.size == 0:
        return np.array([])

    structure = analyze_audio_structure(audio, sr)
    chroma = structure['chroma']
    mfcc = structure['mfcc']
    spectral_centroids = structure['spectral_centroids']

    features = np.array([])
    if chroma.size > 0 and mfcc.size > 0:
        min_frames = min(chroma.shape[1], mfcc.shape[1])
        if min_frames > 0:
            mfcc_sliced = mfcc[:5, :min_frames] if mfcc.shape[0] >= 5 else mfcc[:, :min_frames]
            if mfcc_sliced.shape[1] > 0:
                features = np.vstack([chroma[:, :min_frames], mfcc_sliced])

    mood_labels = np.array([])
    if features.size > 0:
        try:
            features_scaled = StandardScaler().fit_transform(features.T)
            # Determina il numero di cluster
            n_clusters = min(8, features_scaled.shape[0] // 10) # n_clusters non deve essere maggiore del numero di campioni
            if n_clusters < 1:
                n_clusters = 1
            
            # Controllo aggiunto per n_clusters > n_samples (numero di righe in features_scaled)
            if features_scaled.shape[0] < n_clusters:
                n_clusters = features_scaled.shape[0] if features_scaled.shape[0] > 0 else 1 # Se samples < cluster, imposta cluster a samples

            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10) # n_init aggiunto per KMeans >= 1.2.0
            mood_labels = kmeans.fit_predict(features_scaled)
        except Exception as e:
            st.warning(f"Clustering fallito: {e}. Trace: {traceback.format_exc()}")
            mood_labels = np.array([])

    if mood_labels.size == 0:
        return audio

    fragment_samples = int(fragment_size * sr)
    if fragment_samples <= 0:
        return audio

    hop_length = 512
    frames_per_fragment = max(1, fragment_samples // hop_length)

    mood_fragments = {mood: [] for mood in np.unique(mood_labels)}

    max_mood_idx = len(mood_labels) - frames_per_fragment
    if max_mood_idx < 0:
        return audio

    for i in range(0, max_mood_idx + 1, frames_per_fragment):
        start_sample = i * hop_length
        end_sample = min(start_sample + fragment_samples, len(audio))
        if end_sample > start_sample:
            fragment = audio[start_sample:end_sample]
            if fragment.size > 0:
                mood_slice = mood_labels[i:i + frames_per_fragment]
                if mood_slice.size > 0:
                    mood_slice_list = [int(m) for m in mood_slice.tolist()]
                    counts = {}
                    for mood in mood_slice_list:
                        counts[mood] = counts.get(mood, 0) + 1
                    dominant_mood = max(counts, key=counts.get)
                    mood_fragments[dominant_mood].append(fragment)

    mood_fragments = {k: [f for f in v if f.size > 0] for k, v in mood_fragments.items()}
    mood_fragments = {k: v for k, v in mood_fragments.items() if v}

    if not mood_fragments:
        return audio

    result_fragments = []
    available_moods = [m for m in mood_fragments.keys() if mood_fragments[m]]
    if not available_moods: # Double check if after filtering, any mood has fragments
        return audio
    current_mood = random.choice(available_moods)

    total_expected_fragments = sum(len(frags) for frags in mood_fragments.values())
    max_iterations = total_expected_fragments * 2 if total_expected_fragments > 0 else 100
    iteration_count = 0

    # Condizione del while loop più robusta
    while iteration_count < max_iterations and any(len(v) > 0 for v in mood_fragments.values()):
        iteration_count += 1

        if current_mood in mood_fragments and mood_fragments[current_mood]:
            fragment = random.choice(mood_fragments[current_mood])

            # Fix: usa confronto per identità
            for idx, f in enumerate(mood_fragments[current_mood]):
                if f is fragment:
                    del mood_fragments[current_mood][idx]
                    break

            if fragment.size == 0:
                continue

            if random.random() < emotional_shift:
                transforms = [
                    lambda x: librosa.effects.pitch_shift(x, sr=sr, n_steps=random.uniform(-3, 3)) if x.size > 0 else np.array([]),
                    lambda x: x * np.power(np.linspace(0.3, 1, len(x)) if len(x) > 0 else np.array([]), random.uniform(0.5, 2)) if x.size > 0 else np.array([]),
                    lambda x: librosa.effects.time_stretch(x, rate=random.uniform(0.8, 1.3)) if x.size > 0 else np.array([])
                ]
                try:
                    transformed = random.choice(transforms)(fragment)
                    if transformed.size > 0:
                        fragment = transformed
                except Exception as e:
                    st.warning(f"Transform fallita per frammento: {e}. Trace: {traceback.format_exc()}")
                    # Fallback to original if transform fails, to avoid empty fragments
                    if fragment.size == 0: # If original was empty, keep it empty
                        fragment = np.array([])

            if fragment.size > 0:
                result_fragments.append(fragment)

        # Cambio mood casuale
        if random.random() < discontinuity / 2.0:
            available = [m for m in mood_fragments.keys() if mood_fragments[m]]
            if available:
                current_mood = random.choice(available)
            else: # Se non ci sono più mood con frammenti, esci o gestisci
                break # Nessun frammento da cui attingere, esci dal loop

        # Inserisce silenzio
        if random.random() < discontinuity / 4.0:
            silence = np.zeros(int(random.uniform(0.1, 0.5) * sr))
            if silence.size > 0:
                result_fragments.append(silence)

    result_fragments = [f for f in result_fragments if f.size > 0]
    if result_fragments:
        return np.concatenate(result_fragments)
    else:
        return audio

def random_chaos(audio, sr, params):
    """Chaos totale: combina tutti i metodi casualamente"""
    chaos_level = params['chaos_level']

    if audio.size == 0:
        return np.array([])

    methods = [
        cut_up_sonoro,
        remix_destrutturato,
        musique_concrete,
        decostruzione_postmoderna,
        decomposizione_creativa
    ]

    num_methods = min(len(methods), max(1, int(chaos_level)))
    chosen_methods = random.sample(methods, num_methods)

    result = audio.copy()
    for method in chosen_methods:
        if result.size == 0:
            break

        random_params = params.copy()
        random_params.update({
            'cut_randomness': random.uniform(0.3, 1.0),
            'reassembly_style': random.choice(['random', 'reverse', 'palindrome', 'spiral']),
            'beat_preservation': random.uniform(0.0, 0.8),
            'melody_fragmentation': random.uniform(0.5, 2.5),
            'grain_size': random.uniform(0.01, 0.3),
            'texture_density': random.uniform(0.5, 2.0),
            'irony_level': random.uniform(0.5, 2.0),
            'context_shift': random.uniform(0.5, 2.0),
            'discontinuity': random.uniform(0.5, 2.0),
            'emotional_shift': random.uniform(0.3, 1.5)
        })

        try:
            temp_result = method(result, sr, random_params)
            if temp_result is None or (isinstance(temp_result, np.ndarray) and temp_result.size == 0 and result.size > 0):
                st.warning(f"Metodo '{method.__name__}' ha prodotto un array vuoto/non valido. Mantenendo il risultato precedente. Trace: {traceback.format_exc()}")
            elif isinstance(temp_result, np.ndarray):
                result = temp_result
            else:
                st.warning(f"Metodo '{method.__name__}' ha prodotto un risultato di tipo inatteso. Mantenendo il risultato precedente. Trace: {traceback.format_exc()}")

        except Exception as e:
            st.warning(f"Errore in {method.__name__}: {e}. Dettagli tecnici: {e.__class__.__name__}: {e}. Mantenendo il risultato precedente. Trace: {traceback.format_exc()}")
            continue

    if result.size == 0:
        return audio

    return result

def process_audio(uploaded_file, method, params, initial_sr=None):
    """Processa l'audio con il metodo scelto"""

    original_audio_path_temp = None
    output_path = None
    sr = initial_sr
    audio_original = np.array([]) # Inizializza a vuoto per sicurezza

    try:
        # Crea un file temporaneo per l'audio originale dal BytesIO
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_input_file:
            tmp_input_file.write(uploaded_file.read())
            original_audio_path_temp = tmp_input_file.name

        # Carica audio e determina il sample rate
        audio_original, sr = librosa.load(original_audio_path_temp, sr=None, mono=False)

        if audio_original is None or audio_original.size == 0:
            st.error("Il file audio caricato è vuoto o non contiene dati validi dopo il caricamento.")
            return None, None, None # Restituisce None per output_path, sr, original_audio_path_temp

        # Processa l'audio
        processed_audio = np.array([])
        if audio_original.size > 0 and len(audio_original.shape) > 1: # Multicanale
            processed_channels = []
            for channel_idx in range(audio_original.shape[0]):
                channel_audio = audio_original[channel_idx]
                processed_channel = decompose_audio(channel_audio, sr, method, params)
                if processed_channel is not None and processed_channel.size > 0:
                    processed_channels.append(processed_channel)

            if len(processed_channels) == 0:
                st.error("La decomposizione ha prodotto risultati vuoti per tutti i canali.")
                return None, None, None

            # Assicurati che tutti i canali abbiano la stessa lunghezza minima prima di concatenare
            min_length = min(len(ch) for ch in processed_channels)
            processed_channels = [ch[:min_length] for ch in processed_channels]
            processed_audio = np.array(processed_channels)
        elif audio_original.size > 0: # Monocanale
            processed_audio = decompose_audio(audio_original, sr, method, params)
        else: # Audio vuoto all'inizio (dovrebbe essere già gestito dal controllo iniziale, ma per sicurezza)
            st.error("Il file audio caricato è vuoto o non contiene dati validi.")
            return None, None, None

        if processed_audio is None or processed_audio.size == 0:
            st.error("La decomposizione ha prodotto un file audio vuoto.")
            return None, None, None

        # Salva l'audio processato in un file temporaneo
        output_path = tempfile.mktemp(suffix='.wav')
        if len(processed_audio.shape) > 1:
            sf.write(output_path, processed_audio.T, sr) # Trasponi per soundfile se multicanale
        else:
            sf.write(output_path, processed_audio, sr)

        return output_path, sr, original_audio_path_temp # Restituisce i percorsi dei file temporanei

    except Exception as e:
        st.error(f"Errore nel processing: {e}. Dettagli tecnici: {e.__class__.__name__}: {e}. Trace: {traceback.format_exc()}")
        return None, None, None # In caso di errore, restituisce None
    # Rimosso il blocco finally per l'eliminazione dei file temporanei QUI.
    # Verranno eliminati alla fine del blocco principale di Streamlit.


def decompose_audio(audio, sr, method, params):
    """Applica il metodo di decomposizione scelto"""

    if audio.size == 0:
        return np.array([])

    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio)) * 0.8
    else:
        return np.array([])

    methods_map = {
        'cut_up_sonoro': cut_up_sonoro,
        'remix_destrutturato': remix_destrutturato,
        'musique_concrete': musique_concrete,
        'decostruzione_postmoderna': decostruzione_postmoderna,
        'decomposizione_creativa': decomposizione_creativa,
        'random_chaos': random_chaos
    }

    decompose_func = methods_map.get(method)
    if decompose_func == None:
        st.error(f"Metodo {method} non riconosciuto")
        return audio

    try:
        result = decompose_func(audio, sr, params)

        if result is None or not isinstance(result, np.ndarray) or result.size == 0:
            st.warning(f"Il metodo '{method}' ha prodotto un risultato vuoto o non valido. Utilizzo l'audio originale come fallback. Trace: {traceback.format_exc()}")
            return audio # Fallback all'audio originale

        # Pulisci da eventuali NaN o Inf prima della normalizzazione finale
        result[np.isnan(result)] = 0
        result[np.isinf(result)] = 0

        if np.max(np.abs(result)) > 0:
            result = result / np.max(np.abs(result)) * 0.8
        else:
            return np.array([]) # Se l'audio è tutto zero, consideralo vuoto

        return result

    except Exception as e:
        st.error(f"Errore nella decomposizione del metodo '{method}': {e}. Dettagli tecnici: {e.__class__.__name__}: {e}. Trace: {traceback.format_exc()}")
        return audio

def create_visualization(original_audio, processed_audio, sr):
    """Crea visualizzazione comparativa degli spettrogrammi"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    if original_audio.size > 0:
        try:
            D_orig = librosa.amplitude_to_db(np.abs(librosa.stft(original_audio)), ref=np.max)
            librosa.display.specshow(D_orig, y_axis='hz', x_axis='time', sr=sr, ax=ax1)
        except Exception as e:
            st.warning(f"Could not create spectrogram for original audio: {e}. Trace: {traceback.format_exc()}")
    ax1.set_title(' Audio Originale')
    ax1.set_ylabel('Frequenza (Hz)')

    if processed_audio.size > 0:
        try:
            D_proc = librosa.amplitude_to_db(np.abs(librosa.stft(processed_audio)), ref=np.max)
            librosa.display.specshow(D_proc, y_axis='hz', x_axis='time', sr=sr, ax=ax2)
        except Exception as e:
            st.warning(f"Could not create spectrogram for processed audio: {e}. Trace: {traceback.format_exc()}")
    ax2.set_title(' Audio Decomposto/Ricomposto')
    ax2.set_xlabel('Tempo (s)')
    ax2.set_ylabel('Frequenza (Hz)')

    plt.tight_layout()
    return fig

# Interfaccia principale
if uploaded_file is not None:
    st.success(f" File caricato: {uploaded_file.name}")

    # Prepara i parametri comuni
    params = {
        'fragment_size': fragment_size,
        'chaos_level': chaos_level,
        'structure_preservation': structure_preservation
    }

    # Aggiungi i parametri specifici per metodo
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

    if st.button(" DECOMPONI E RICOMPONI", type="primary", use_container_width=True):
        with st.spinner(" Decomponendo il brano in arte sonora sperimentale..."):
            # Passa l'oggetto uploaded_file direttamente a process_audio
            output_path, sr, original_audio_path_temp = process_audio(uploaded_file, decomposition_method, params)

            # ✅ Controlli robusti
            if not output_path or not os.path.exists(output_path):
                st.error("❌ Errore: il file audio decomposto non è stato generato. Il metodo selezionato potrebbe aver restituito un array vuoto.")
                # Pulisci original_audio_path_temp se esiste ancora a questo punto
                if original_audio_path_temp and os.path.exists(original_audio_path_temp):
                    try:
                        os.unlink(original_audio_path_temp)
                    except Exception as e:
                        st.warning(f"⚠️ Impossibile eliminare il file temporaneo originale: {original_audio_path_temp} - {e}")
                st.stop() # FERMA L'ESECUZIONE QUI

            if not original_audio_path_temp or not os.path.exists(original_audio_path_temp):
                st.error("❌ Errore: file audio originale temporaneo non trovato.")
                # Pulisci output_path se esiste ancora a questo punto
                if output_path and os.path.exists(output_path):
                    try:
                        os.unlink(output_path)
                    except Exception as e:
                        st.warning(f"⚠️ Impossibile eliminare il file temporaneo di output: {output_path} - {e}")
                st.stop() # FERMA L'ESECUZIONE QUI

            try:
                original_audio, _ = librosa.load(original_audio_path_temp, sr=sr)
                processed_audio, _ = librosa.load(output_path, sr=sr)
            except Exception as e:
                st.error(f"❌ Errore durante il caricamento dei file audio per la visualizzazione/riproduzione: {e}. Trace: {traceback.format_exc()}")
                # Pulisci entrambi i file temporanei in caso di errore di caricamento
                if output_path and os.path.exists(output_path):
                    try: os.unlink(output_path)
                    except: pass
                if original_audio_path_temp and os.path.exists(original_audio_path_temp):
                    try: os.unlink(original_audio_path_temp)
                    except: pass
                st.stop() # FERMA L'ESECUZIONE QUI

            # UI
            col1, col2 = st.columns(2)

            with col1:
                st.subheader(" Audio Originale")
                # Carica l'audio originale direttamente da uploaded_file (che è un BytesIO)
                uploaded_file.seek(0) # Reset del puntatore prima della riproduzione
                st.audio(uploaded_file.read(), format=uploaded_file.type)
                
                duration_orig = len(original_audio) / sr if original_audio.size > 0 else 0
                st.info(f"""
                **Durata:** {duration_orig:.2f} secondi
                **Sample Rate:** {sr} Hz
                **Samples:** {len(original_audio):,}
                """)

            with col2:
                st.subheader(" Audio Decomposto")
                with open(output_path, 'rb') as f:
                    audio_bytes = f.read()
                st.audio(audio_bytes, format='audio/wav')
                
                duration_proc = len(processed_audio) / sr if processed_audio.size > 0 else 0
                transform_ratio = duration_proc/duration_orig if duration_orig > 0 else 0
                st.info(f"""
                **Durata:** {duration_proc:.2f} secondi
                **Trasformazione:** {transform_ratio:.2f}x
                **Samples:** {len(processed_audio):,}
                """)

            st.subheader(" Analisi Spettrale Comparativa")
            with st.spinner("Generando visualizzazioni..."):
                fig = create_visualization(original_audio, processed_audio, sr)
                st.pyplot(fig)

            st.subheader(" Download Risultato")
            # Assicurati che audio_bytes sia disponibile per il download
            # È già stato letto sopra, quindi riusalo.
            original_name = uploaded_file.name.rsplit('.', 1)[0]
            output_filename = f"{original_name}_{decomposition_method}.wav"

            st.download_button(
                label=f" Scarica {output_filename}",
                data=audio_bytes,
                file_name=output_filename,
                mime="audio/wav",
                use_container_width=True
            )

            with st.expander(" Analisi Tecnica Dettagliata"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Audio Originale:**")
                    if original_audio.size > 0:
                        orig_structure = analyze_audio_structure(original_audio, sr)
                        st.write(f"- Tempo stimato: {orig_structure['tempo']:.1f} BPM")
                        st.write(f"- Beat rilevati: {len(orig_structure['beats'])}")
                        st.write(f"- Onset rilevati: {len(orig_structure['onset_times'])}")
                        st.write(f"- Centroide spettrale medio: {np.mean(orig_structure['spectral_centroids']):.1f} Hz" if orig_structure['spectral_centroids'].size > 0 else "- Centroide spettrale medio: N/A")
                    else:
                        st.write("Nessun dato audio per l'analisi.")

                with col2:
                    st.write("**Audio Processato:**")
                    if processed_audio.size > 0:
                        proc_structure = analyze_audio_structure(processed_audio, sr)
                        st.write(f"- Tempo stimato: {proc_structure['tempo']:.1f} BPM")
                        st.write(f"- Beat rilevati: {len(proc_structure['beats'])}")
                        st.write(f"- Onset rilevati: {len(proc_structure['onset_times'])}")
                        st.write(f"- Centroide spettrale medio: {np.mean(proc_structure['spectral_centroids']):.1f} Hz" if proc_structure['spectral_centroids'].size > 0 else "- Centroide spettrale medio: N/A")
                    else:
                        st.write("Nessun dato audio per l'analisi.")

            # Pulisci i file temporanei dopo l'uso completo (ORA È NEL POSTO GIUSTO)
            try:
                if output_path and os.path.exists(output_path):
                    os.unlink(output_path)
                if original_audio_path_temp and os.path.exists(original_audio_path_temp):
                    os.unlink(original_audio_path_temp)
            except Exception as e:
                st.warning(f"⚠️ Impossibile eliminare file temporanei: {e}")

else:
    st.markdown("""
    <div style='text-align: center; padding: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px;
    margin: 20px 0;'>
        <h2 style='color: white; margin-bottom: 20px;'> Benvenuto in MusicDecomposer</h2>
        <p style='color: #e0e0e0; font-size: 1.1em; margin-bottom: 30px;'>
            Trasforma i tuoi brani in arte sonora sperimentale attraverso tecniche di decomposizione e ricomposizione ispirate ai grandi movimenti dell'avanguardia musicale.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## Metodi di Decomposizione Disponibili")

    methods_info = {
        " Cut-up Sonoro (Burroughs)": {
            "desc": "Ispirrato alla tecnica letteraria di William S. Burroughs, taglia l'audio in frammenti e li riassembla in ordine casuale o secondo pattern specifici.",
            "best_for": "Creare texture ritmiche inaspettate, rompere la narrativa musicale tradizionale"
        },
        " Remix Destrutturato": {
            "desc": "Mantiene elementi riconoscibili del brano originale ma li ricontestualizza attraverso pitch shifting, time stretching e riorganizzazione strutturale.",
            "best_for": "Remix creativi che mantengono l'identità del brano originale"
        },
        " Musique Concrète": {
            "desc": "Applica tecniche di sintesi granulare e manipolazioni concrete, scomponendo l'audio in micro-elementi (grani) e ricomponendoli.",
            "best_for": "Creare texture ambientali, soundscape astratti"
        },
        " Decostruzione Postmoderna": {
            "desc": "Identifica i momenti 'importanti' del brano e li trasforma ironicamente, creando false aspettative e anticlimax.",
            "best_for": "Commenti critici sull'opera originale, arte concettuale"
        },
        " Decomposizione Creativa": {
            "desc": "Analizza le caratteristiche emotive dell'audio e crea discontinuità attraverso cluster di mood e emotional shifting.",
            "best_for": "Esplorare nuove connessioni emotive, creare narrazioni sonore non-lineari"
        },
        " Random Chaos": {
            "desc": "Combina casualmente tutti i metodi precedenti per risultati completamente imprevedibili.",
            "best_for": "Sperimentazione estrema, scoperta di combinazioni inaspettate"
        }
    }

    for method, info in methods_info.items():
        with st.expander(method):
            st.write(f"**Descrizione:** {info['desc']}")
            st.write(f"**Ideale per:** {info['best_for']}")

    st.markdown("## Influenze e Ispirazione")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        ** Musique Concrète**
        - Pierre Schaeffer
        - Karlheinz Stockhausen
        - Jean-Claude Risset
        """)

    with col2:
        st.markdown("""
        ** Plunderphonics**
        - John Oswald
        - The Avalanches
        - Girl Talk
        """)

    with col3:
        st.markdown("""
        ** Cut-up Technique**
        - William S. Burroughs
        - Brion Gysin
        - David Bowie
        """)

    st.markdown("""
    ---
    <div style='text-align: center; color: #666; font-style: italic;'>
         MusicDecomposer by loop507 - Esplorando i confini tra ordine e caos sonoro
    </div>
    """, unsafe_allow_html=True)

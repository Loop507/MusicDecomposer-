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

    # Parametri generali
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
        st.subheader("Postmodern Parameters")
        irony_level = st.slider("Livello Ironia", 0.1, 2.0, 1.0, 0.1)
        context_shift = st.slider("Shift di Contesto", 0.1, 2.0, 1.2, 0.1)

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
    """Decostruzione ironica e postmoderna del brano - Versione robusta"""
    irony_level = params.get('irony_level', 1.0)
    context_shift = params.get('context_shift', 1.2)
    fragment_size = params['fragment_size']

    if audio.size == 0:
        return np.array([])

    fragment_samples = int(fragment_size * sr)
    if fragment_samples <= 0:
        return audio

    try:
        # Calcolo energia con controlli di sicurezza
        hop_length = 512
        energy = np.array([])
        
        if len(audio) >= hop_length:
            try:
                energy = librosa.feature.rms(y=audio, hop_length=hop_length)[0]
            except Exception as e:
                st.warning(f"RMS calculation failed: {e}")
                # Fallback: usa energia semplice
                num_frames = len(audio) // hop_length
                if num_frames > 0:
                    energy = np.array([np.sqrt(np.mean(audio[i*hop_length:(i+1)*hop_length]**2)) 
                                       for i in range(num_frames)])

        # Identifica frame importanti
        important_frames = np.array([])
        if energy.size > 0:
            energy_threshold = np.percentile(energy, 70)
            important_frames = np.where(energy > energy_threshold)[0]

        # Converti frame in tempi
        important_times = np.array([])
        if important_frames.size > 0:
            important_times = librosa.frames_to_time(important_frames, sr=sr, hop_length=hop_length)

        fragments = []
        fragment_types = []

        # Estrai frammenti importanti
        for t in important_times:
            start_sample = int(t * sr)
            end_sample = min(start_sample + fragment_samples, len(audio))

            if end_sample > start_sample:
                fragment = audio[start_sample:end_sample]
                if fragment.size > 0:
                    fragments.append(fragment)
                    fragment_types.append('important')

        # Aggiungi frammenti casuali
        num_random = max(1, int(len(fragments) * 0.5))  # Almeno 1 frammento
        for _ in range(num_random):
            if len(audio) < fragment_samples:
                # Se l'audio è più corto del frammento, prendi tutto
                fragment = audio.copy()
            else:
                start = random.randint(0, len(audio) - fragment_samples)
                fragment = audio[start:start + fragment_samples]
            
            if fragment.size > 0:
                fragments.append(fragment)
                fragment_types.append('random')

        if len(fragments) == 0:
            return audio  # Fallback all'originale se non ci sono frammenti

        # Processa frammenti con transformazioni ironiche
        processed_fragments = []
        for i, (fragment, frag_type) in enumerate(zip(fragments, fragment_types)):
            if fragment.size == 0:
                continue

            # Transformazioni ironiche per frammenti importanti
            if frag_type == 'important' and random.random() < irony_level / 2.0:
                ironic_transforms = [
                    # Inversione temporale
                    lambda x: x[::-1] if x.size > 0 else np.array([]),
                    # Pitch shift verso il basso (meno aggressivo)
                    lambda x: safe_pitch_shift(x, sr, -6) if x.size > 0 else np.array([]),
                    # Time stretch più conservativo
                    lambda x: safe_time_stretch(x, 0.5) if x.size > 0 else np.array([]),
                    # Riduzione volume
                    lambda x: x * 0.2 if x.size > 0 else np.array([]),
                    # Ripetizione controllata (max 3 volte per evitare OOM)
                    lambda x: np.tile(x[:min(len(x)//3, 2000)] if len(x) > 0 else np.array([]), 3) if len(x) > 0 else np.array([]),
                ]
                
                transform = random.choice(ironic_transforms)
                try:
                    transformed_fragment = transform(fragment)
                    if transformed_fragment.size > 0:
                        fragment = transformed_fragment
                    else:
                        st.warning(f"Ironic transform produced empty fragment. Keeping original.")
                except Exception as e:
                    st.warning(f"Ironic transform failed: {e}. Keeping original.")

            # Context shift effects
            if fragment.size > 0 and random.random() < context_shift / 2.0:
                context_effects = [
                    # Pitch shift più conservativo
                    lambda x: safe_pitch_shift(x, sr, random.uniform(-3, 3)) if x.size > 0 else np.array([]),
                    # Fade in
                    lambda x: x * np.linspace(0, 1, len(x)) if len(x) > 0 else np.array([]),
                    # Fade out
                    lambda x: x * np.linspace(1, 0, len(x)) if len(x) > 0 else np.array([]),
                    # Rumore controllato
                    lambda x: x + np.random.normal(0, 0.02, len(x)) if len(x) > 0 else np.array([]),
                ]
                
                effect = random.choice(context_effects)
                try:
                    effected_fragment = effect(fragment)
                    if effected_fragment.size > 0:
                        fragment = effected_fragment
                    else:
                        st.warning(f"Context shift effect produced empty fragment. Keeping original.")
                except Exception as e:
                    st.warning(f"Context shift effect failed: {e}. Keeping original.")

            if fragment.size > 0:
                processed_fragments.append(fragment)

        if len(processed_fragments) == 0:
            return audio  # Fallback all'originale

        # Calcola energie dei frammenti processati
        fragment_energies = []
        processed_fragments_filtered = []
        
        for frag in processed_fragments:
            if frag.size > 0:
                energy_val = np.mean(np.abs(frag))
                if not np.isnan(energy_val) and not np.isinf(energy_val):
                    fragment_energies.append(energy_val)
                    processed_fragments_filtered.append(frag)

        if len(processed_fragments_filtered) == 0:
            return audio  # Fallback all'originale

        # Ordina per energia
        if len(fragment_energies) > 0:
            sorted_indices = np.argsort(fragment_energies)
            
            # Dividi in alta e bassa energia
            mid_point = len(sorted_indices) // 2
            low_energy = sorted_indices[:mid_point]
            high_energy = sorted_indices[mid_point:]
            
            # Crea ordine alternato
            result_order_indices = []
            max_pairs = min(len(low_energy), len(high_energy))
            
            for i in range(max_pairs):
                if random.random() < 0.6:  # Preferenza per alta energia
                    result_order_indices.append(high_energy[i])
                result_order_indices.append(low_energy[i])
            
            # Aggiungi rimanenti
            if len(high_energy) > max_pairs:
                result_order_indices.extend(high_energy[max_pairs:])
            if len(low_energy) > max_pairs:
                result_order_indices.extend(low_energy[max_pairs:])
            
            random.shuffle(result_order_indices) # Aggiunge un tocco di casualità all'ordine finale
            
            final_fragments = [processed_fragments_filtered[i] for i in result_order_indices]
        else:
            final_fragments = processed_fragments_filtered # Nessuna energia per ordinare, usa filtrati


        # Riassembla con crossfades per fluidità postmoderna
        if len(final_fragments) == 0:
            return audio # Fallback all'originale

        result_audio = np.array([])
        fade_duration = int(sr * 0.05) # 50 ms fade

        for i, frag in enumerate(final_fragments):
            if frag.size == 0:
                continue

            if result_audio.size == 0:
                result_audio = frag
            else:
                overlap_samples = min(fade_duration, result_audio.size, frag.size)
                if overlap_samples > 0:
                    crossfade_out = np.linspace(1, 0, overlap_samples)
                    crossfade_in = np.linspace(0, 1, overlap_samples)
                    
                    overlap_section_result = result_audio[-overlap_samples:]
                    overlap_section_frag = frag[:overlap_samples]
                    
                    crossfaded_section = overlap_section_result * crossfade_out + overlap_section_frag * crossfade_in
                    
                    result_audio = np.concatenate([result_audio[:-overlap_samples], crossfaded_section, frag[overlap_samples:]])
                else:
                    result_audio = np.concatenate([result_audio, frag])
        
        # Normalizza il risultato finale
        if result_audio.size > 0:
            result_audio = librosa.util.normalize(result_audio)
        else:
            return np.array([]) # Se il risultato è vuoto, restituisci un array vuoto
        
        return result_audio

    except Exception as e:
        st.error(f"Errore durante la decostruzione postmoderna: {e}. Trace: {traceback.format_exc()}")
        return audio # Fallback all'originale in caso di errore grave


def decomposizione_creativa(audio, sr, params):
    """Scompone e ricompone in modo 'creativo' usando clustering e manipolazione dinamica."""
    fragment_size = params['fragment_size']
    chaos_level = params['chaos_level']
    discontinuity = params.get('discontinuity', 1.0)
    emotional_shift = params.get('emotional_shift', 0.8)

    if audio.size == 0:
        return np.array([])

    fragment_samples = int(fragment_size * sr)
    if fragment_samples <= 0:
        return audio

    fragments = []
    # Assicurati che il loop sia valido
    for i in range(0, len(audio) - fragment_samples + 1, fragment_samples):
        fragment = audio[i:i + fragment_samples]
        if fragment.size > 0:
            fragments.append(fragment)

    if len(fragments) == 0:
        return np.array([])

    # Estrai feature per clustering (es. MFCC)
    feature_vectors = []
    original_indices = []
    for i, frag in enumerate(fragments):
        if frag.size > 0:
            try:
                # Pad o tronca il frammento per avere una dimensione fissa per MFCC
                # Questo è cruciale per K-Means che richiede input di dimensione costante
                mfcc_length = int(sr * 0.1) # Una lunghezza fissa, es. 0.1 secondi di audio
                if len(frag) < mfcc_length:
                    padded_frag = np.pad(frag, (0, mfcc_length - len(frag)))
                else:
                    padded_frag = frag[:mfcc_length]
                    
                mfcc = librosa.feature.mfcc(y=padded_frag, sr=sr, n_mfcc=13)
                if mfcc.size > 0:
                    feature_vectors.append(np.mean(mfcc, axis=1)) # Media delle MFCC per il frammento
                    original_indices.append(i)
            except Exception as e:
                st.warning(f"Errore estrazione MFCC per frammento {i}: {e}. Salto il frammento.")
                # Continua con il prossimo frammento, non aggiungere questo ai feature_vectors

    if len(feature_vectors) == 0:
        return np.array([]) # Se nessuna feature valida, non possiamo procedere

    # Normalizzazione feature
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_vectors)

    # Clustering K-Means
    num_clusters = min(len(scaled_features), max(2, int(len(fragments) * structure_preservation * 0.5))) # Almeno 2 cluster
    if num_clusters < 2:
        return np.concatenate(fragments) # Se non abbastanza cluster, riassemblo in ordine
    
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init='auto')
    try:
        clusters = kmeans.fit_predict(scaled_features)
    except Exception as e:
        st.warning(f"Errore nel clustering KMeans: {e}. Riassemblaggio sequenziale.")
        return np.concatenate(fragments) # Fallback

    clustered_fragments = [[] for _ in range(num_clusters)]
    for i, cluster_idx in enumerate(clusters):
        # Usa original_indices per mappare correttamente al frammento originale
        clustered_fragments[cluster_idx].append(fragments[original_indices[i]])

    # Riassemblaggio e manipolazione
    recomposed_audio = []
    for cluster_idx in range(num_clusters):
        cluster_frags = clustered_fragments[cluster_idx]
        if len(cluster_frags) == 0:
            continue

        # Applica manipolazioni al cluster
        if random.random() < discontinuity: # Maggiore discontinuità = più manipolazioni
            if random.random() < emotional_shift: # Pitch shift o time stretch
                manipulated_frags = []
                for frag in cluster_frags:
                    if frag.size > 0:
                        if random.random() < 0.5:
                            shift = random.uniform(-6, 6) # Pitch shift emotivo
                            frag = safe_pitch_shift(frag, sr, shift)
                        else:
                            stretch_rate = random.uniform(0.7, 1.3) # Time stretch emotivo
                            frag = safe_time_stretch(frag, stretch_rate)
                        if frag.size > 0:
                            manipulated_frags.append(frag)
                cluster_frags = manipulated_frags

            if random.random() < chaos_level / 2.0: # Inversione o randomizzazione
                if random.random() < 0.5:
                    cluster_frags = [f[::-1] for f in cluster_frags if f.size > 0] # Inverti frammenti
                random.shuffle(cluster_frags) # Randomizza ordine all'interno del cluster

        # Concatena i frammenti del cluster (con crossfade)
        if len(cluster_frags) > 0:
            current_cluster_audio = np.array([])
            fade_duration_samples = int(sr * 0.02) # Breve crossfade

            for frag in cluster_frags:
                if frag.size == 0:
                    continue
                if current_cluster_audio.size == 0:
                    current_cluster_audio = frag
                else:
                    overlap = min(fade_duration_samples, current_cluster_audio.size, frag.size)
                    if overlap > 0:
                        crossfade_out = np.linspace(1, 0, overlap)
                        crossfade_in = np.linspace(0, 1, overlap)
                        
                        current_cluster_audio = np.concatenate([
                            current_cluster_audio[:-overlap],
                            (current_cluster_audio[-overlap:] * crossfade_out + frag[:overlap] * crossfade_in),
                            frag[overlap:]
                        ])
                    else:
                        current_cluster_audio = np.concatenate([current_cluster_audio, frag])
            if current_cluster_audio.size > 0:
                recomposed_audio.append(current_cluster_audio)

    if len(recomposed_audio) == 0:
        return np.array([])

    # Riassembla i cluster (randomizzando l'ordine dei cluster per "discontinuità")
    random.shuffle(recomposed_audio)
    final_audio = np.concatenate(recomposed_audio)

    # Normalizza il risultato finale
    if final_audio.size > 0:
        final_audio = librosa.util.normalize(final_audio)
    else:
        return np.array([])

    return final_audio

def random_chaos(audio, sr, params):
    """Applica manipolazioni estreme e casuali per un output imprevedibile."""
    chaos_level = params['chaos_level']
    fragment_size = params['fragment_size']

    if audio.size == 0:
        return np.array([])

    fragment_samples = int(fragment_size * sr)
    if fragment_samples <= 0:
        return audio

    fragments = []
    # Suddividi l'audio in frammenti
    for i in range(0, len(audio) - fragment_samples + 1, fragment_samples):
        fragment = audio[i:i + fragment_samples]
        if fragment.size > 0:
            fragments.append(fragment)

    if len(fragments) == 0:
        return np.array([])

    processed_fragments = []
    for frag in fragments:
        if frag.size == 0:
            continue

        # Applica manipolazioni basate sul livello di chaos
        if random.random() < chaos_level * 0.3: # Inversione casuale
            frag = frag[::-1]

        if random.random() < chaos_level * 0.4: # Pitch shift estremo
            shift = random.uniform(-24, 24) # Due ottave su o giù
            frag = safe_pitch_shift(frag, sr, shift)

        if frag.size > 0 and random.random() < chaos_level * 0.4: # Time stretch/compress estremo
            stretch_rate = random.uniform(0.1, 10.0) # Molto lento o molto veloce
            frag = safe_time_stretch(frag, stretch_rate)
        
        if frag.size > 0 and random.random() < chaos_level * 0.2: # Aggiungi rumore
            noise = np.random.normal(0, np.max(np.abs(frag)) * random.uniform(0.05, 0.2), len(frag))
            frag = frag + noise

        if frag.size > 0 and random.random() < chaos_level * 0.1: # Distorsione (semplice clipping)
            clip_threshold = random.uniform(0.3, 0.8)
            frag = np.clip(frag, -clip_threshold, clip_threshold)

        if frag.size > 0:
            processed_fragments.append(frag)

    if len(processed_fragments) == 0:
        return np.array([])

    random.shuffle(processed_fragments) # Randomizza l'ordine dei frammenti

    # Riassemblaggio con crossfade
    final_audio = np.array([])
    fade_duration = int(sr * 0.05) # 50 ms crossfade

    for i, frag in enumerate(processed_fragments):
        if frag.size == 0:
            continue
        if final_audio.size == 0:
            final_audio = frag
        else:
            overlap = min(fade_duration, final_audio.size, frag.size)
            if overlap > 0:
                crossfade_out = np.linspace(1, 0, overlap)
                crossfade_in = np.linspace(0, 1, overlap)
                
                final_audio = np.concatenate([
                    final_audio[:-overlap],
                    (final_audio[-overlap:] * crossfade_out + frag[:overlap] * crossfade_in),
                    frag[overlap:]
                ])
            else:
                final_audio = np.concatenate([final_audio, frag])

    # Normalizza il risultato finale
    if final_audio.size > 0:
        final_audio = librosa.util.normalize(final_audio)
    else:
        return np.array([])

    return final_audio


def generate_analysis_text(original_structure, decomposed_audio, sr, method, params):
    """Genera un'analisi testuale dettagliata del processo di decomposizione."""
    analysis = f"## Analisi della Decomposizione\n\n"
    analysis += f"**Metodo Selezionato:** {method.replace('_', ' ').title()}\n"
    analysis += f"**Durata Originale:** {len(original_structure['audio']) / sr:.2f} secondi\n"
    analysis += f"**Durata Decomposta:** {len(decomposed_audio) / sr:.2f} secondi\n\n"

    analysis += "### Parametri Utilizzati:\n"
    for key, value in params.items():
        analysis += f"- **{key.replace('_', ' ').title()}:** {value}\n"
    analysis += "\n"

    analysis += "### Dettagli Strutturali Originali:\n"
    if original_structure['tempo'] > 0:
        analysis += f"- **Tempo (BPM):** {original_structure['tempo']:.2f}\n"
    if original_structure['beats'].size > 0:
        analysis += f"- **Battiti Rilevati:** {len(original_structure['beats'])}\n"
    if original_structure['onset_times'].size > 0:
        analysis += f"- **Eventi di Attacco Rilevati:** {len(original_structure['onset_times'])}\n"
    if original_structure['chroma'].size > 0:
        analysis += f"- **Armonia (Chroma Features):** Presente e analizzata.\n"
    if original_structure['mfcc'].size > 0:
        analysis += f"- **Timbro (MFCC):** Presente e analizzato.\n"
    if original_structure['spectral_centroids'].size > 0:
        analysis += f"- **Brillantezza Spettrale (Spectral Centroid):** Presente e analizzata.\n"
    analysis += "\n"

    analysis += "### Impatto del Metodo di Decomposizione:\n"

    if method == "cut_up_sonoro":
        analysis += (
            f"Il metodo 'Cut-up Sonoro' ha segmentato il brano in frammenti di circa {params['fragment_size']:.1f} secondi. "
            f"Con una casualità dei tagli del {params.get('cut_randomness', 0.7)*100:.0f}%, molti frammenti sono stati manipolati (es. accorciati, allungati, invertiti). "
            f"Lo stile di riassemblaggio '{params.get('reassembly_style', 'random').replace('_', ' ').title()}' ha determinato l'ordine finale, creando una narrazione sonora frammentata e disorientante, tipica della tecnica di William S. Burroughs."
        )
    elif method == "remix_destrutturato":
        analysis += (
            f"Il 'Remix Destrutturato' ha cercato di mantenere un legame con la struttura originale del brano. "
            f"Con una conservazione del ritmo del {params.get('beat_preservation', 0.4)*100:.0f}%, alcuni punti di battito sono stati rispettati, "
            f"mentre la frammentazione della melodia ({params.get('melody_fragmentation', 1.5):.1f}x) ha introdotto alterazioni di pitch e tempo. "
            f"Questo ha generato un'esperienza familiare ma allo stesso tempo alienata, dove elementi riconoscibili emergono in un contesto nuovo e inaspettato."
        )
    elif method == "musique_concrete":
        analysis += (
            f"La tecnica 'Musique Concrète' ha trasformato l'audio in grani minuscoli (circa {params.get('grain_size', 0.1):.2f} secondi). "
            f"La densità della texture ({params.get('texture_density', 1.0):.1f}x) ha influenzato il numero di grani sovrapposti. "
            f"Il livello di chaos ({params['chaos_level']:.1f}x) ha introdotto manipolazioni estreme come inversioni, pitch shift e time stretch, "
            f"creando un paesaggio sonoro astratto, focalizzato sulle qualità intrinseche del suono piuttosto che sulla sua origine musicale."
        )
    elif method == "decostruzione_postmoderna":
        analysis += (
            f"La 'Decostruzione Postmoderna' ha analizzato l'energia del brano per identificare frammenti 'importanti' e li ha posti in dialogo con frammenti casuali. "
            f"Il livello di ironia ({params.get('irony_level', 1.0):.1f}x) ha portato a manipolazioni come inversioni, pitch shift bassi o riduzioni di volume sui momenti chiave, "
            f"mentre lo 'shift di contesto' ({params.get('context_shift', 1.2):.1f}x) ha applicato dissolvenze, rumori o leggere alterazioni di pitch. "
            f"Il risultato è una rilettura critica e spesso surreale dell'originale, che ne svela le convenzioni e ne esplora nuove interpretazioni."
        )
    elif method == "decomposizione_creativa":
        analysis += (
            f"La 'Decomposizione Creativa' ha utilizzato tecniche di apprendimento automatico (K-Means Clustering) per raggruppare frammenti con caratteristiche timbriche simili. "
            f"Successivamente, ha applicato manipolazioni dinamiche e casuali a questi cluster, influenzate dai parametri di discontinuità ({params.get('discontinuity', 1.0):.1f}x) "
            f"e shift emotivo ({params.get('emotional_shift', 0.8):.1f}x). "
            f"Questo ha permesso di esplorare nuove relazioni tra le parti del brano, generando un'opera che bilancia coesione tematica e frammentazione sonora."
        )
    elif method == "random_chaos":
        analysis += (
            f"Il metodo 'Random Chaos' ha applicato manipolazioni estreme e imprevedibili a ogni frammento del brano, "
            f"guidato principalmente dal livello di chaos ({params['chaos_level']:.1f}x). "
            f"Ogni frammento ha subito trasformazioni radicali come inversione, pitch shift e time stretch estremi, aggiunta di rumore e distorsione. "
            f"Il riassemblaggio casuale ha eliminato ogni traccia della struttura originale, risultando in un'esperienza sonora densa, imprevedibile e completamente astratta."
        )
    else:
        analysis += "Nessun impatto specifico descritto per il metodo selezionato."

    analysis += "\n\n**Nota:** Le durate possono variare leggermente a causa delle manipolazioni di time-stretching e riassemblaggio che modificano la lunghezza complessiva dell'audio."

    return analysis

# Placeholder per i risultati
processed_audio_data = None
original_sr = None
original_audio_info = None

if uploaded_file is not None:
    st.audio(uploaded_file, format=uploaded_file.type, start_time=0)

    # Pulsante per avviare la decomposizione
    if st.button("DECOMPONI E RICOMPONI", type="primary"):
        with st.spinner("Decomposizione in corso... potrebbe richiedere del tempo per brani lunghi."):
            try:
                # Carica audio usando tempfile per gestire i vari formati
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name

                audio, sr = librosa.load(tmp_file_path, sr=None, mono=True) # sr=None per mantenere sample rate originale
                os.remove(tmp_file_path) # Pulisci il file temporaneo

                if audio.size == 0:
                    st.error("Il file audio caricato è vuoto o non valido.")
                else:
                    # Raccogli i parametri specifici del metodo
                    params = {
                        'fragment_size': fragment_size,
                        'chaos_level': chaos_level,
                        'structure_preservation': structure_preservation,
                    }
                    if decomposition_method == "cut_up_sonoro":
                        params['cut_randomness'] = cut_randomness
                        params['reassembly_style'] = reassembly_style
                    elif decomposition_method == "remix_destrutturato":
                        params['beat_preservation'] = beat_preservation
                        params['melody_fragmentation'] = melody_fragmentation
                    elif decomposition_method == "musique_concrete":
                        params['grain_size'] = grain_size
                        params['texture_density'] = texture_density
                    elif decomposition_method == "decostruzione_postmoderna":
                        params['irony_level'] = irony_level
                        params['context_shift'] = context_shift
                    elif decomposition_method == "decomposizione_creativa":
                        params['discontinuity'] = discontinuity
                        params['emotional_shift'] = emotional_shift

                    # Analizza la struttura originale (anche per l'analisi testuale)
                    original_audio_info = analyze_audio_structure(audio, sr)
                    original_audio_info['audio'] = audio # Salva l'audio originale per il calcolo della durata

                    # Esegui la decomposizione
                    if decomposition_method == "cut_up_sonoro":
                        processed_audio_data = cut_up_sonoro(audio, sr, params)
                    elif decomposition_method == "remix_destrutturato":
                        processed_audio_data = remix_destrutturato(audio, sr, params)
                    elif decomposition_method == "musique_concrete":
                        processed_audio_data = musique_concrete(audio, sr, params)
                    elif decomposition_method == "decostruzione_postmoderna":
                        processed_audio_data = decostruzione_postmoderna(audio, sr, params)
                    elif decomposition_method == "decomposizione_creativa":
                        processed_audio_data = decomposizione_creativa(audio, sr, params)
                    elif decomposition_method == "random_chaos":
                        processed_audio_data = random_chaos(audio, sr, params)
                    else:
                        st.error("Metodo di decomposizione non riconosciuto.")
                        processed_audio_data = np.array([]) # Imposta a vuoto per evitare errori successivi
                    
                    original_sr = sr # Salva sample rate per uso futuro

                    if processed_audio_data.size == 0:
                        st.warning("La decomposizione ha prodotto un audio vuoto. Si prega di provare con parametri diversi o un altro file.")

            except Exception as e:
                st.error(f"Si è verificato un errore durante l'elaborazione: {e}")
                st.exception(e) # Mostra il traceback completo
                processed_audio_data = None # Resetta per non mostrare output invalidi

# Se l'audio processato è disponibile, mostralo
if processed_audio_data is not None and processed_audio_data.size > 0 and original_sr is not None:
    st.subheader("Brano Decomposto e Ricomposto")
    
    # Normalizza l'audio per una riproduzione sicura
    processed_audio_data = librosa.util.normalize(processed_audio_data)

    # Salva l'audio processato in un buffer per la riproduzione e il download
    audio_buffer = io.BytesIO()
    try:
        sf.write(audio_buffer, processed_audio_data, original_sr, format='WAV')
        audio_buffer.seek(0)
        st.audio(audio_buffer, format='audio/wav')
    except Exception as e:
        st.error(f"Errore durante la creazione del file audio per la riproduzione: {e}")
        st.exception(e)

    # Grafico della waveform
    st.subheader("Visualizzazione Waveform")
    fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(processed_audio_data, sr=original_sr, ax=ax)
    ax.set_title("Waveform del brano decomposto")
    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("Ampiezza")
    st.pyplot(fig)
    plt.close(fig) # Chiudi la figura per liberare memoria

    # Pulsante per il download
    st.download_button(
        label="Scarica Brano Decomposto",
        data=audio_buffer.getvalue(),
        file_name=f"decomposed_audio_{decomposition_method}.wav",
        mime="audio/wav"
    )

    # Analisi Scritta
    st.subheader("Analisi Scritta del Processo")
    if original_audio_info is not None:
        current_params = {
            'fragment_size': fragment_size,
            'chaos_level': chaos_level,
            'structure_preservation': structure_preservation,
        }
        if decomposition_method == "cut_up_sonoro":
            current_params['cut_randomness'] = cut_randomness
            current_params['reassembly_style'] = reassembly_style
        elif decomposition_method == "remix_destrutturato":
            current_params['beat_preservation'] = beat_preservation
            current_params['melody_fragmentation'] = melody_fragmentation
        elif decomposition_method == "musique_concrete":
            current_params['grain_size'] = grain_size
            current_params['texture_density'] = texture_density
        elif decomposition_method == "decostruzione_postmoderna":
            current_params['irony_level'] = irony_level
            current_params['context_shift'] = context_shift
        elif decomposition_method == "decomposizione_creativa":
            current_params['discontinuity'] = discontinuity
            current_params['emotional_shift'] = emotional_shift

        analysis_text = generate_analysis_text(original_audio_info, processed_audio_data, original_sr, decomposition_method, current_params)
        st.markdown(analysis_text)
    else:
        st.info("Carica un brano e decomponilo per visualizzare l'analisi.")

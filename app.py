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
import base64

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
    # Aggiungi un controllo per audio vuoto
    if audio.size == 0:
        return {
            'tempo': 0, 'beats': np.array([]), 'chroma': np.array([]),
            'mfcc': np.array([]), 'spectral_centroids': np.array([]),
            'onset_times': np.array([]), 'onset_frames': np.array([])
        }

    # Estrazione features
    # Aggiungi un blocco try-except per la robustezza
    try:
        tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)
    except Exception as e:
        st.warning(f"Warning: Could not track beats, {e}. Setting tempo to 0.")
        tempo = 0
        beats = np.array([])

    # Analisi armonica
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)

    # Analisi timbrica
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)

    # Analisi spettrale
    spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]

    # Segmentazione strutturale
    onset_frames = librosa.onset.onset_detect(y=audio, sr=sr)
    onset_times = librosa.times_like(onset_frames, sr=sr)

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

    # Calcola dimensioni frammenti in samples
    fragment_samples = int(fragment_size * sr)
    total_samples = len(audio)

    if fragment_samples <= 0: # Ensure fragment size is positive
        return audio

    # Crea frammenti
    fragments = []
    for i in range(0, total_samples - fragment_samples + 1, fragment_samples): # +1 per includere l'ultimo frammento
        fragment = audio[i:i + fragment_samples]

        if fragment.size == 0: # Salta frammenti vuoti
            continue

        # Aggiungi variazioni casuali alla dimensione
        if random.random() < randomness:
            variation = random.uniform(0.5, 1.5)
            new_size = int(fragment.size * variation) # Usa fragment.size invece di fragment_samples per la variazione

            if new_size <= 0: # Evita frammenti di dimensione zero o negativa
                continue

            if new_size < fragment.size:
                fragment = fragment[:new_size]
            else:
                # Stretch del frammento
                # Evita interp se il frammento è troppo piccolo per essere stirato
                if fragment.size > 1:
                    indices = np.linspace(0, fragment.size - 1, new_size)
                    fragment = np.interp(indices, np.arange(fragment.size), fragment)
                else: # Se il frammento è 1 solo sample, duplicalo per raggiungere new_size
                    fragment = np.tile(fragment, new_size)

        if fragment.size > 0:
            fragments.append(fragment)

    # Se non ci sono frammenti validi, restituisci un array vuoto
    if not fragments:
        return np.array([])

    # Riassembla secondo lo stile scelto
    if reassembly == 'random':
        random.shuffle(fragments)
    elif reassembly == 'reverse':
        fragments = fragments[::-1]
        fragments = [frag[::-1] for frag in fragments if frag.size > 0]  # Reverse anche i singoli frammenti
    elif reassembly == 'palindrome':
        fragments = fragments + [frag for frag in fragments[::-1] if frag.size > 0]
    elif reassembly == 'spiral':
        # Prende alternativamente dall'inizio e dalla fine
        new_fragments = []
        start, end = 0, len(fragments) - 1
        while start <= end:
            if len(new_fragments) % 2 == 0:
                if fragments[start].size > 0:
                    new_fragments.append(fragments[start])
                start += 1
            else:
                if fragments[end].size > 0:
                    new_fragments.append(fragments[end])
                end -= 1
        fragments = new_fragments

    # Ricompone l'audio
    if fragments:
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

    # Analizza struttura originale
    structure = analyze_audio_structure(audio, sr)
    beats = structure['beats']

    fragment_samples = int(fragment_size * sr)
    if fragment_samples <= 0:
        return audio

    cut_points = []
    # Se conserviamo il ritmo, usiamo i beat come punti di taglio
    if beat_preservation > 0.5 and beats.size > 0:
        beat_times = librosa.frames_to_time(beats, sr=sr)
        cut_points = [int(t * sr) for t in beat_times if t * sr < len(audio)] # Assicura che i cut_points siano entro i limiti
    
    # Se i beat non sono sufficienti o non vengono usati, genera tagli casuali
    if not cut_points or (beat_preservation <= 0.5):
        num_cuts = int(len(audio) / fragment_samples)
        if num_cuts == 0 and len(audio) > 0: # Assicura almeno 1 taglio se l'audio non è vuoto
            num_cuts = 1
        if len(audio) > 0 and num_cuts > 0:
            cut_points = sorted(random.sample(range(0, len(audio)), num_cuts))
        else:
            return np.array([]) # Nessun taglio possibile, audio vuoto o troppo corto
    
    # Assicurati che i cut_points siano unici e ordinati
    cut_points = sorted(list(set(cut_points)))
    if not cut_points:
        return np.array([])
    
    # Aggiungi l'inizio e la fine dell'audio se non presenti
    if 0 not in cut_points:
        cut_points.insert(0, 0)
    if len(audio) not in cut_points:
        cut_points.append(len(audio))
    cut_points = sorted(list(set(cut_points)))


    # Crea frammenti tra i punti di taglio
    fragments = []
    for i in range(len(cut_points) - 1):
        start = cut_points[i]
        end = cut_points[i + 1]
        if end <= start: # Salta frammenti di lunghezza zero o negativa
            continue
        fragment = audio[start:end]

        if fragment.size == 0: # Salta frammenti vuoti
            continue

        # Applica frammentazione melodica
        if random.random() < melody_fragmentation / 3.0:
            # Pitch shift casuale
            try:
                shift_steps = random.uniform(-7, 7)  # Semitoni
                fragment = librosa.effects.pitch_shift(fragment, sr=sr, n_steps=shift_steps)
            except Exception as e:
                st.warning(f"Pitch shift failed for fragment: {e}")

        if random.random() < melody_fragmentation / 3.0:
            # Time stretch
            try:
                stretch_factor = random.uniform(0.7, 1.4)
                # Ensure fragment is not too short for time_stretch
                if fragment.size > 0:
                    fragment = librosa.effects.time_stretch(fragment, rate=stretch_factor)
            except Exception as e:
                st.warning(f"Time stretch failed for fragment: {e}")

        if fragment.size > 0:
            fragments.append(fragment)

    if not fragments:
        return np.array([])

    # Riorganizza mantenendo parzialmente la struttura originale
    if beat_preservation > 0.3:
        # Mantieni alcuni frammenti nella posizione originale
        preserve_count = int(len(fragments) * beat_preservation)
        preserve_indices = random.sample(range(len(fragments)), preserve_count)

        new_order = list(range(len(fragments)))
        remaining = [i for i in range(len(fragments)) if i not in preserve_indices]
        random.shuffle(remaining)

        # Assicurati che new_order abbia la dimensione corretta per i frammenti riordinati
        ordered_fragments_temp = [None] * len(fragments)
        preserved_map = {idx: frag for idx, frag in enumerate(fragments) if idx in preserve_indices}

        remaining_fragments_iter = iter([fragments[i] for i in remaining])

        for i in range(len(fragments)):
            if i in preserve_indices:
                ordered_fragments_temp[i] = preserved_map[i]
            else:
                try:
                    ordered_fragments_temp[i] = next(remaining_fragments_iter)
                except StopIteration:
                    # Should not happen if logic is correct, but for safety
                    ordered_fragments_temp[i] = np.array([]) # Fallback

        fragments = [f for f in ordered_fragments_temp if f is not None and f.size > 0] # Filtra frammenti null o vuoti

    else:
        random.shuffle(fragments)

    if not fragments:
        return np.array([])


    # Concatena con crossfade per fluidità
    result = fragments[0]
    fade_samples = int(0.05 * sr) # 50ms crossfade

    for fragment in fragments[1:]:
        if result.size == 0: # Se il risultato accumulato è vuoto, inizia con il nuovo frammento
            result = fragment
            continue

        if fragment.size == 0: # Salta frammenti vuoti
            continue

        # Assicurati che i frammenti siano sufficientemente lunghi per il crossfade
        current_fade_samples = min(fade_samples, result.size, fragment.size)

        if current_fade_samples > 0:
            # Crossfade
            fade_out = np.linspace(1, 0, current_fade_samples)
            fade_in = np.linspace(0, 1, current_fade_samples)

            # Prendi la parte finale di result e iniziale di fragment per il crossfade
            overlap_result = result[-current_fade_samples:]
            overlap_fragment = fragment[:current_fade_samples]

            # Applica fade e somma
            overlapped_section = overlap_result * fade_out + overlap_fragment * fade_in

            # Ricostruisci il risultato
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

    # Crea grani dall'audio originale
    grains = []
    # Assicurati che il loop possa produrre almeno un grano valido
    for i in range(0, len(audio) - grain_samples + 1, grain_samples // 2 if grain_samples // 2 > 0 else 1):
        grain = audio[i:i + grain_samples]

        if grain.size == 0:
            continue

        # Applica envelope gaussiano a ogni grano
        try:
            window = signal.windows.gaussian(len(grain), std=len(grain)/6)
            grain = grain * window
        except Exception as e:
            st.warning(f"Gaussian window failed for grain: {e}")
            grain = np.array([]) # Marca il grano come vuoto se fallisce

        if grain.size == 0:
            continue

        # Variazioni casuali per ogni grano
        if random.random() < chaos_level / 3.0:
            # Reverse del grano
            grain = grain[::-1]

        if random.random() < chaos_level / 3.0:
            # Pitch shift estremo
            try:
                if grain.size > 0:
                    shift = random.uniform(-12, 12)
                    grain = librosa.effects.pitch_shift(grain, sr=sr, n_steps=shift)
            except Exception as e:
                st.warning(f"Pitch shift failed for grain: {e}")

        if random.random() < chaos_level / 3.0:
            # Time stretch estremo
            try:
                if grain.size > 0:
                    stretch = random.uniform(0.25, 4.0)
                    grain = librosa.effects.time_stretch(grain, rate=stretch)
            except Exception as e:
                st.warning(f"Time stretch failed for grain: {e}")

        if grain.size > 0:
            grains.append(grain)

    if not grains:
        return np.array([])

    # Riorganizza i grani secondo density texture
    num_grains_output = int(len(grains) * texture_density)
    if num_grains_output <= 0:
        return np.array([])

    if num_grains_output > len(grains):
        # Duplica alcuni grani se serve più densità
        extra_grains = random.choices(grains, k=num_grains_output - len(grains))
        grains.extend(extra_grains)
    else:
        # Seleziona subset casuale
        grains = random.sample(grains, num_grains_output)

    # Crea texture riposizionando i grani
    max_length = int(len(audio) * (1 + texture_density * 0.5))
    if max_length <= 0:
        return np.array([])
    result = np.zeros(max_length)

    for grain in grains:
        if grain.size == 0: # Salta grani vuoti
            continue

        # Posizione casuale per ogni grano
        if grain.size < max_length:
            start_pos = random.randint(0, max_length - grain.size)

            # Somma il grano alla posizione (permette sovrapposizioni)
            end_pos = start_pos + grain.size
            if end_pos > max_length: # Truncate if overflows
                grain = grain[:max_length - start_pos]
                end_pos = max_length
            result[start_pos:end_pos] += grain * random.uniform(0.3, 1.0)
        else: # Se il grano è più lungo del max_length, lo taglia
            start_pos = random.randint(0, max_length // 2) if max_length > 0 else 0
            if start_pos + grain.size > max_length:
                 result[start_pos:max_length] += grain[:max_length-start_pos] * random.uniform(0.3, 1.0)
            else:
                 result[start_pos:start_pos+grain.size] += grain * random.uniform(0.3, 1.0)


    # Normalizza per evitare clipping
    if result.size > 0 and np.max(np.abs(result)) > 0:
        result = result / np.max(np.abs(result)) * 0.8
    elif result.size == 0: # Assicurati che non restituisca un array di nan/inf o vuoto
        return np.array([])

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

    # Identifica sezioni "importanti" (maggiore energia)
    hop_length = 512
    # Ensure audio is long enough for RMS calculation
    if len(audio) < hop_length:
        energy = np.array([])
    else:
        energy = librosa.feature.rms(y=audio, hop_length=hop_length)[0]

    important_frames = np.array([])
    if energy.size > 0:
        energy_threshold = np.percentile(energy, 70)  # Top 30% energy
        important_frames = np.where(energy > energy_threshold)[0]

    important_times = librosa.frames_to_time(important_frames, sr=sr, hop_length=hop_length)

    # Crea frammenti focalizzandosi sulle parti importanti
    fragments = []
    fragment_types = []  # Traccia il tipo di ogni frammento

    for t in important_times:
        start_sample = int(t * sr)
        end_sample = min(start_sample + fragment_samples, len(audio))

        if end_sample > start_sample:
            fragment = audio[start_sample:end_sample]
            if fragment.size > 0:
                fragments.append(fragment)
                fragment_types.append('important')

    # Aggiungi anche frammenti casuali per contrasto
    num_random = int(len(fragments) * 0.5)
    for _ in range(num_random):
        if len(audio) < fragment_samples: # Audio troppo corto per frammento casuale
            break
        start = random.randint(0, len(audio) - fragment_samples)
        fragment = audio[start:start + fragment_samples]
        if fragment.size > 0:
            fragments.append(fragment)
            fragment_types.append('random')

    if not fragments:
        return np.array([])

    # Applica ironia: trasforma i momenti "importanti" in modi inaspettati
    processed_fragments = []
    for i, (fragment, frag_type) in enumerate(zip(fragments, fragment_types)):
        if fragment.size == 0:
            continue

        original_fragment_len = fragment.size # Salva la lunghezza originale per il caso di np.tile

        if frag_type == 'important' and random.random() < irony_level / 2.0:
            # Trasformazioni ironiche
            ironic_transforms = [
                lambda x: x[::-1],  # Reverse
                lambda x: librosa.effects.pitch_shift(x, sr=sr, n_steps=-12) if x.size > 0 else x,  # Octave down
                lambda x: librosa.effects.time_stretch(x, rate=0.25) if x.size > 0 else x,  # Very slow
                lambda x: x * 0.1,  # Very quiet
                # Assicurati che len(x)//4 sia almeno 1 per np.tile, se x.size è 0, len(x)//4 sarà 0
                lambda x: np.tile(x[:len(x)//4 if len(x)//4 > 0 else 1], 4) if len(x) > 0 else x,  # Stutter
            ]

            transform = random.choice(ironic_transforms)
            try:
                fragment = transform(fragment)
            except Exception as e:
                st.warning(f"Ironic transform failed for fragment: {e}")
                # In caso di errore, mantieni il frammento originale o vuoto se era già vuoto
                fragment = fragment if fragment.size > 0 else np.array([])


        # Context shift: cambia il "significato" del frammento
        if random.random() < context_shift / 2.0:
            # Applica effetti che cambiano il contesto percettivo
            context_effects = [
                lambda x: librosa.effects.pitch_shift(x, sr=sr, n_steps=random.uniform(-7, 7)) if x.size > 0 else x,
                lambda x: x * np.linspace(0, 1, len(x)) if len(x) > 0 else x,  # Fade in
                lambda x: x * np.linspace(1, 0, len(x)) if len(x) > 0 else x,  # Fade out
                lambda x: x + np.random.normal(0, 0.05, len(x)) if len(x) > 0 else x,  # Add noise
            ]

            effect = random.choice(context_effects)
            try:
                fragment = effect(fragment)
            except Exception as e:
                st.warning(f"Context shift effect failed for fragment: {e}")
                fragment = fragment if fragment.size > 0 else np.array([])

        if fragment.size > 0:
            processed_fragments.append(fragment)

    if not processed_fragments:
        return np.array([])

    # Riassembla in modo da creare "falsi climax" e anticlimax
    # Ordina per energia originale
    fragment_energies = [np.mean(np.abs(frag)) for frag in processed_fragments if frag.size > 0]
    processed_fragments_filtered = [frag for frag in processed_fragments if frag.size > 0]

    if not processed_fragments_filtered:
        return np.array([])

    sorted_indices = np.argsort(fragment_energies)

    # Crea struttura postmoderna: inizia forte, poi decostruisce
    result_order_indices = []

    # Alternanza tra frammenti ad alta e bassa energia
    # Assicurati che le liste non siano vuote
    high_energy = sorted_indices[-len(sorted_indices)//2:] if sorted_indices.size > 0 else np.array([])
    low_energy = sorted_indices[:len(sorted_indices)//2] if sorted_indices.size > 0 else np.array([])

    min_len_energy = min(len(high_energy), len(low_energy))
    if min_len_energy == 0 and sorted_indices.size > 0: # Se una delle due è vuota ma ci sono frammenti, usa solo quelli
        result_order_indices.extend(list(sorted_indices))
    else:
        for i in range(min_len_energy):
            if random.random() < 0.6 and high_energy.size > i:
                result_order_indices.append(high_energy[i])
            if low_energy.size > i:
                result_order_indices.append(low_energy[i])
            if random.random() < 0.4 and high_energy.size > (len(high_energy) - (i+1)):
                result_order_indices.append(high_energy[-(i+1)])

    # Concatena i frammenti
    result_fragments_final = [processed_fragments_filtered[i] for i in result_order_indices if i < len(processed_fragments_filtered)]

    if result_fragments_final:
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

    # Analisi delle caratteristiche emotive dell'audio
    structure = analyze_audio_structure(audio, sr)
    chroma = structure['chroma']
    mfcc = structure['mfcc']
    spectral_centroids = structure['spectral_centroids']

    # Clustering delle caratteristiche per identificare "mood" diversi
    # Assicurati che features non sia vuoto
    if chroma.size == 0 or mfcc.size == 0:
        return np.array([]) # O gestire un fallback più specifico

    # Assicurati che mfcc abbia almeno 5 righe
    mfcc_sliced = mfcc[:5] if mfcc.shape[0] >= 5 else mfcc

    # Assicurati che le dimensioni delle features siano compatibili per vstack
    min_frames = min(chroma.shape[1], mfcc_sliced.shape[1])
    if min_frames == 0:
        return np.array([])

    features = np.vstack([chroma[:, :min_frames], mfcc_sliced[:, :min_frames]])
    
    if features.shape[0] == 0: # Nessun feature estratto
        return np.array([])

    features = StandardScaler().fit_transform(features.T)

    n_clusters = min(8, features.shape[0] // 10)  # Max 8 cluster
    if n_clusters < 1 and features.shape[0] > 0: # Se i dati sono pochi, almeno un cluster
        n_clusters = 1
    elif n_clusters < 1 and features.shape[0] == 0: # Se i dati sono vuoti, 0 cluster
        n_clusters = 0

    if n_clusters >= 1: # Assicurati che KMeans venga chiamato con n_clusters valido
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        mood_labels = kmeans.fit_predict(features)
    else: # Nessun cluster possibile (es. features vuote)
        mood_labels = np.array([])

    if mood_labels.size == 0: # Se non ci sono etichette di mood
        return audio # Restituisci l'audio originale o un array vuoto

    # Crea frammenti basati sui mood clusters
    fragment_samples = int(fragment_size * sr)
    if fragment_samples <= 0:
        return audio

    hop_length = 512
    frames_per_fragment = fragment_samples // hop_length
    if frames_per_fragment <= 0:
        frames_per_fragment = 1 # Almeno 1 frame per frammento

    mood_fragments = {mood: [] for mood in np.unique(mood_labels)}

    # Assicurati che il range sia valido
    max_mood_idx = len(mood_labels) - frames_per_fragment
    if max_mood_idx < 0:
        return audio # Audio troppo corto per la frammentazione

    for i in range(0, max_mood_idx + 1, frames_per_fragment):
        start_sample = i * hop_length
        end_sample = min(start_sample + fragment_samples, len(audio))

        if end_sample > start_sample:
            fragment = audio[start_sample:end_sample]
            if fragment.size > 0:
                # Assicurati che la slice di mood_labels non sia vuota
                mood_slice = mood_labels[i:i+frames_per_fragment]
                if mood_slice.size > 0:
                    dominant_mood = max(set(mood_slice), key=list(mood_slice).count)
                    mood_fragments[dominant_mood].append(fragment)

    # Filtra mood_fragments per rimuovere chiavi senza frammenti
    mood_fragments = {k: v for k, v in mood_fragments.items() if v}

    if not mood_fragments:
        return audio # Nessun frammento è stato creato

    # Crea discontinuità mischiando mood in modi inaspettati
    result_fragments = []
    current_mood = random.choice(list(mood_fragments.keys()))

    # Determina un numero massimo di iterazioni per evitare loop infiniti
    total_expected_fragments = sum(len(frags) for frags in mood_fragments.values())
    max_iterations = total_expected_fragments * 2 # Permetti un po' di "spreco" o ripetizioni

    iteration_count = 0
    while iteration_count < max_iterations and any(mood_fragments.values()): # Continua finché ci sono frammenti disponibili
        iteration_count += 1
        # Scegli frammento dal mood corrente
        if mood_fragments[current_mood]: # Questa condizione è ora su una lista, non un array
            fragment = random.choice(mood_fragments[current_mood])
            mood_fragments[current_mood].remove(fragment)

            if fragment.size == 0: # Salta frammenti vuoti
                continue

            # Applica emotional shift
            if random.random() < emotional_shift:
                # Trasformazioni che alterano l'emozione
                emotion_transforms = [
                    lambda x: librosa.effects.pitch_shift(x, sr=sr, n_steps=random.uniform(-3, 3)) if x.size > 0 else x,
                    lambda x: x * np.power(np.linspace(0.3, 1, len(x)) if len(x) > 0 else np.array([]), random.uniform(0.5, 2)) if x.size > 0 else x,
                    lambda x: librosa.effects.time_stretch(x, rate=random.uniform(0.8, 1.3)) if x.size > 0 else x,
                ]

                transform = random.choice(emotion_transforms)
                try:
                    fragment = transform(fragment)
                except Exception as e:
                    st.warning(f"Emotional shift transform failed for fragment: {e}")
                    fragment = fragment if fragment.size > 0 else np.array([])


            if fragment.size > 0:
                result_fragments.append(fragment)

        # Discontinuità: cambia mood casualmente
        if random.random() < discontinuity / 2.0:
            available_moods = [m for m in mood_fragments.keys() if mood_fragments[m]]
            if available_moods:
                current_mood = random.choice(available_moods)
            elif any(mood_fragments.values()): # Se non ci sono mood disponibili, ma ci sono ancora frammenti in altri mood
                 current_mood = random.choice([m for m in mood_fragments.keys() if mood_fragments[m]]) # scegli un mood a caso con frammenti


        # Aggiungi silenzi per enfatizzare discontinuità
        if random.random() < discontinuity / 4.0:
            silence_duration = random.uniform(0.1, 0.5)
            silence = np.zeros(int(silence_duration * sr))
            if silence.size > 0:
                result_fragments.append(silence)

    # Concatena tutto
    if result_fragments:
        result = np.concatenate(result_fragments)
    else:
        result = audio  # Fallback se non ci sono frammenti

    return result

def random_chaos(audio, sr, params):
    """Chaos totale: combina tutti i metodi casualmente"""
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

    # Applica 1-3 metodi in sequenza
    num_methods = min(len(methods), max(1, int(chaos_level)))
    chosen_methods = random.sample(methods, num_methods)

    result = audio.copy()
    for method in chosen_methods:
        if result.size == 0: # Se il risultato intermedio diventa vuoto, interrompi
            break

        # Randomizza i parametri per ogni metodo
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
            if temp_result.size == 0 and result.size > 0: # Se il metodo restituisce vuoto ma l'input non lo era, usa l'input precedente
                st.warning(f"Metodo '{method.__name__}' ha prodotto un array vuoto. Mantenendo il risultato precedente.")
            else:
                result = temp_result # Aggiorna il risultato solo se non è vuoto (o se l'input era già vuoto)

        except Exception as e:
            st.warning(f"Errore in {method.__name__}: {e}. Mantenendo il risultato precedente.")
            continue # Continua con il risultato precedente in caso di errore

    if result.size == 0: # Assicurati che random_chaos restituisca qualcosa se tutto fallisce
        return audio # Restituisci l'audio originale come fallback

    return result

def process_audio(audio_file, method, params):
    """Processa l'audio con il metodo scelto"""

    # Carica audio
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        tmp_file.write(audio_file.read())
        audio_path = tmp_file.name

    try:
        audio, sr = librosa.load(audio_path, sr=None, mono=False)

        if audio.size == 0: # Gestisci il caso di audio vuoto subito
            st.error("Il file audio caricato è vuoto o non contiene dati validi.")
            return None, None, None

        # Se stereo, processa ogni canale separatamente
        if len(audio.shape) > 1:
            processed_channels = []
            for channel in range(audio.shape[0]):
                channel_audio = audio[channel]
                processed_channel = decompose_audio(channel_audio, sr, method, params)
                if processed_channel.size > 0: # Aggiungi solo canali con dati
                    processed_channels.append(processed_channel)

            if not processed_channels: # Se tutti i canali processati sono vuoti
                st.error("La decomposizione ha prodotto risultati vuoti per tutti i canali.")
                return None, None, None

            # Allinea lunghezze dei canali
            min_length = min(len(ch) for ch in processed_channels)
            processed_channels = [ch[:min_length] for ch in processed_channels]
            processed_audio = np.array(processed_channels)
        else:
            processed_audio = decompose_audio(audio, sr, method, params)

        if processed_audio.size == 0: # Se l'audio processato è vuoto
            st.error("La decomposizione ha prodotto un file audio vuoto.")
            return None, None, None

        # Salva risultato
        output_path = tempfile.mktemp(suffix='.wav')
        # Gestisci il caso di audio mono vs stereo per soundfile.write
        if len(processed_audio.shape) > 1:
            sf.write(output_path, processed_audio.T, sr) # soundfile si aspetta (samples, channels)
        else:
            sf.write(output_path, processed_audio, sr)

        return output_path, sr, audio_path # Restituisce anche audio_path per il caricamento successivo

    except Exception as e:
        st.error(f"Errore nel processing: {e}")
        return None, None, None
    finally:
        # La pulizia di audio_path è ora gestita dopo la visualizzazione
        pass # Non rimuovere qui, lo facciamo alla fine della logica principale

def decompose_audio(audio, sr, method, params):
    """Applica il metodo di decomposizione scelto"""

    if audio.size == 0: # Controllo iniziale per audio vuoto
        return np.array([])

    # Normalizza audio di input
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio)) * 0.8
    else: # Se l'audio è tutto zero dopo la normalizzazione, consideralo vuoto
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
    if not decompose_func:
        st.error(f"Metodo {method} non riconosciuto")
        return audio

    try:
        result = decompose_func(audio, sr, params)

        # Normalizza output
        if result.size > 0 and np.max(np.abs(result)) > 0:
            result = result / np.max(np.abs(result)) * 0.8
        elif result.size == 0: # Se il risultato è vuoto, restituisci un array vuoto
            return np.array([])


        return result

    except Exception as e:
        st.error(f"Errore nella decomposizione: {e}. Dettagli tecnici: {e.__class__.__name__}: {e}")
        return audio

def create_visualization(original_audio, processed_audio, sr):
    """Crea visualizzazione comparativa degli spettrogrammi"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Spettrogramma originale
    # Controlla se l'audio è vuoto prima di STFT
    if original_audio.size > 0:
        D_orig = librosa.amplitude_to_db(np.abs(librosa.stft(original_audio)), ref=np.max)
        librosa.display.specshow(D_orig, y_axis='hz', x_axis='time', sr=sr, ax=ax1)
    ax1.set_title(' Audio Originale')
    ax1.set_ylabel('Frequenza (Hz)')

    # Spettrogramma processato
    if processed_audio.size > 0:
        D_proc = librosa.amplitude_to_db(np.abs(librosa.stft(processed_audio)), ref=np.max)
        librosa.display.specshow(D_proc, y_axis='hz', x_axis='time', sr=sr, ax=ax2)
    ax2.set_title(' Audio Decomposto/Ricomposto')
    ax2.set_xlabel('Tempo (s)')
    ax2.set_ylabel('Frequenza (Hz)')

    plt.tight_layout()
    return fig

# Interfaccia principale
if uploaded_file is not None:
    st.success(f" File caricato: {uploaded_file.name}")

    # Prepara parametri
    params = {
        'fragment_size': fragment_size,
        'chaos_level': chaos_level,
        'structure_preservation': structure_preservation
    }

    # Aggiungi parametri specifici del metodo
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

    # Pulsante per processare
    if st.button(" DECOMPONI E RICOMPONI", type="primary", use_container_width=True):

        with st.spinner(" Decomponendo il brano in arte sonora sperimentale..."):

            # Processa l'audio e ottieni anche il percorso del file temporaneo originale
            output_path, sr, original_audio_path_temp = process_audio(uploaded_file, decomposition_method, params)

            if output_path and sr:
                # Carica l'audio originale dalla sua copia temporanea sul disco
                original_audio, _ = librosa.load(original_audio_path_temp, sr=sr)
                processed_audio, _ = librosa.load(output_path, sr=sr)

                # Layout a due colonne
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader(" Audio Originale")
                    # Per la visualizzazione nel player, possiamo usare uploaded_file direttamente
                    st.audio(uploaded_file, format='audio/wav')

                    # Info originale
                    duration_orig = len(original_audio) / sr if original_audio.size > 0 else 0
                    st.info(f"""
                    **Durata:** {duration_orig:.2f} secondi
                    **Sample Rate:** {sr} Hz
                    **Samples:** {len(original_audio):,}
                    """)

                with col2:
                    st.subheader(" Audio Decomposto")

                    # Leggi il file processato per il player
                    with open(output_path, 'rb') as f:
                        audio_bytes = f.read()
                    st.audio(audio_bytes, format='audio/wav')

                    # Info processato
                    duration_proc = len(processed_audio) / sr if processed_audio.size > 0 else 0
                    st.info(f"""
                    **Durata:** {duration_proc:.2f} secondi
                    **Trasformazione:** {duration_proc/duration_orig:.2f}x (se audio originale > 0)
                    **Samples:** {len(processed_audio):,}
                    """)

                # Visualizzazione spettrogrammi
                st.subheader(" Analisi Spettrale Comparativa")

                with st.spinner("Generando visualizzazioni..."):
                    fig = create_visualization(original_audio, processed_audio, sr)
                    st.pyplot(fig)


                # Download del risultato
                st.subheader(" Download Risultato")

                with open(output_path, 'rb') as f:
                    audio_bytes = f.read()

                # Nome file di output
                original_name = uploaded_file.name.rsplit('.', 1)[0]
                output_filename = f"{original_name}_{decomposition_method}.wav"

                st.download_button(
                    label=f" Scarica {output_filename}",
                    data=audio_bytes,
                    file_name=output_filename,
                    mime="audio/wav",
                    use_container_width=True
                )

                # Analisi tecnica dettagliata
                with st.expander(" Analisi Tecnica Dettagliata"):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.write("**Audio Originale:**")
                        if original_audio.size > 0:
                            orig_structure = analyze_audio_structure(original_audio, sr)
                            st.write(f"- Tempo stimato: {orig_structure['tempo']:.1f} BPM")
                            st.write(f"- Beat rilevati: {len(orig_structure['beats'])}")
                            st.write(f"- Onset rilevati: {len(orig_structure['onset_times'])}")
                            st.write(f"- Centroide spettrale medio: {np.mean(orig_structure['spectral_centroids']):.1f} Hz")
                        else:
                            st.write("Nessun dato audio per l'analisi.")


                    with col2:
                        st.write("**Audio Processato:**")
                        if processed_audio.size > 0:
                            proc_structure = analyze_audio_structure(processed_audio, sr)
                            st.write(f"- Tempo stimato: {proc_structure['tempo']:.1f} BPM")
                            st.write(f"- Beat rilevati: {len(proc_structure['beats'])}")
                            st.write(f"- Onset rilevati: {len(proc_structure['onset_times'])}")
                            st.write(f"- Centroide spettrale medio: {np.mean(proc_structure['spectral_centroids']):.1f} Hz")
                        else:
                            st.write("Nessun dato audio per l'analisi.")


                # Pulizia file temporaneo originale e processato
                try:
                    os.unlink(output_path)
                    os.unlink(original_audio_path_temp) # Rimuovi anche il file temporaneo originale
                except Exception as e:
                    st.warning(f"Errore durante la pulizia dei file temporanei: {e}")

            else:
                st.error(" Errore nel processing dell'audio. Il risultato potrebbe essere vuoto o non valido.")

else:
    # Interfaccia di benvenuto
    st.markdown("""
    <div style='text-align: center; padding: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px;
    margin: 20px 0;'>
        <h2 style='color: white; margin-bottom: 20px;'> Benvenuto in MusicDecomposer</h2>
        <p style='color: #e0e0e0; font-size: 1.1em; margin-bottom: 30px;'>
            Trasforma i tuoi brani in arte sonora sperimentale attraverso tecniche di decomposizione e ricomposizione ispirate ai grandi movimenti dell'avanguardia musicale.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Spiegazione dei metodi
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

    # Esempi e ispirazione
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

    # Footer
    st.markdown("""
    ---
    <div style='text-align: center; color: #666; font-style: italic;'>
         MusicDecomposer by loop507 - Esplorando i confini tra ordine e caos sonoro
    </div>
    """, unsafe_allow_html=True)

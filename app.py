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
    # Estrazione features
    tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)

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

    # Calcola dimensioni frammenti in samples
    fragment_samples = int(fragment_size * sr)
    total_samples = len(audio)

    # Crea frammenti
    fragments = []
    for i in range(0, total_samples - fragment_samples, fragment_samples):
        fragment = audio[i:i + fragment_samples]

        # Aggiungi variazioni casuali alla dimensione
        if random.random() < randomness:
            variation = random.uniform(0.5, 1.5)
            new_size = int(fragment_samples * variation)

            if new_size < len(fragment):
                fragment = fragment[:new_size]
            else:
                # Stretch del frammento
                indices = np.linspace(0, len(fragment)-1, new_size)
                fragment = np.interp(indices, np.arange(len(fragment)), fragment)


        fragments.append(fragment)

    # Riassembla secondo lo stile scelto
    if reassembly == 'random':
        random.shuffle(fragments)
    elif reassembly == 'reverse':
        fragments = fragments[::-1]
        fragments = [frag[::-1] for frag in fragments]  # Reverse anche i singoli frammenti
    elif reassembly == 'palindrome':
        fragments = fragments + fragments[::-1]
    elif reassembly == 'spiral':
        # Prende alternativamente dall'inizio e dalla fine
        new_fragments = []
        start, end = 0, len(fragments) - 1
        while start <= end:
            if len(new_fragments) % 2 == 0:
                new_fragments.append(fragments[start])
                start += 1
            else:
                new_fragments.append(fragments[end])
                end -= 1
        fragments = new_fragments

    # Ricompone l'audio
    result = np.concatenate(fragments)

    return result

def remix_destrutturato(audio, sr, params):
    """Remix che mantiene elementi riconoscibili ma li ricontestualizza"""
    fragment_size = params['fragment_size']
    beat_preservation = params.get('beat_preservation', 0.4)
    melody_fragmentation = params.get('melody_fragmentation', 1.5)

    # Analizza struttura originale
    structure = analyze_audio_structure(audio, sr)
    beats = structure['beats']

    fragment_samples = int(fragment_size * sr)

    # Se conserviamo il ritmo, usiamo i beat come punti di taglio
    if beat_preservation > 0.5:
        beat_times = librosa.frames_to_time(beats, sr=sr)
        cut_points = [int(t * sr) for t in beat_times]
    else:
        # Tagli casuali
        num_cuts = int(len(audio) / fragment_samples)
        cut_points = sorted(random.sample(range(0, len(audio)), num_cuts))

    # Crea frammenti tra i punti di taglio
    fragments = []
    for i in range(len(cut_points) - 1):
        start = cut_points[i]
        end = cut_points[i + 1]
        fragment = audio[start:end]

        # Applica frammentazione melodica
        if random.random() < melody_fragmentation / 3.0:
            # Pitch shift casuale
            shift_steps = random.uniform(-7, 7)  # Semitoni
            fragment = librosa.effects.pitch_shift(fragment, sr=sr, n_steps=shift_steps)


        if random.random() < melody_fragmentation / 3.0:
            # Time stretch
            stretch_factor = random.uniform(0.7, 1.4)
            fragment = librosa.effects.time_stretch(fragment, rate=stretch_factor)

        fragments.append(fragment)

    # Riorganizza mantenendo parzialmente la struttura originale
    if beat_preservation > 0.3:
        # Mantieni alcuni frammenti nella posizione originale
        preserve_count = int(len(fragments) * beat_preservation)
        preserve_indices = random.sample(range(len(fragments)), preserve_count)

        new_order = list(range(len(fragments)))
        remaining = [i for i in range(len(fragments)) if i not in preserve_indices]
        random.shuffle(remaining)

        for i, orig_idx in enumerate(remaining):
            if i < len(preserve_indices):
                continue
            new_order[i] = orig_idx

        fragments = [fragments[i] for i in new_order]
    else:
        random.shuffle(fragments)

    # Concatena con crossfade per fluidità
    result = fragments[0]
    fade_samples = int(0.05 * sr) # 50ms crossfade

    for fragment in fragments[1:]:
        if len(result) > fade_samples and len(fragment) > fade_samples:
            # Crossfade
            fade_out = np.linspace(1, 0, fade_samples)
            fade_in = np.linspace(0, 1, fade_samples)

            result[-fade_samples:] *= fade_out
            fragment[:fade_samples] *= fade_in
            result[-fade_samples:] += fragment[:fade_samples]
            result = np.concatenate([result, fragment[fade_samples:]])
        else:
            result = np.concatenate([result, fragment])

    return result

def musique_concrete(audio, sr, params):
    """Applica tecniche di musique concrète: granular synthesis e manipolazioni concrete"""
    grain_size = params.get('grain_size', 0.1)
    texture_density = params.get('texture_density', 1.0)
    chaos_level = params['chaos_level']

    grain_samples = int(grain_size * sr)

    # Crea grani dall'audio originale
    grains = []
    for i in range(0, len(audio) - grain_samples, grain_samples // 2):  # Overlap del 50%
        grain = audio[i:i + grain_samples]

        # Applica envelope gaussiano a ogni grano
        window = signal.windows.gaussian(len(grain), std=len(grain)/6)
        grain = grain * window

        # Variazioni casuali per ogni grano
        if random.random() < chaos_level / 3.0:
            # Reverse del grano
            grain = grain[::-1]

        if random.random() < chaos_level / 3.0:
            # Pitch shift estremo
            shift = random.uniform(-12, 12)
            grain = librosa.effects.pitch_shift(grain, sr=sr, n_steps=shift)

        if random.random() < chaos_level / 3.0:
            # Time stretch estremo
            stretch = random.uniform(0.25, 4.0)
            grain = librosa.effects.time_stretch(grain, rate=stretch)

        grains.append(grain)

    # Riorganizza i grani secondo density texture
    num_grains_output = int(len(grains) * texture_density)
    if num_grains_output > len(grains):
        # Duplica alcuni grani se serve più densità
        extra_grains = random.choices(grains, k=num_grains_output - len(grains))
        grains.extend(extra_grains)
    else:
        # Seleziona subset casuale
        grains = random.sample(grains, num_grains_output)

    # Crea texture riposizionando i grani
    max_length = int(len(audio) * (1 + texture_density * 0.5))
    result = np.zeros(max_length)

    for grain in grains:
        # Posizione casuale per ogni grano
        if len(grain) < max_length:
            start_pos = random.randint(0, max_length - len(grain))

            # Somma il grano alla posizione (permette sovrapposizioni)
            result[start_pos:start_pos + len(grain)] += grain * random.uniform(0.3, 1.0)

    # Normalizza per evitare clipping
    if np.max(np.abs(result)) > 0:
        result = result / np.max(np.abs(result)) * 0.8

    return result

def decostruzione_postmoderna(audio, sr, params):
    """Decostruzione ironica e postmoderna del brano"""
    irony_level = params.get('irony_level', 1.0)
    context_shift = params.get('context_shift', 1.2)
    fragment_size = params['fragment_size']

    fragment_samples = int(fragment_size * sr)

    # Identifica sezioni "importanti" (maggiore energia)
    hop_length = 512
    energy = librosa.feature.rms(y=audio, hop_length=hop_length)[0]
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
            fragments.append(fragment)
            fragment_types.append('important')

    # Aggiungi anche frammenti casuali per contrasto
    num_random = int(len(fragments) * 0.5)
    for _ in range(num_random):
        start = random.randint(0, len(audio) - fragment_samples)
        fragment = audio[start:start + fragment_samples]
        fragments.append(fragment)
        fragment_types.append('random')

    # Applica ironia: trasforma i momenti "importanti" in modi inaspettati
    processed_fragments = []
    for i, (fragment, frag_type) in enumerate(zip(fragments, fragment_types)):
        if frag_type == 'important' and random.random() < irony_level / 2.0:
            # Trasformazioni ironiche
            ironic_transforms = [
                lambda x: x[::-1],  # Reverse
                lambda x: librosa.effects.pitch_shift(x, sr=sr, n_steps=-12),  # Octave down
                lambda x: librosa.effects.time_stretch(x, rate=0.25),  # Very slow
                lambda x: x * 0.1,  # Very quiet
                lambda x: np.tile(x[:len(x)//4], 4) if len(x) > 4 else x,  # Stutter
            ]

            transform = random.choice(ironic_transforms)
            try:
                fragment = transform(fragment)
            except:
                pass  # Se la trasformazione fallisce, mantieni l'originale

        # Context shift: cambia il "significato" del frammento
        if random.random() < context_shift / 2.0:
            # Applica effetti che cambiano il contesto percettivo
            context_effects = [
                lambda x: librosa.effects.pitch_shift(x, sr=sr, n_steps=random.uniform(-7, 7)),
                lambda x: x * np.linspace(0, 1, len(x)),  # Fade in
                lambda x: x * np.linspace(1, 0, len(x)),  # Fade out
                lambda x: x + np.random.normal(0, 0.05, len(x)),  # Add noise
            ]

            effect = random.choice(context_effects)
            try:
                fragment = effect(fragment)
            except:
                pass

        processed_fragments.append(fragment)

    # Riassembla in modo da creare "falsi climax" e anticlimax
    # Ordina per energia originale
    fragment_energies = [np.mean(np.abs(frag)) for frag in processed_fragments]
    sorted_indices = np.argsort(fragment_energies)

    # Crea struttura postmoderna: inizia forte, poi decostruisce
    result_order = []

    # Alternanza tra frammenti ad alta e bassa energia
    high_energy = sorted_indices[-len(sorted_indices)//2:]
    low_energy = sorted_indices[:len(sorted_indices)//2]

    for i in range(min(len(high_energy), len(low_energy))):
        if random.random() < 0.6:
            result_order.append(high_energy[i])
        result_order.append(low_energy[i])
        if random.random() < 0.4:
            result_order.append(high_energy[-(i+1)])

    # Concatena i frammenti
    result_fragments = [processed_fragments[i] for i in result_order]
    result = np.concatenate(result_fragments)

    return result

def decomposizione_creativa(audio, sr, params):
    """Decomposizione che enfatizza discontinuità e nuove connessioni emotive"""
    discontinuity = params.get('discontinuity', 1.0)
    emotional_shift = params.get('emotional_shift', 0.8)
    fragment_size = params['fragment_size']

    # Analisi delle caratteristiche emotive dell'audio
    structure = analyze_audio_structure(audio, sr)
    chroma = structure['chroma']
    mfcc = structure['mfcc']
    spectral_centroids = structure['spectral_centroids']

    # Clustering delle caratteristiche per identificare "mood" diversi
    features = np.vstack([chroma, mfcc[:5]])  # Usa prime 5 MFCC per semplicità
    features = StandardScaler().fit_transform(features.T)

    n_clusters = min(8, features.shape[0] // 10)  # Max 8 cluster
    if n_clusters > 1:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        mood_labels = kmeans.fit_predict(features)
    else:
        mood_labels = np.zeros(features.shape[0])

    # Crea frammenti basati sui mood clusters
    fragment_samples = int(fragment_size * sr)
    hop_length = 512
    frames_per_fragment = fragment_samples // hop_length

    mood_fragments = {mood: [] for mood in np.unique(mood_labels)}

    for i in range(0, len(mood_labels) - frames_per_fragment, frames_per_fragment):
        start_sample = i * hop_length
        end_sample = min(start_sample + fragment_samples, len(audio))

        if end_sample > start_sample:
            fragment = audio[start_sample:end_sample]
            dominant_mood = max(set(mood_labels[i:i+frames_per_fragment]),
                                key=list(mood_labels[i:i+frames_per_fragment]).count)
            mood_fragments[dominant_mood].append(fragment)

    # Crea discontinuità mischiando mood in modi inaspettati
    result_fragments = []
    current_mood = random.choice(list(mood_fragments.keys()))

    for i in range(sum(len(frags) for frags in mood_fragments.values())):
        # Scegli frammento dal mood corrente
        if mood_fragments[current_mood]:
            fragment = random.choice(mood_fragments[current_mood])
            mood_fragments[current_mood].remove(fragment)

            # Applica emotional shift
            if random.random() < emotional_shift:
                # Trasformazioni che alterano l'emozione
                emotion_transforms = [
                    lambda x: librosa.effects.pitch_shift(x, sr=sr, n_steps=random.uniform(-3, 3)),
                    lambda x: x * np.power(np.linspace(0.3, 1, len(x)), random.uniform(0.5, 2)),
                    lambda x: librosa.effects.time_stretch(x, rate=random.uniform(0.8, 1.3)),
                ]

                transform = random.choice(emotion_transforms)
                try:
                    fragment = transform(fragment)
                except:
                    pass


            result_fragments.append(fragment)

        # Discontinuità: cambia mood casualmente
        if random.random() < discontinuity / 2.0:
            available_moods = [m for m in mood_fragments.keys() if mood_fragments[m]]
            if available_moods:
                current_mood = random.choice(available_moods)

        # Aggiungi silenzi per enfatizzare discontinuità
        if random.random() < discontinuity / 4.0:
            silence_duration = random.uniform(0.1, 0.5)
            silence = np.zeros(int(silence_duration * sr))
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
            result = method(result, sr, random_params)
        except Exception as e:
            st.warning(f"Errore in {method.__name__}: {e}")
            continue

    return result

def process_audio(audio_file, method, params):
    """Processa l'audio con il metodo scelto"""

    # Carica audio
    # Questo blocco scrive il file uploaded_file su disco in un file temporaneo
    # in modo che librosa possa caricarlo correttamente in seguito.
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        tmp_file.write(audio_file.read())
        audio_path = tmp_file.name

    try:
        # Carica l'audio originale dal percorso temporaneo
        audio, sr = librosa.load(audio_path, sr=None, mono=False)

        # Se stereo, processa ogni canale separatamente
        if len(audio.shape) > 1:
            processed_channels = []
            for channel in range(audio.shape[0]):
                channel_audio = audio[channel]
                processed_channel = decompose_audio(channel_audio, sr, method, params)
                processed_channels.append(processed_channel)

            # Allinea lunghezze dei canali
            min_length = min(len(ch) for ch in processed_channels)
            processed_channels = [ch[:min_length] for ch in processed_channels]
            processed_audio = np.array(processed_channels)
        else:
            processed_audio = decompose_audio(audio, sr, method, params)

        # Salva risultato
        output_path = tempfile.mktemp(suffix='.wav')
        sf.write(output_path, processed_audio.T if len(processed_audio.shape) > 1 else processed_audio, sr)

        return output_path, sr, audio_path # Restituisce anche audio_path per il caricamento successivo

    except Exception as e:
        st.error(f"Errore nel processing: {e}")
        return None, None, None
    finally:
        # La pulizia di audio_path è ora gestita dopo la visualizzazione
        pass # Non rimuovere qui, lo facciamo alla fine della logica principale

def decompose_audio(audio, sr, method, params):
    """Applica il metodo di decomposizione scelto"""

    # Normalizza audio di input
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio)) * 0.8

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
        if len(result) > 0 and np.max(np.abs(result)) > 0:
            result = result / np.max(np.abs(result)) * 0.8

        return result

    except Exception as e:
        st.error(f"Errore nella decomposizione: {e}")
        return audio

def create_visualization(original_audio, processed_audio, sr):
    """Crea visualizzazione comparativa degli spettrogrammi"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Spettrogramma originale
    D_orig = librosa.amplitude_to_db(np.abs(librosa.stft(original_audio)), ref=np.max)
    librosa.display.specshow(D_orig, y_axis='hz', x_axis='time', sr=sr, ax=ax1)
    ax1.set_title(' Audio Originale')
    ax1.set_ylabel('Frequenza (Hz)')

    # Spettrogramma processato
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
                # Questo evita l'errore LibsndfileError
                original_audio, _ = librosa.load(original_audio_path_temp, sr=sr)
                processed_audio, _ = librosa.load(output_path, sr=sr)

                # Layout a due colonne
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader(" Audio Originale")
                    # Per la visualizzazione nel player, possiamo usare uploaded_file direttamente
                    # o il path temporaneo (se vuoi essere coerente)
                    st.audio(uploaded_file, format='audio/wav')

                    # Info originale
                    duration_orig = len(original_audio) / sr
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
                    duration_proc = len(processed_audio) / sr
                    st.info(f"""
                    **Durata:** {duration_proc:.2f} secondi
                    **Trasformazione:** {duration_proc/duration_orig:.2f}x
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
                        orig_structure = analyze_audio_structure(original_audio, sr)
                        st.write(f"- Tempo stimato: {orig_structure['tempo']:.1f} BPM")
                        st.write(f"- Beat rilevati: {len(orig_structure['beats'])}")
                        st.write(f"- Onset rilevati: {len(orig_structure['onset_times'])}")
                        st.write(f"- Centroide spettrale medio: {np.mean(orig_structure['spectral_centroids']):.1f} Hz")

                    with col2:
                        st.write("**Audio Processato:**")
                        proc_structure = analyze_audio_structure(processed_audio, sr)
                        st.write(f"- Tempo stimato: {proc_structure['tempo']:.1f} BPM")
                        st.write(f"- Beat rilevati: {len(proc_structure['beats'])}")
                        st.write(f"- Onset rilevati: {len(proc_structure['onset_times'])}")
                        st.write(f"- Centroide spettrale medio: {np.mean(proc_structure['spectral_centroids']):.1f} Hz")

                # Pulizia file temporaneo originale e processato
                try:
                    os.unlink(output_path)
                    os.unlink(original_audio_path_temp) # Rimuovi anche il file temporaneo originale
                except Exception as e:
                    st.warning(f"Errore durante la pulizia dei file temporanei: {e}")

            else:
                st.error(" Errore nel processing dell'audio")

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
        - Luc Ferrari
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

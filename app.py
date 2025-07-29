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

# PARAMETRI FISSI PER MASSIMA ELABORAZIONE (NON PIÙ DIPENDENTI DALL'INTENSITÀ)
# Ho rimosso i preset 'soft', 'medio', 'hard'
# Questi parametri ora rappresentano un'elaborazione "massima" o "profonda"
FIXED_PARAMS = {
    'cut_up_sonoro': {
        'fragment_size': 0.8, # Frammenti più piccoli per maggiore manipolazione
        'chaos_level': 0.9,
        'structure_preservation': 0.2,
        'cut_randomness': 0.8,
        'reassembly_style': 'spiral' # Stile più complesso
    },
    'remix_destrutturato': {
        'fragment_size': 0.6, # Frammenti più piccoli
        'chaos_level': 0.8,
        'structure_preservation': 0.3,
        'beat_preservation': 0.3,
        'melody_fragmentation': 1.8 # Maggiore frammentazione
    },
    'musique_concrete': {
        'fragment_size': 0.5, # Grani più piccoli
        'chaos_level': 0.8,
        'structure_preservation': 0.2,
        'grain_size': 0.1,
        'texture_density': 1.5 # Maggiore densità
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
        'discontinuity': 1.2, # Maggiore discontinuità
        'emotional_shift': 1.0
    },
    'random_chaos': {
        'fragment_size': 1.0,
        'chaos_level': 1.0, # Massimo caos
        'structure_preservation': 0.1
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
    # MANTENUTO IL LIMITE DI 60 SECONDI PER L'ANALISI INIZIALE PER PREVENIRE BLOCCHI
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

    # *** MODIFICA QUI: Rimosso il limite 'max_fragments' o aumentato significativamente ***
    # Calcola il numero massimo di frammenti basato sull'intera lunghezza dell'audio
    # Considera una sovrapposizione per frammenti più piccoli o più densità
    num_possible_fragments = len(audio) // fragment_samples
    
    # Se il numero di frammenti è estremamente alto, possiamo impostare un limite superiore ragionevole
    # per evitare un consumo eccessivo di memoria nel caso di frammenti minuscoli su audio lunghi.
    # Ad esempio, non più di 500 frammenti, che su un audio di 5 minuti a 22050Hz (6.6M campioni)
    # significherebbe frammenti di almeno 13200 campioni (circa 0.6 secondi)
    max_fragments_to_process = min(num_possible_fragments, 500) 

    fragments = []
    # Genera i punti di inizio dei frammenti in modo più uniforme sull'intero audio
    # Usa un passo per controllare quanti frammenti vengono estratti
    if max_fragments_to_process > 0:
        step = max(1, len(audio) // max_fragments_to_process)
    else:
        step = fragment_samples # Fallback

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

    # *** MODIFICA QUI: Aumentato il numero massimo di tagli (max_cuts) ***
    # Ora calcoliamo max_cuts in base alla lunghezza totale dell'audio e alla dimensione del frammento,
    # poi impostiamo un limite massimo ragionevole per evitare un sovraccarico eccessivo.
    max_cuts_based_on_length = len(audio) // fragment_samples
    max_cuts = min(max_cuts_based_on_length, 200) # Limite massimo di 200 tagli

    cut_points = []
    if beat_preservation > 0.5 and beats.size > 0:
        beat_times = librosa.frames_to_time(beats, sr=sr)
        cut_points = [int(t * sr) for t in beat_times[:int(max_cuts)] if t * sr < len(audio)]

    if len(cut_points) == 0 or beat_preservation <= 0.5:
        num_cuts = min(int(max_cuts), int(len(audio) / fragment_samples))
        if num_cuts > 0 and len(audio) > 0:
            # Assicurati che il range per random.sample sia valido
            if (len(audio) // fragment_samples) > 0:
                cut_points = sorted(random.sample(range(0, len(audio), fragment_samples), min(num_cuts, (len(audio) // fragment_samples))))
            else:
                cut_points = [] # Nessun punto di taglio se l'audio è troppo corto

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
            if random.random() < melody_fragmentation / 2.0:  # Aumentata probabilità
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

    # Riassemblaggio semplificato
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
    """Applica tecniche di musique concrète - Versione ottimizzata"""
    grain_size = params.get('grain_size', 0.1)
    texture_density = params.get('texture_density', 1.0)
    chaos_level = params['chaos_level']

    if audio.size == 0:
        return np.array([])

    grain_samples = int(grain_size * sr)
    if grain_samples <= 0:
        return audio

    # *** MODIFICA QUI: Aumentato il numero massimo di grani (max_grains) ***
    # Calcola il numero massimo di grani in base alla lunghezza totale dell'audio e alla dimensione del grano.
    # Poi imposta un limite superiore ragionevole per evitare sovraccarico.
    max_grains_based_on_length = len(audio) // (grain_samples // 4) # Step più piccolo per più grani
    max_grains = min(max_grains_based_on_length, 1000) # Limite massimo di 1000 grani

    # step controlla la densità di estrazione dei grani
    step = max(grain_samples // 4, len(audio) // max_grains) if max_grains > 0 else grain_samples // 4

    grains = []
    for i in range(0, min(len(audio) - grain_samples + 1, max_grains * step), step):
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

        if random.random() < chaos_level / 2.0: # Aumentata probabilità
            grain = grain[::-1]

        if grain.size > 0 and random.random() < chaos_level / 2.0:
            try:
                shift = random.uniform(-12, 12) # Range ampliato
                grain = safe_pitch_shift(grain, sr, shift)
            except Exception:
                grain = np.array([])

        if grain.size > 0 and random.random() < chaos_level / 2.0:
            try:
                stretch = random.uniform(0.25, 4.0) # Range ampliato
                grain = safe_time_stretch(grain, stretch)
            except Exception:
                grain = np.array([])

        if grain.size > 0:
            grains.append(grain)

    if len(grains) == 0:
        return np.array([])

    # Seleziona un numero gestibile di grani (basato sulla densità e sul caos_level)
    # Aumentiamo il numero massimo di grani da includere nel risultato
    num_grains_output = min(int(len(grains) * texture_density), 500) # Aumentato limite di output
    if num_grains_output <= 0:
        return np.array([])

    if num_grains_output < len(grains):
        grains = random.sample(grains, num_grains_output)

    # Crea il risultato finale
    # La lunghezza massima del risultato può essere più grande per accogliere più grani sovrapposti
    max_length = min(int(len(audio) * (1 + texture_density * 0.5)), len(audio) * 3) # Limite lunghezza ampliato
    result = np.zeros(max_length)

    for grain in grains:
        if grain.size == 0:
            continue

        if grain.size < max_length:
            start_pos = random.randint(0, max_length - grain.size)
            end_pos = start_pos + grain.size
            
            # Assicurati che grain non ecceda la dimensione di result
            grain_to_add = grain[:min(grain.size, end_pos - start_pos)]
            if grain_to_add.size > 0:
                result[start_pos : start_pos + grain_to_add.size] += grain_to_add * random.uniform(0.5, 0.8)

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

    # *** MODIFICA QUI: Aumentato il numero massimo di frammenti (max_fragments) ***
    max_fragments_based_on_length = len(audio) // fragment_samples
    max_fragments = min(max_fragments_based_on_length, 100) # Limite massimo di 100 frammenti
    
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
        if len(audio) > fragment_samples * 10:
            sample_indices = np.linspace(0, len(audio) - fragment_samples, min(10, len(audio) // fragment_samples), dtype=int)
            sample_energies = [np.sqrt(np.mean(audio[j:j+fragment_samples]**2)) for j in sample_indices]
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

        if frag_type == 'important' and random.random() < irony_level * 0.5: # Aumentata probabilità
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

        if processed_frag.size > 0 and random.random() < context_shift * 0.4: # Aumentata probabilità
            effect_choice = random.choice(['fade_in', 'light_noise', 'time_stretch_subtle'])
            
            try:
                if effect_choice == 'fade_in':
                    fade = np.linspace(0.1, 1, len(processed_frag))
                    processed_frag = processed_frag * fade
                elif effect_choice == 'light_noise':
                    noise = np.random.normal(0, 0.01, len(processed_frag)) # Più rumore
                    processed_frag = processed_frag + noise
                elif effect_choice == 'time_stretch_subtle':
                    processed_frag = safe_time_stretch(processed_frag, random.uniform(0.8, 1.2))
            except Exception:
                processed_frag = fragment

        if processed_frag.size > 0:
            processed_fragments.append(processed_frag)

    if len(processed_fragments) == 0:
        return audio

    # Riassemblaggio: usa tutti i frammenti processati
    random.shuffle(processed_fragments)
    final_fragments = processed_fragments

    if len(final_fragments) == 0:
        return audio
    
    if len(final_fragments) == 1:
        return final_fragments[0]

    result = final_fragments[0].copy()
    
    for next_fragment in final_fragments[1:]:
        crossfade_samples = min(256, len(result) // 4, len(next_fragment) // 4) # Crossfade più lungo
        
        if crossfade_samples > 0:
            fade_out = np.linspace(1, 0, crossfade_samples)
            fade_in = np.linspace(0, 1, crossfade_samples)
            
            # Evita errori se il segmento finale è più corto del crossfade
            if len(result) >= crossfade_samples:
                result[-crossfade_samples:] = result[-crossfade_samples:] * fade_out
            else: # Se troppo corto, svanisce del tutto
                result = result * np.linspace(1, 0, len(result))
                
            if len(next_fragment) >= crossfade_samples:
                result = np.concatenate([result, next_fragment[crossfade_samples:] + next_fragment[:crossfade_samples] * fade_in])
            else: # Se il prossimo frammento è troppo corto, aggiungilo con un fade-in parziale
                result = np.concatenate([result, next_fragment * np.linspace(0, 1, len(next_fragment))])
        else:
            result = np.concatenate([result, next_fragment])

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

    # *** MODIFICA QUI: Aumentato il numero massimo di frammenti (max_fragments) ***
    max_fragments_based_on_length = len(audio) // fragment_samples
    max_fragments = min(max_fragments_based_on_length, 150) # Limite massimo di 150 frammenti
    
    # Crea punti di taglio cercando di coprire l'intero audio
    cut_points = sorted(list(set([0] + 
                                 [int(t * sr) for t in onset_times[:int(max_fragments * 0.5)] if int(t * sr) < len(audio)] + # Usa più onset
                                 [i * fragment_samples for i in range(0, max_fragments) if i * fragment_samples < len(audio)]))) # E più frammenti regolari
    
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

        if random.random() < discontinuity * 0.15: # Aumentata probabilità
            if random.random() < 0.5:
                fragment = np.zeros_like(fragment) * random.uniform(0.05, 0.2) # Silenzio parziale
            else:
                continue # Rimuove il frammento

        if random.random() < emotional_shift * 0.3: # Aumentata probabilità
            if random.random() < 0.5:
                shift_steps = random.uniform(-12 * emotional_shift, 12 * emotional_shift) # Range ampliato
                fragment = safe_pitch_shift(fragment, sr, shift_steps)
            else:
                stretch_rate = random.uniform(1 - 0.4 * emotional_shift, 1 + 0.4 * emotional_shift) # Range ampliato
                fragment = safe_time_stretch(fragment, stretch_rate)
        
        if fragment.size > 0 and random.random() < chaos_level * 0.1: # Aumentata probabilità
            fragment = fragment[::-1] # Inversione

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
    """Random Chaos ottimizzato per performance"""
    chaos_level = params['chaos_level']

    if audio.size == 0:
        return np.array([])

    processed_audio = audio.copy()

    # Pitch Shift
    if random.random() < 0.5 * chaos_level: # Probabilità aumentata
        shift_steps = random.uniform(-12, 12) 
        processed_audio = safe_pitch_shift(processed_audio, sr, shift_steps)
        if processed_audio.size == 0: 
            return np.array([])

    # Time Stretch
    if random.random() < 0.5 * chaos_level: # Probabilità aumentata
        stretch_rate = random.uniform(0.25, 4.0) 
        processed_audio = safe_time_stretch(processed_audio, stretch_rate)
        if processed_audio.size == 0: 
            return np.array([])

    # Inversione casuale
    if random.random() < 0.4 * chaos_level: # Probabilità aumentata
        if processed_audio.size > sr * 2 and random.random() < 0.7: # Più probabile su segmenti lunghi
            segment_len = int(random.uniform(sr * 1, sr * 5)) # Inverte segmenti più lunghi (1-5 sec)
            start_idx = random.randint(0, max(0, processed_audio.size - segment_len))
            end_idx = min(processed_audio.size, start_idx + segment_len)
            if end_idx > start_idx:
                processed_audio[start_idx:end_idx] = processed_audio[start_idx:end_idx][::-1]
        else:
            processed_audio = processed_audio[::-1] # Inverte tutto

    # Rumore
    if random.random() < 0.4 * chaos_level: # Probabilità aumentata
        if processed_audio.size > 0:
            noise_amplitude = random.uniform(0.01, 0.1) * chaos_level # Più rumore
            noise = np.random.normal(0, noise_amplitude, processed_audio.size)
            processed_audio = processed_audio + noise

    # Frammentazione
    if random.random() < 0.5 * chaos_level: # Probabilità aumentata
        if processed_audio.size > 0:
            max_fragment_len = int(sr * 5) # Frammenti fino a 5 secondi
            chaos_fragments = []
            current_pos = 0
            # *** MODIFICA QUI: Aumentato il numero massimo di frammenti ***
            max_fragments = 50 # Aumentato limite numero frammenti
            
            while current_pos < processed_audio.size and len(chaos_fragments) < max_fragments:
                frag_len = min(random.randint(int(sr * 0.2), max_fragment_len), 
                              processed_audio.size - current_pos)
                if frag_len <= 0: 
                    break
                chaos_fragments.append(processed_audio[current_pos : current_pos + frag_len])
                current_pos += frag_len + int(sr * random.uniform(0, 0.2)) # Più spazi/sovrapposizioni

            if chaos_fragments:
                random.shuffle(chaos_fragments)
                # Concatenazione con crossfade casuale per evitare click
                result_chaos = chaos_fragments[0]
                for next_frag_chaos in chaos_fragments[1:]:
                    if result_chaos.size == 0:
                        result_chaos = next_frag_chaos
                        continue
                    if next_frag_chaos.size == 0:
                        continue
                    
                    crossfade_len_chaos = int(sr * random.uniform(0.01, 0.05)) # Crossfade casuale
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

# Logica principale per processare l'audio
if uploaded_file is not None:
    # Sample rate fisso a 22050 Hz
    target_sr = 22050
    st.sidebar.subheader("⚙️ Impostazioni Caricamento Audio")
    st.sidebar.markdown(f"**Sample Rate fisso per l'elaborazione:** `{target_sr} Hz` (Qualità Radio)")
    st.sidebar.info("Questa impostazione massimizza la compatibilità e l'efficienza per file audio più lunghi.")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        # Leggi l'audio con limitazione a 5 minuti e il sample rate fisso
        audio, sr = librosa.load(tmp_file_path, sr=target_sr, duration=300)  # Massimo 5 minuti

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Durata Caricata", f"{len(audio)/sr:.2f} sec")
        with col2:
            st.metric("Sample Rate Elaborazione", f"{sr} Hz")
        with col3:
            st.metric("Canali", "Mono")

        st.subheader("Audio Originale (Caricato)")
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
                
                # Ottieni parametri fissi
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
                    processed_tmp_path = ""
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as processed_tmp:
                            sf.write(processed_tmp.name, processed_audio, sr)
                            processed_tmp_path = processed_tmp.name

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
                        
                        with open(processed_tmp_path, 'rb') as audio_file:
                            audio_bytes = audio_file.read()
                            st.audio(audio_bytes, format='audio/wav')

                        method_names = {
                            "cut_up_sonoro": "CutUp",
                            "remix_destrutturato": "RemixDestrutturato", 
                            "musique_concrete": "MusiqueConcrete",
                            "decostruzione_postmoderna": "DecostruzionePostmoderna",
                            "decomposizione_creativa": "DecomposizioneCreativa",
                            "random_chaos": "RandomChaos"
                        }
                        
                        filename = f"{uploaded_file.name.split('.')[0]}_{method_names[selected_method]}_MAX.wav"
                        
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
                        
                        st.subheader("Sintesi Artistica della Decomposizione")
                        st.markdown(f"""
                        Con il metodo **"{method_labels[selected_method]}"** applicato con la **massima elaborazione**, il brano originale è stato trasformato in un'opera sonora unica e profonda.
                        """)
                        
                        st.markdown("---")
                        st.markdown("**Descrizione della Tecnica Applicata:**")
                        st.markdown(technique_descriptions[selected_method])

                        st.markdown("---")
                        st.markdown("### Riepilogo dei Cambiamenti Quantitativi:")

                        analysis_text = f"""
                        * **Durata:** L'audio originale (fino a 5 minuti) è stato trasformato in **{new_duration:.2f} secondi** ({'allungamento' if new_duration > (len(audio) / sr) else 'accorciamento'} di **{abs(new_duration - (len(audio) / sr)):.2f} secondi**).
                        * **Sample Rate Elaborazione:** Il brano è stato elaborato a **{sr} Hz**.
                        * **Energia RMS:** Variazione di **{processed_rms - original_rms:.4f}** - il suono risultante è {'più forte' if processed_rms > original_rms else 'più debole' if processed_rms < original_rms else 'simile'} in volume medio.
                        * **Variazione Spettrale:** **{spectral_diff:.2e}** - quantifica il cambiamento del "colore" e distribuzione delle frequenze rispetto all'originale.
                        
                        **Parametri Applicati** (Massima Elaborazione):
                        """
                        st.markdown(analysis_text)
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

                        with st.expander("📊 Confronto Forme d'Onda"):
                            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
                            
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
                            ax2.set_title(f"Forma d'Onda Decomposta ({method_labels[selected_method]})")
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
                        try:
                            if os.path.exists(tmp_file_path):
                                os.unlink(tmp_file_path)
                            if processed_tmp_path and os.path.exists(processed_tmp_path):
                                os.unlink(processed_tmp_path)
                        except Exception as e:
                            st.error(f"Errore durante la pulizia: {e}")
    except Exception as e:
        st.error(f"❌ Errore nel processamento: {str(e)}")
        st.error(f"Dettagli: {traceback.format_exc()}")

else:
    st.info("👆 Carica un file audio per iniziare la decomposizione")
    
    with st.expander("📖 Come usare MusicDecomposer"):
        st.markdown("""
        ### Modalità di utilizzo:

        1.  **Carica il tuo file audio** (MP3, WAV, M4A, FLAC, OGG).
        2.  L'audio verrà automaticamente processato a **22050 Hz** per ottimizzare le performance.
        3.  **Seleziona il metodo di decomposizione** che preferisci. Tutti i metodi applicheranno la **massima elaborazione**.
        4.  **Clicca "SCOMPONI E RICOMPONI"** per generare l'arte sonora.

        ### Metodi disponibili (Massima Elaborazione):
        - **🎭 Cut-up Sonoro**: Collage sonoro con frammenti riassemblati in modo intenso.
        - **🔄 Remix Destrutturato**: Remix creativo che spinge i limiti della riorganizzazione.  
        - **🎵 Musique Concrète**: Manipolazione granulare profonda per texture astratte complesse.
        - **🏛️ Decostruzione Postmoderna**: Approccio critico radicale con ironia e rotture marcate.
        - **🎨 Decomposizione Creativa**: Focus su discontinuità estreme e shift emotivi pronunciati.
        - **🌪️ Random Chaos**: Trasformazioni altamente imprevedibili e massimali.

        ### Ottimizzazioni:
        - ⚡ Tutti i metodi sono ora ottimizzati per file fino a 5 minuti con elaborazione approfondita.
        - 🎛️ Parametri preimpostati per garantire la massima intensità per ogni stile.
        - 🚀 Performance bilanciate per un'esperienza d'uso stabile.
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

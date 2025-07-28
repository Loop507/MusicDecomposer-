import streamlit as st
import librosa
import numpy as np
import soundfile as sf
import io
import gc
import traceback
import random

# --- Funzioni di utilità per la gestione degli errori e il pitch/time shift sicuro ---
def safe_pitch_shift(y, sr, n_steps):
    try:
        return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps)
    except Exception as e:
        st.warning(f"⚠️ Errore durante il pitch shift: {e}. Verrà usato l'audio originale.")
        return y

def safe_time_stretch(y, sr, rate):
    try:
        return librosa.effects.time_stretch(y=y, sr=sr, rate=rate)
    except Exception as e:
        st.warning(f"⚠️ Errore durante il time stretch: {e}. Verrà usato l'audio originale.")
        return y

# --- Funzioni di elaborazione audio ---

def cut_up_sonoro(audio, sr, num_segments=10, overlap_factor=0.5):
    segment_length = len(audio) // num_segments
    output_audio = np.array([])
    for i in range(num_segments):
        start = int(i * segment_length * (1 - overlap_factor))
        end = int(start + segment_length)
        if end > len(audio):
            end = len(audio)
            start = end - segment_length # Assicurati che il segmento abbia la lunghezza corretta alla fine

        segment = audio[start:end]

        # Inverti casualmente
        if random.random() < 0.3:
            segment = segment[::-1]

        # Pitch shift casuale (con gestione errori)
        if random.random() < 0.4:
            segment = safe_pitch_shift(segment, sr, random.uniform(-2, 2))

        # Time stretch casuale (con gestione errori)
        if random.random() < 0.4:
            segment = safe_time_stretch(segment, sr, random.uniform(0.8, 1.2))

        output_audio = np.concatenate((output_audio, segment))
    return output_audio

def remix_destrutturato(audio, sr, num_shuffles=5, segment_duration=0.5):
    segment_samples = int(segment_duration * sr)
    segments = [audio[i:i + segment_samples] for i in range(0, len(audio), segment_samples)]

    if not segments:
        return np.array([])

    shuffled_segments = segments[:]
    for _ in range(num_shuffles):
        random.shuffle(shuffled_segments)

    output_audio = np.array([])
    for segment in shuffled_segments:
        # Inverti, pitch shift, time stretch casuale per ogni segmento (con gestione errori)
        processed_segment = segment
        if random.random() < 0.2:
            processed_segment = processed_segment[::-1]
        if random.random() < 0.3:
            processed_segment = safe_pitch_shift(processed_segment, sr, random.uniform(-3, 3))
        if random.random() < 0.3:
            processed_segment = safe_time_stretch(processed_segment, sr, random.uniform(0.7, 1.3))

        output_audio = np.concatenate((output_audio, processed_segment))
    return output_audio

def optimized_musique_concrete(audio, sr, grain_duration=0.1, density=0.5, pan=0.0):
    grain_samples = int(grain_duration * sr)
    output_audio = np.zeros_like(audio) # Inizializza un array di zeri delle stesse dimensioni

    # Limita il numero massimo di grani per evitare eccessivo consumo di memoria
    # Questo è un limite critico per file lunghi.
    max_possible_grains = len(audio) // grain_samples
    num_grains = min(int(max_possible_grains * density * 2), 500) # Max 500 grani o meno se l'audio è troppo corto

    if num_grains == 0:
        return np.array([]) # Nessun grano da processare

    for _ in range(num_grains):
        start_sample = random.randint(0, len(audio) - grain_samples)
        grain = audio[start_sample : start_sample + grain_samples]

        # Applica effetti casuali
        if random.random() < 0.5:
            grain = grain[::-1] # Inverti

        if random.random() < 0.5:
            grain = safe_pitch_shift(grain, sr, random.uniform(-5, 5)) # Pitch shift

        if random.random() < 0.5:
            grain = safe_time_stretch(grain, sr, random.uniform(0.5, 2.0)) # Time stretch

        # Mixa il grano nell'output
        mix_start = random.randint(0, len(output_audio) - len(grain))
        output_audio[mix_start : mix_start + len(grain)] += grain * random.uniform(0.3, 1.0) # Con un po' di volume casuale

    return output_audio

def decomposizione_creativa(audio, sr, n_fft=2048, hop_length=512, effect_strength=0.5):
    # Esempio: applica uno spectral gating o manipolazione dello spettrogramma
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    magnitude, phase = librosa.magphase(stft)

    # Applica un effetto semplice, es. enfatizzare/ridurre alcune frequenze
    # Questa è una semplificazione. Un effetto reale sarebbe più complesso.
    magnitude = magnitude * (1 + np.sin(np.linspace(0, np.pi, magnitude.shape[0]))[:, np.newaxis] * effect_strength)

    stft_processed = magnitude * phase
    processed_audio = librosa.istft(stft_processed, hop_length=hop_length)
    return processed_audio

def ultra_optimized_random_chaos(audio, sr, fragment_duration=0.1, chaos_level=0.5):
    fragment_length = int(fragment_duration * sr)
    if fragment_length == 0: # Evita divisione per zero
        return np.array([])

    # Limita il numero di frammenti per audio lunghi
    max_fragments_allowed = 20 # Limite massimo di frammenti totali
    num_fragments = min(max_fragments_allowed, len(audio) // fragment_length)

    if num_fragments == 0:
        return np.array([]) # Non ci sono abbastanza frammenti

    fragments = []
    for i in range(num_fragments):
        start_idx = random.randint(0, len(audio) - fragment_length)
        fragments.append(audio[start_idx : start_idx + fragment_length])

    output_audio = np.array([])
    for fragment in fragments:
        if random.random() < chaos_level: # Applica effetti solo se sotto il livello di caos
            # Evitiamo pitch_shift e time_stretch perché sono quelli che crashano di più
            if random.random() < 0.5:
                fragment = fragment[::-1] # Inverti
            # Altri effetti leggeri, es. guadagno
            fragment = fragment * random.uniform(0.5, 1.5)

        output_audio = np.concatenate((output_audio, fragment))
    return output_audio

# --- Gestione del chunking per audio lunghi ---
def process_audio_in_chunks(audio, sr, processing_function, params, chunk_duration_sec=30):
    chunk_samples = int(chunk_duration_sec * sr)
    processed_chunks = []
    total_chunks = (len(audio) + chunk_samples - 1) // chunk_samples

    progress_text = f"Elaborazione in chunks... 0/{total_chunks} completato."
    progress_bar = st.progress(0, text=progress_text)

    for i in range(total_chunks):
        start_sample = i * chunk_samples
        end_sample = min((i + 1) * chunk_samples, len(audio))
        chunk = audio[start_sample:end_sample]

        # Processa il chunk con la funzione specificata
        processed_chunk = processing_function(chunk, sr, **params)
        processed_chunks.append(processed_chunk)

        # Aggiorna la barra di progresso
        progress_bar.progress((i + 1) / total_chunks, text=f"Elaborazione in chunks... {i+1}/{total_chunks} completato.")

        # Garbage collection forzato dopo ogni chunk
        del chunk
        del processed_chunk
        gc.collect()

    return np.concatenate(processed_chunks)


# --- Wrapper per la funzione di processing, gestisce il chunking ---
def safe_process_audio(audio, sr, decomposition_method, params):
    duration = librosa.get_duration(y=audio, sr=sr)

    # --- NUOVI LIMITI: Soglia di durata per il chunking ---
    # Se l'audio supera questa durata, useremo il chunking.
    # Puoi regolare questo valore. Un valore più basso significa chunking più frequente.
    CHUNK_THRESHOLD_SECONDS = 30 # Esempio: inizia il chunking per audio più lunghi di 30 secondi
    DYNAMIC_CHUNK_DURATION = 30 # Durata base del chunk. Può essere ridotta per audio ESTREMAMENTE lunghi.

    if duration > 180: # Se l'audio è estremamente lungo (es. > 3 minuti)
        DYNAMIC_CHUNK_DURATION = 15 # Riduci la dimensione del chunk a 15 secondi
        st.warning(f"Audio molto lungo ({duration:.2f}s). Ridotto la dimensione del chunk a {DYNAMIC_CHUNK_DURATION}s per maggiore stabilità.")


    processing_function = globals().get(decomposition_method) # Ottieni la funzione dal nome

    if processing_function is None:
        raise ValueError(f"Metodo di decomposizione '{decomposition_method}' non trovato.")

    if duration > CHUNK_THRESHOLD_SECONDS:
        st.info(f"⏳ Audio lungo rilevato ({duration:.2f}s). Elaborazione in chunks di {DYNAMIC_CHUNK_DURATION}s per risparmiare memoria...")
        return process_audio_in_chunks(audio, sr, processing_function, params, chunk_duration_sec=DYNAMIC_CHUNK_DURATION)
    else:
        return processing_function(audio, sr, **params)


# --- Interfaccia utente Streamlit ---
st.title("🎼 Sound Decomposer e Remix")
st.markdown("Carica un file audio e applica varie tecniche di decomposizione e remix!")

uploaded_file = st.file_uploader("Carica un file audio (WAV, MP3, FLAC, ecc.)", type=["wav", "mp3", "flac", "ogg"])

if uploaded_file is not None:
    # --- NUOVO LIMITE: Durata massima del file caricato ---
    MAX_FILE_DURATION_SECONDS = 300 # 5 minuti (300 secondi)
    # Puoi impostare un limite superiore se necessario, ma considera le risorse.
    # Per una web app, 5-10 minuti è un buon punto di partenza.

    try:
        # Carica il file audio
        with st.spinner("Caricamento audio..."):
            audio, sr = librosa.load(uploaded_file, sr=None) # sr=None per mantenere il sample rate originale

        duration = librosa.get_duration(y=audio, sr=sr)

        if duration > MAX_FILE_DURATION_SECONDS:
            st.error(f"❌ File troppo lungo! La durata massima consentita è {MAX_FILE_DURATION_SECONDS / 60:.0f} minuti. Il tuo file è di {duration:.2f} secondi.")
            st.stop() # Ferma l'esecuzione

        st.success(f"✅ Audio caricato. Durata: {duration:.2f} secondi, Sample Rate: {sr} Hz")

        st.sidebar.header("Parametri di Elaborazione")
        decomposition_method = st.sidebar.selectbox(
            "Seleziona il Metodo di Decomposizione:",
            ("cut_up_sonoro", "remix_destrutturato", "optimized_musique_concrete", "decomposizione_creativa", "ultra_optimized_random_chaos")
        )

        params = {}
        if decomposition_method == "cut_up_sonoro":
            params["num_segments"] = st.sidebar.slider("Numero di Segmenti", 2, 50, 10)
            params["overlap_factor"] = st.sidebar.slider("Fattore di Sovrapposizione", 0.0, 0.9, 0.5)
        elif decomposition_method == "remix_destrutturato":
            params["num_shuffles"] = st.sidebar.slider("Numero di Rimescolamenti", 1, 20, 5)
            params["segment_duration"] = st.sidebar.slider("Durata Segmento (secondi)", 0.1, 2.0, 0.5)
        elif decomposition_method == "optimized_musique_concrete":
            params["grain_duration"] = st.sidebar.slider("Durata Grano (secondi)", 0.01, 0.5, 0.1)
            # NUOVO LIMITE: Limita la densità per file lunghi
            max_density = 1.0
            if duration > 60: # Se l'audio è più lungo di 60 secondi
                max_density = 0.5 # Riduci la densità massima consentita a 0.5
                st.sidebar.warning(f"Densità massima ridotta a {max_density} per audio > 60s.")
            params["density"] = st.sidebar.slider("Densità dei Grani", 0.01, max_density, 0.5)
            params["pan"] = st.sidebar.slider("Pan (0.0 centrale)", -1.0, 1.0, 0.0) # Il pan non è usato nella funzione attuale, ma si può aggiungere
        elif decomposition_method == "decomposizione_creativa":
            params["n_fft"] = st.sidebar.slider("NFFT (dimensione finestra FFT)", 512, 4096, 2048, step=256)
            params["hop_length"] = st.sidebar.slider("Hop Length", 128, 1024, 512, step=64)
            params["effect_strength"] = st.sidebar.slider("Intensità Effetto Spettrale", 0.0, 1.0, 0.5)
        elif decomposition_method == "ultra_optimized_random_chaos":
            params["fragment_duration"] = st.sidebar.slider("Durata Frammento (secondi)", 0.01, 0.5, 0.1)
            # NUOVO LIMITE: Limita il livello di caos per file lunghi
            max_chaos_level = 1.0
            if duration > 60: # Se l'audio è più lungo di 60 secondi
                max_chaos_level = 0.7 # Riduci il caos massimo consentito a 0.7
                st.sidebar.warning(f"Livello di Caos massimo ridotto a {max_chaos_level} per audio > 60s.")
            params["chaos_level"] = st.sidebar.slider("Livello di Caos", 0.0, max_chaos_level, 0.5)


        if st.button("Avvia Elaborazione"):
            st.write(f"Elaborando con il metodo: **{decomposition_method}**")
            st.json(params) # Mostra i parametri scelti

            try:
                with st.spinner("Elaborazione audio in corso..."):
                    processed_audio = safe_process_audio(audio, sr, decomposition_method, params)
                st.success("✅ Processing completato!")

                # Assicurati che l'audio processato non sia vuoto
                if processed_audio is not None and len(processed_audio) > 0:
                    # Normalizza l'audio se necessario (per evitare clipping)
                    max_val = np.max(np.abs(processed_audio))
                    if max_val > 1.0:
                        processed_audio = processed_audio / max_val
                        st.info("Audio normalizzato per evitare clipping.")

                    # Esporta l'audio processato
                    st.subheader("Audio Elaborato")
                    # Crea un buffer di memoria per l'audio
                    audio_buffer = io.BytesIO()
                    sf.write(audio_buffer, processed_audio, sr, format='wav')
                    st.audio(audio_buffer.getvalue(), format='audio/wav')

                    # Pulsante per il download
                    st.download_button(
                        label="Scarica Audio Elaborato",
                        data=audio_buffer.getvalue(),
                        file_name=f"processed_audio_{decomposition_method}.wav",
                        mime="audio/wav"
                    )
                else:
                    st.warning("L'elaborazione ha prodotto un audio vuoto. Prova a regolare i parametri o a usare un altro metodo.")

            except Exception as e:
                st.error(f"❌ Crash durante {decomposition_method}: {str(e)}")
                st.write("Stack trace:", traceback.format_exc())

    except Exception as e:
        st.error(f"Errore durante il caricamento o l'inizializzazione del file audio: {str(e)}")
        st.write("Stack trace:", traceback.format_exc())

import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import random

# Le librerie seguenti sono nel tuo requirements.txt ma non direttamente usate in questo snippet.
# Le includo per completezza, ma se non le usi nel resto del tuo codice, potresti rimuoverle.
# import matplotlib.pyplot as plt
# import scipy
# import sklearn
# import ffmpeg_python
# import pydub
# import mido
# import pretty_midi

# --- Funzioni Helper Sicure ---

def safe_pitch_shift(audio, sr, n_steps):
    """
    Applica un pitch shift sicuro a un segmento audio.
    Gestisce errori e normalizza l'output.
    """
    try:
        if audio is None or audio.size == 0:
            return np.array([])
        
        # Assicurati che l'audio sia float per librosa.effects (necessario per pitch_shift)
        audio_float = audio.astype(np.float32)

        result = librosa.effects.pitch_shift(audio_float, sr=sr, n_steps=n_steps)
        # Rimuove valori NaN/Inf che potrebbero comparire occasionalmente
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Normalizzazione finale per evitare clipping e mantenere un volume ragionevole
        max_val = np.max(np.abs(result))
        if max_val > 0:
            result = result / max_val * 0.8 # Normalizza a 80% del massimo per sicurezza
        
        return result
    except Exception as e:
        # Avvisa l'utente se il pitch shift fallisce e restituisce l'audio originale
        st.warning(f"Pitch shift fallito: {e}. Restituisco l'audio originale del frammento.")
        return audio 

def safe_time_stretch(audio, rate):
    """
    Applica un time stretch sicuro a un segmento audio.
    Gestisce errori e normalizza l'output.
    """
    try:
        if audio is None or audio.size == 0:
            return np.array([])

        # Assicurati che l'audio sia float per librosa.effects
        audio_float = audio.astype(np.float32)
            
        result = librosa.effects.time_stretch(audio_float, rate=rate)
        # Rimuove valori NaN/Inf
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Normalizzazione finale
        max_val = np.max(np.abs(result))
        if max_val > 0:
            result = result / max_val * 0.8 # Normalizza a 80%
        
        return result
    except Exception as e:
        # Avvisa l'utente se il time stretch fallisce e restituisce l'audio originale
        st.warning(f"Time stretch fallito: {e}. Restituisco l'audio originale del frammento.")
        return audio 

---

## Funzione di Decostruzione Postmoderna

Questa funzione implementa la logica di "decostruzione ironica" descritta, con attenzione alla robustezza e alla gestione degli errori.

```python
def decostruzione_postmoderna(audio, sr, params):
    """
    Applica una decostruzione postmoderna e ironica del brano audio.
    Estrae frammenti importanti e casuali, li trasforma e li riorganizza.
    """
    irony_level = params.get('irony_level', 1.0)
    context_shift = params.get('context_shift', 1.2)
    fragment_size = params['fragment_size'] # Dimensione in secondi dei frammenti

    if audio is None or audio.size == 0:
        return np.array([]) # Ritorna un array vuoto se l'input è vuoto

    fragment_samples = int(fragment_size * sr)
    # Controlli sulla dimensione del frammento per evitare divisioni per zero o frammenti troppo grandi
    if fragment_samples <= 0:
        st.warning("Dimensione del frammento non valida (<= 0 campioni). Ritorno l'audio originale.")
        return audio
    if fragment_samples > len(audio):
        # Se il frammento è più lungo dell'audio stesso, consideriamo l'audio intero come un frammento
        st.info("Dimensione frammento maggiore dell'audio. Processo l'audio intero.")
        return process_single_fragment_postmodern(audio, sr, irony_level, context_shift)


    try:
        # --- 1. Calcolo Energia (RMS) per identificare frammenti "importanti" ---
        hop_length = 512
        energy = np.array([])
        
        if len(audio) >= hop_length: # L'audio deve essere abbastanza lungo per RMS
            try:
                energy = librosa.feature.rms(y=audio, hop_length=hop_length)[0]
            except Exception as e:
                st.warning(f"Calcolo RMS fallito: {e}. Fallback a energia semplice basata su finestre.")
                # Fallback: calcolo RMS manuale se librosa.feature.rms fallisce
                num_frames = len(audio) // hop_length
                if num_frames > 0:
                    energy = np.array([np.sqrt(np.mean(audio[i*hop_length:(i+1)*hop_length]**2)) 
                                        for i in range(num_frames)])
                else:
                    energy = np.array([0.0]) # Se l'audio è troppo corto
        else:
            energy = np.array([0.0]) # Se l'audio è troppo corto per hop_length

        important_frames = np.array([])
        if energy.size > 0 and np.max(energy) > 0: # Assicurati che ci sia energia non zero
            energy_threshold = np.percentile(energy, 70) # Soglia per identificare frammenti importanti
            important_frames = np.where(energy > energy_threshold)[0]

        important_times = np.array([])
        if important_frames.size > 0:
            important_times = librosa.frames_to_time(important_frames, sr=sr, hop_length=hop_length)

        fragments = []
        fragment_types = []

        # --- 2. Estrazione Frammenti ---
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
        # Garantisce almeno un frammento casuale, o il 50% dei frammenti importanti
        num_random = max(1, int(len(fragments) * 0.5)) if len(fragments) > 0 else 1
        for _ in range(num_random):
            if len(audio) < fragment_samples:
                # Se l'audio è più corto del frammento, prende tutto l'audio
                fragment = audio.copy()
            else:
                # Sceglie un punto di inizio casuale
                start = random.randint(0, len(audio) - fragment_samples)
                fragment = audio[start:start + fragment_samples]
            
            if fragment.size > 0:
                fragments.append(fragment)
                fragment_types.append('random')

        if len(fragments) == 0:
            st.warning("Nessun frammento valido è stato estratto. Ritorno l'audio originale.")
            return audio # Fallback se nessun frammento è stato creato

        # --- 3. Processa Frammenti con Trasformazioni e Effetti ---
        processed_fragments = []
        for i, (fragment, frag_type) in enumerate(zip(fragments, fragment_types)):
            if fragment.size == 0:
                continue

            current_fragment = fragment.copy()

            # Applicazione di trasformazioni "ironiche" sui frammenti importanti
            if frag_type == 'important' and random.random() < irony_level / 2.0:
                ironic_transforms = [
                    lambda x: x[::-1] if x.size > 0 else np.array([]), # Inversione temporale
                    lambda x: safe_pitch_shift(x, sr, -6) if x.size > 0 else np.array([]), # Pitch shift verso il basso
                    lambda x: safe_time_stretch(x, 0.5) if x.size > 0 else np.array([]), # Time stretch conservativo
                    lambda x: x * 0.2 if x.size > 0 else np.array([]), # Riduzione volume
                    # Ripetizione controllata: limita la parte da ripetere e il numero di ripetizioni per gestire la memoria
                    lambda x: np.tile(x[:min(len(x), int(sr * 0.5))] if len(x) > 0 else np.array([]), 3)
                ]
                
                transform = random.choice(ironic_transforms)
                try:
                    transformed_fragment = transform(current_fragment)
                    if transformed_fragment.size > 0:
                        current_fragment = transformed_fragment
                    else:
                        st.warning(f"Trasformazione ironica ha prodotto un frammento vuoto per tipo {frag_type}. Mantenuto frammento originale.")
                except Exception as e:
                    st.warning(f"Trasformazione ironica fallita per tipo {frag_type}: {e}. Mantenuto frammento originale.")

            # Applicazione di effetti di "spostamento contestuale"
            if current_fragment.size > 0 and random.random() < context_shift / 2.0:
                context_effects = [
                    lambda x: safe_pitch_shift(x, sr, random.uniform(-3, 3)) if x.size > 0 else np.array([]), # Pitch shift leggero
                    lambda x: x * np.linspace(0, 1, len(x)) if len(x) > 0 else np.array([]), # Fade in
                    lambda x: x * np.linspace(1, 0, len(x)) if len(x) > 0 else np.array([]), # Fade out
                    lambda x: x + np.random.normal(0, 0.02, len(x)) if len(x) > 0 else np.array([]), # Rumore leggero
                ]
                
                effect = random.choice(context_effects)
                try:
                    effected_fragment = effect(current_fragment)
                    if effected_fragment.size > 0:
                        current_fragment = effected_fragment
                    else:
                        st.warning(f"Effetto di spostamento contestuale ha prodotto un frammento vuoto per tipo {frag_type}. Mantenuto frammento corrente.")
                except Exception as e:
                    st.warning(f"Effetto di spostamento contestuale fallito per tipo {frag_type}: {e}. Mantenuto frammento corrente.")

            if current_fragment.size > 0:
                processed_fragments.append(current_fragment)

        if len(processed_fragments) == 0:
            st.warning("Nessun frammento è stato processato validamente. Ritorno l'audio originale.")
            return audio 

        # --- 4. Riordino dei Frammenti ---
        fragment_energies = []
        processed_fragments_filtered = []
        
        for frag in processed_fragments:
            if frag.size > 0:
                energy_val = np.mean(np.abs(frag)) # Energia media del frammento
                if not np.isnan(energy_val) and not np.isinf(energy_val):
                    fragment_energies.append(energy_val)
                    processed_fragments_filtered.append(frag)

        if len(processed_fragments_filtered) == 0:
            st.warning("Nessun frammento processato ha energia valida. Ritorno l'audio originale.")
            return audio 

        # Ordina frammenti per energia per la "decostruzione"
        if len(fragment_energies) > 0:
            sorted_indices = np.argsort(fragment_energies) # Indici dei frammenti ordinati per energia
            
            mid_point = len(sorted_indices) // 2
            low_energy = sorted_indices[:mid_point]
            high_energy = sorted_indices[mid_point:]
            
            result_order_indices = []
            max_pairs = min(len(low_energy), len(high_energy))
            
            # Alterna frammenti ad alta e bassa energia per un effetto "postmoderno"
            for i in range(max_pairs):
                if random.random() < 0.6: # Leggera preferenza per l'alta energia
                    result_order_indices.append(high_energy[i])
                result_order_indices.append(low_energy[i])
            
            # Aggiunge eventuali frammenti rimanenti
            if len(high_energy) > max_pairs:
                result_order_indices.extend(high_energy[max_pairs:])
            if len(low_energy) > max_pairs:
                result_order_indices.extend(low_energy[max_pairs:])
        else: 
            result_order_indices = list(range(len(processed_fragments_filtered))) # Ordine originale se non c'è energia

        # --- 5. Costruzione del Risultato Finale ---
        result_fragments_final = []
        for i in result_order_indices:
            if i < len(processed_fragments_filtered):
                result_fragments_final.append(processed_fragments_filtered[i])

        if len(result_fragments_final) > 0:
            result = np.concatenate(result_fragments_final) # Unisce tutti i frammenti
            
            result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0) # Pulizia finale
            
            # Normalizzazione finale per l'output complessivo
            max_val = np.max(np.abs(result))
            if max_val > 0:
                result = result / max_val * 0.8 # Normalizza a 80% per sicurezza anti-clipping
            else:
                st.warning("Il risultato finale della decostruzione è silenzioso. Ritorno l'audio originale.")
                return audio 
            
            return result
        else:
            st.warning("La costruzione finale non ha prodotto frammenti validi. Ritorno l'audio originale.")
            return audio # Fallback se nessun frammento è arrivato alla fine

    except Exception as e:
        # Cattura qualsiasi altro errore imprevisto nella funzione principale
        st.error(f"Errore critico in decostruzione_postmoderna: {e}. Ritorno l'audio originale.")
        return audio

# Funzione ausiliaria per il caso in cui fragment_size > len(audio)
def process_single_fragment_postmodern(audio, sr, irony_level, context_shift):
    """Processa l'intero audio come un singolo frammento se fragment_size è troppo grande."""
    current_fragment = audio.copy()

    # Applica trasformazioni ironiche (sempre considerandolo 'importante')
    if random.random() < irony_level / 2.0:
        ironic_transforms = [
            lambda x: x[::-1] if x.size > 0 else np.array([]),
            lambda x: safe_pitch_shift(x, sr, -6) if x.size > 0 else np.array([]),
            lambda x: safe_time_stretch(x, 0.5) if x.size > 0 else np.array([]),
            lambda x: x * 0.2 if x.size > 0 else np.array([]),
            lambda x: np.tile(x[:min(len(x), int(sr * 0.5))] if len(x) > 0 else np.array([]), 3)
        ]
        transform = random.choice(ironic_transforms)
        try:
            transformed_fragment = transform(current_fragment)
            if transformed_fragment.size > 0:
                current_fragment = transformed_fragment
        except Exception as e:
            st.warning(f"Trasformazione ironica su audio intero fallita: {e}.")

    # Applica effetti di "spostamento contestuale"
    if current_fragment.size > 0 and random.random() < context_shift / 2.0:
        context_effects = [
            lambda x: safe_pitch_shift(x, sr, random.uniform(-3, 3)) if x.size > 0 else np.array([]),
            lambda x: x * np.linspace(0, 1, len(x)) if len(x) > 0 else np.array([]),
            lambda x: x * np.linspace(1, 0, len(x)) if len(x) > 0 else np.array([]),
            lambda x: x + np.random.normal(0, 0.02, len(x)) if len(x) > 0 else np.array([]),
        ]
        effect = random.choice(context_effects)
        try:
            effected_fragment = effect(current_fragment)
            if effected_fragment.size > 0:
                current_fragment = effected_fragment
        except Exception as e:
            st.warning(f"Effetto contestuale su audio intero fallito: {e}.")

    current_fragment = np.nan_to_num(current_fragment, nan=0.0, posinf=0.0, neginf=0.0)
    max_val = np.max(np.abs(current_fragment))
    if max_val > 0:
        return current_fragment / max_val * 0.8
    return audio # Fallback se diventa silenzioso

---

## Funzione Generale di Decomposizione Audio

Questa funzione agisce come un "dispatcher" per i vari metodi di decomposizione, inclusa la `decostruzione_postmoderna`.

```python
def decompose_audio(audio, sr, method, params):
    """
    Applica il metodo di decomposizione audio scelto.
    Fornisce robustezza e fallback in caso di errori.
    """

    if audio is None or audio.size == 0:
        st.warning("Input audio vuoto o non valido per la decomposizione.")
        return np.array([])

    # Normalizzazione e pulizia iniziale dell'audio
    max_val_initial = np.max(np.abs(audio))
    if max_val_initial > 0:
        audio = audio / max_val_initial * 0.8 # Normalizza per sicurezza
    else:
        st.warning("Audio iniziale silenzioso. Nessuna decomposizione applicata.")
        return audio # Se l'audio è silenzioso, non decomponiamo

    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0) # Rimuovi NaN/Inf

    # Mappa dei metodi di decomposizione disponibili.
    # Puoi aggiungere qui le tue altre funzioni di decomposizione.
    methods_map = {
        'decostruzione_postmoderna': decostruzione_postmoderna,
        # 'cut_up_sonoro': cut_up_sonoro,
        # 'remix_destrutturato': remix_destrutturato,
        # 'musique_concrete': musique_concrete,
        # 'decomposizione_creativa': decomposizione_creativa,
        # 'random_chaos': random_chaos
    }

    decompose_func = methods_map.get(method)
    if decompose_func is None:
        st.error(f"Metodo di decomposizione '{method}' non riconosciuto. Ritorno l'audio originale.")
        return audio

    try:
        audio_backup = audio.copy() # Salva una copia dell'audio originale come backup
        
        # Applica il metodo di decomposizione scelto
        result = decompose_func(audio, sr, params)

        # Verifica la validità del risultato dopo la decomposizione
        if result is None:
            st.warning(f"Il metodo '{method}' ha restituito None. Utilizzo l'audio originale come fallback.")
            return audio_backup
            
        if not isinstance(result, np.ndarray):
            st.warning(f"Il metodo '{method}' ha restituito un tipo non valido. Utilizzo l'audio originale come fallback.")
            return audio_backup
            
        if result.size == 0:
            st.warning(f"Il metodo '{method}' ha prodotto un array audio vuoto. Utilizzo l'audio originale come fallback.")
            return audio_backup

        # Pulizia finale del risultato da NaN/Inf
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

        # Controlla se il risultato è silenzioso dopo la trasformazione
        if np.max(np.abs(result)) == 0:
            st.warning(f"Il metodo '{method}' ha prodotto audio silenzioso. Utilizzo l'audio originale come fallback.")
            return audio_backup

        # Normalizzazione finale del risultato per l'output
        max_result = np.max(np.abs(result))
        if max_result > 0:
            result = result / max_result * 0.8 # Normalizza a 80% del massimo
        else:
            st.warning(f"Normalizzazione finale fallita per il metodo '{method}'. Utilizzo l'audio originale come fallback.")
            return audio_backup

        return result

    except MemoryError:
        st.error(f"Memoria insufficiente per il metodo '{method}'. Prova con frammenti più piccoli o un file audio più corto.")
        return audio_backup # Fallback in caso di esaurimento memoria
        
    except Exception as e:
        # Cattura qualsiasi altro errore generale durante la decomposizione
        st.error(f"Errore generale nella decomposizione con il metodo '{method}': {e}")
        st.error(f"Dettagli dell'errore: {e.__class__.__name__}")
        return audio_backup # Fallback sicuro all'originale

---

## Struttura dell'Applicazione Streamlit

```python
st.set_page_config(layout="wide", page_title="Decostruttore Audio Postmoderno")

st.title("🎶 Decostruttore Audio Postmoderno")
st.markdown("Carica un file audio e applica una 'decostruzione' ironica e frammentata, ispirata all'arte postmoderna.")

# --- Interfaccia Utente ---
uploaded_file = st.file_uploader("Carica il tuo file audio (WAV, MP3, FLAC, OGG)", type=["wav", "mp3", "flac", "ogg"])

if uploaded_file is not None:
    try:
        st.info("Caricamento e analisi del file audio in corso...")
        # librosa.load può caricare vari formati. sr=None mantiene il sample rate originale.
        audio, sr = librosa.load(uploaded_file, sr=None, mono=True) 
        
        st.success(f"File caricato! Durata: {len(audio)/sr:.2f} secondi, Campioni: {len(audio)}, Sample Rate: {sr} Hz")
        
        st.subheader("Audio Originale")
        st.audio(audio, sample_rate=sr, format='audio/wav', start_time=0)

        st.subheader("Parametri di Decostruzione Postmoderna")

        # Slider per i parametri specifici di decostruzione_postmoderna
        irony_level = st.slider(
            "Livello di Ironia", 
            0.0, 1.0, 0.7, 0.1, 
            help="Controlla l'aggressività e la probabilità delle trasformazioni 'ironiche' (es. inversione, pitch shift drastico) sui frammenti importanti."
        )
        context_shift = st.slider(
            "Spostamento Contestuale", 
            0.0, 1.0, 0.5, 0.1, 
            help="Controlla la probabilità di applicare effetti sottili di 'spostamento' (es. fade, rumore, pitch shift leggero) ai frammenti."
        )
        fragment_size = st.slider(
            "Dimensione Frammento (secondi)", 
            0.1, 5.0, 0.5, 0.1, 
            help="La durata in secondi di ogni frammento estratto per la riorganizzazione. Valori più piccoli = più frammentazione."
        )

        params = {
            'irony_level': irony_level,
            'context_shift': context_shift,
            'fragment_size': fragment_size
        }

        # Bottone per avviare la decostruzione
        if st.button("🚀 Avvia la Decostruzione!"):
            with st.spinner("Applicando la decostruzione postmoderna... questo potrebbe richiedere un po' di tempo per file grandi."):
                # Chiama la funzione decompose_audio con il metodo specifico
                result_audio = decompose_audio(audio, sr, 'decostruzione_postmoderna', params)

            if result_audio.size > 0:
                st.subheader("Risultato Decostruito")
                st.audio(result_audio, sample_rate=sr, format='audio/wav', start_time=0)
                
                # Opzione per scaricare l'audio
                # sf.write richiede un oggetto simile a file, quindi usiamo io.BytesIO
                import io
                buffer = io.BytesIO()
                sf.write(buffer, result_audio, sr, format='WAV')
                buffer.seek(0) # Riporta il "cursore" all'inizio del buffer

                st.download_button(
                    label="💾 Scarica Audio Decostruito (WAV)",
                    data=buffer,
                    file_name="audio_decostruito.wav",
                    mime="audio/wav"
                )
            else:
                st.error("La decostruzione ha prodotto un audio vuoto o non valido. Prova a modificare i parametri o usa un altro file.")

    except Exception as e:
        st.error(f"Si è verificato un errore critico durante l'elaborazione del file audio: {e}")
        st.info("Assicurati che il file sia un formato audio valido (WAV, MP3, FLAC, OGG) e che l'**ambiente Python 3.9** sia configurato correttamente come spiegato in precedenza.")
        st.code(f"Dettagli tecnici dell'errore: {e.__class__.__name__}: {e}")

st.markdown("---")
st.markdown("Questo strumento è stato creato per esplorare la frammentazione, la riorganizzazione e l'ironia nel suono, ispirandosi ai concetti dell'arte postmoderna.")

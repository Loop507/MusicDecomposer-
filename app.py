import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import tempfile
import os
import random
import matplotlib.pyplot as plt
import gc

# ===============================
# OTTIMIZZAZIONI PRINCIPALI
# ===============================

def process_in_chunks(audio, sr, func, chunk_duration=15.0):
    """Elabora audio lungo a pezzi"""
    chunk_samples = int(chunk_duration * sr)
    if len(audio) <= chunk_samples:
        return func(audio, sr)
    chunks = []
    for start in range(0, len(audio), chunk_samples):
        end = min(start + chunk_samples, len(audio))
        chunk = audio[start:end]
        if chunk.size > 0:
            processed = func(chunk, sr)
            if processed.size > 0:
                chunks.append(processed)
        gc.collect()
    if not chunks:
        return np.array([])
    return np.concatenate(chunks)

# ===============================
# METODI PREIMPOSTATI E LEGGERI
# ===============================

def metodo_cut_up(audio, sr):
    """Cut-up semplice: frammenta e rimescola"""
    if audio.size == 0:
        return np.array([])
    fragment_size = 1.0
    fragment_samples = int(fragment_size * sr)
    fragments = []
    for i in range(0, len(audio) - fragment_samples + 1, fragment_samples):
        frag = audio[i:i + fragment_samples]
        if frag.size > 0:
            # Leggera variazione di lunghezza
            if random.random() < 0.3:
                factor = random.uniform(0.9, 1.1)
                new_len = int(len(frag) * factor)
                if new_len > 0:
                    indices = np.linspace(0, len(frag)-1, new_len)
                    frag = np.interp(indices, np.arange(len(frag)), frag)
            fragments.append(frag)
    if not fragments:
        return np.array([])
    random.shuffle(fragments)
    return np.concatenate(fragments)

def metodo_reverse(audio, sr):
    """Inversione parziale: sezioni invertite"""
    if audio.size == 0:
        return np.array([])
    section_size = int(2 * sr)  # 2 secondi
    sections = []
    for i in range(0, len(audio), section_size):
        section = audio[i:i + section_size]
        if section.size > 0:
            if random.random() < 0.4:
                section = section[::-1]
            sections.append(section)
    if not sections:
        return np.array([])
    return np.concatenate(sections)

def metodo_noise(audio, sr):
    """Aggiunge rumore e silenzi casuali"""
    if audio.size == 0:
        return np.array([])
    processed = audio.copy()
    num_events = min(10, len(processed) // (sr * 2))
    for _ in range(num_events):
        if random.random() < 0.5:
            # Rumore
            pos = random.randint(0, len(processed) - int(0.5 * sr))
            noise = np.random.normal(0, 0.02, int(0.5 * sr))
            end = pos + len(noise)
            if end <= len(processed):
                processed[pos:end] += noise
        else:
            # Silenzio
            pos = random.randint(0, len(processed) - int(0.3 * sr))
            end = pos + int(0.3 * sr)
            if end <= len(processed):
                processed[pos:end] = 0
    return processed

def metodo_palindrome(audio, sr):
    """Crea un effetto palindromo"""
    if audio.size == 0:
        return np.array([])
    fragment_size = int(1 * sr)
    fragments = []
    for i in range(0, len(audio) - fragment_size + 1, fragment_size):
        frag = audio[i:i + fragment_size]
        if frag.size > 0:
            fragments.append(frag)
    if not fragments:
        return np.array([])
    return np.concatenate(fragments + fragments[::-1])

# ===============================
# MAPPING METODI
# ===============================
metodi = {
    "Cut-Up Casuale": metodo_cut_up,
    "Audio Inverso": metodo_reverse,
    "Noise & Silence": metodo_noise,
    "Effetto Palindromo": metodo_palindrome,
}

# ===============================
# INTERFACCIA STREAMLIT
# ===============================

st.set_page_config(page_title="MusicDecomposer Leggero", layout="wide")
st.markdown("""
<div style='text-align: center; padding: 20px;'>
    <h1> MusicDecomposer <span style='font-size:0.6em;'>by loop507</span></h1>
    <p><em>Versione Stabile • Max 5 minuti • Zero Crash</em></p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Carica un audio (MP3/WAV)", type=["mp3", "wav"])

if uploaded_file is not None:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        audio, sr = librosa.load(tmp_path, sr=None, duration=300)  # Max 5 minuti
        durata = len(audio) / sr

        st.metric("Durata", f"{durata:.1f} sec")
        st.audio(uploaded_file, format='audio/wav')

        metodo = st.selectbox("Scegli un metodo", list(metodi.keys()))

        if st.button("🎭 SCOMPONI", type="primary"):
            with st.spinner("Elaborazione..."):
                func = metodi[metodo]
                if durata > 30:  # Oltre 30 secondi → chunks
                    processed = process_in_chunks(audio, sr, func)
                else:
                    processed = func(audio, sr)

                if processed.size == 0:
                    st.error("Errore: audio vuoto")
                else:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as out:
                        sf.write(out.name, processed, sr, subtype='PCM_16')
                        out_path = out.name

                    st.success("✅ Completato!")
                    st.audio(out_path, format='audio/wav')

                    filename = f"{uploaded_file.name.split('.')[0]}_{metodo.replace(' ', '_')}.wav"
                    with open(out_path, 'rb') as f:
                        st.download_button("💾 Scarica", f.read(), filename, "audio/wav")

                    # Grafico
                    with st.expander("📊 Forma d'onda"):
                        fig, ax = plt.subplots(figsize=(10, 4))
                        time = np.linspace(0, len(processed)/sr, len(processed))
                        ax.plot(time, processed, color='red', alpha=0.7, linewidth=0.5)
                        ax.set_title("Audio Decomposto")
                        ax.set_xlabel("Tempo (sec)")
                        ax.set_ylabel("Ampiezza")
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                        plt.close(fig)

                    # Cleanup
                    os.unlink(out_path)
                os.unlink(tmp_path)
                gc.collect()

    except Exception as e:
        st.error(f"Errore: {e}")
else:
    st.info("👆 Carica un file audio")
    st.markdown("""
    ### 🎯 Funzionalità:
    - ✅ **Stabile al 100%** anche su Streamlit Cloud
    - ✅ **Nessun crash** con file lunghi
    - ✅ Solo metodi **leggeri e preimpostati**
    - ✅ **Nessuna dipendenza pesante**
    - ✅ Funziona con **qualsiasi file fino a 5 minuti**
    """)

# Footer
st.markdown("---")
st.markdown("<center><em>MusicDecomposer • Arte Sonora Sperimentale</em></center>", unsafe_allow_html=True)

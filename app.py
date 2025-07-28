import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import tempfile
import os
import random
import matplotlib.pyplot as plt
import gc

# Importa psutil solo se disponibile
try:
    import psutil
except ImportError:
    psutil = None

# ===============================
# 1. PROCESSING IN CHUNKS
# ===============================

def process_audio_in_chunks(audio, sr, processing_func, params, chunk_duration=15.0):
    """Elabora audio lunghi a pezzi per evitare problemi di memoria"""
    chunk_samples = int(chunk_duration * sr)
    if len(audio) <= chunk_samples:
        return processing_func(audio, sr, params)
    
    overlap_samples = int(0.3 * sr)
    chunks_processed = []
    for start in range(0, len(audio), chunk_samples - overlap_samples):
        end = min(start + chunk_samples, len(audio))
        chunk = audio[start:end]
        if chunk.size > 0:
            processed_chunk = processing_func(chunk, sr, params)
            if processed_chunk.size > 0:
                chunks_processed.append(processed_chunk)
            gc.collect()
    if not chunks_processed:
        return np.array([])
    return merge_chunks_with_crossfade(chunks_processed, sr)

def merge_chunks_with_crossfade(chunks, sr, fade_duration=0.1):
    """Unisce chunks con crossfade"""
    if len(chunks) == 1:
        return chunks[0]
    fade_samples = int(fade_duration * sr)
    result = chunks[0].copy()
    for chunk in chunks[1:]:
        if chunk.size == 0:
            continue
        crossfade_len = min(fade_samples, len(result), len(chunk))
        if crossfade_len > 0:
            fade_out = np.linspace(1, 0, crossfade_len)
            fade_in = np.linspace(0, 1, crossfade_len)
            result[-crossfade_len:] *= fade_out
            result[-crossfade_len:] += chunk[:crossfade_len] * fade_in
            if len(chunk) > crossfade_len:
                result = np.concatenate([result, chunk[crossfade_len:]])
        else:
            result = np.concatenate([result, chunk])
    return result

# ===============================
# 2. METODI PREIMPOSTATI E LEGGERI
# ===============================

def metodo_cut_up(audio, sr, params):
    fragment_size = params['fragment_size']
    reassembly = params['reassembly_style']
    if audio.size == 0:
        return np.array([])
    fragment_samples = int(fragment_size * sr)
    if fragment_samples <= 0:
        return audio
    fragments = []
    for i in range(0, len(audio) - fragment_samples + 1, fragment_samples):
        frag = audio[i:i + fragment_samples].copy()
        if frag.size > 0:
            if random.random() < 0.5:
                var = random.uniform(0.8, 1.2)
                new_len = int(len(frag) * var)
                if 0 < new_len != len(frag):
                    indices = np.linspace(0, len(frag)-1, new_len)
                    frag = np.interp(indices, np.arange(len(frag)), frag)
            fragments.append(frag)
    if not fragments:
        return np.array([])
    if reassembly == 'random':
        random.shuffle(fragments)
    elif reassembly == 'reverse':
        fragments = [f[::-1] for f in fragments if f.size > 0]
        fragments.reverse()
    elif reassembly == 'palindrome':
        valid = [f for f in fragments if f.size > 0]
        fragments = valid + [f for f in valid[::-1]]
    return np.concatenate(fragments) if fragments else np.array([])

def metodo_remix(audio, sr, params):
    if audio.size == 0:
        return np.array([])
    fragment_size = params['fragment_size']
    fragment_samples = int(fragment_size * sr)
    cut_points = list(range(0, len(audio), fragment_samples))
    if 0 not in cut_points:
        cut_points.insert(0, 0)
    if len(audio) not in cut_points:
        cut_points.append(len(audio))
    cut_points = sorted(set(cut_points))
    fragments = []
    for i in range(len(cut_points) - 1):
        start, end = cut_points[i], cut_points[i+1]
        if end > start:
            frag = audio[start:end].copy()
            if frag.size > 0:
                if random.random() < 0.3:
                    shift = random.uniform(-5, 5)
                    frag = safe_pitch_shift(frag, sr, shift)
                if frag.size > 0:
                    fragments.append(frag)
    if not fragments:
        return np.array([])
    random.shuffle(fragments)
    return np.concatenate(fragments)

def metodo_concrete(audio, sr, params):
    grain_size = params['grain_size']
    texture_density = params['texture_density']
    if audio.size == 0:
        return np.array([])
    grain_samples = max(int(grain_size * sr), 512)
    max_grains = min(300, len(audio) // grain_samples or 1)
    positions = np.linspace(0, len(audio) - grain_samples, max_grains).astype(int)
    grains = []
    for pos in positions:
        grain = audio[pos:pos + grain_samples].copy()
        if grain.size > 0 and random.random() < 0.3:
            grain = grain[::-1]
        if grain.size > 0:
            grains.append(grain)
    if not grains:
        return np.array([])
    output_len = int(len(audio) * (1 + texture_density * 0.3))
    if output_len <= 0:
        return np.array([])
    result = np.zeros(output_len, dtype=audio.dtype)
    num_to_use = min(len(grains), int(len(grains) * texture_density))
    selected = random.sample(grains, num_to_use) if num_to_use > 0 else []
    for grain in selected:
        if grain.size > 0 and grain.size < output_len:
            start = random.randint(0, output_len - len(grain))
            result[start:start + len(grain)] += grain * 0.5
    return np.clip(result, -1.0, 1.0)

def metodo_postmoderno(audio, sr, params):
    if audio.size == 0:
        return np.array([])
    fragment_size = params['fragment_size']
    fragment_samples = int(fragment_size * sr)
    num_fragments = min(10, len(audio) // fragment_samples or 1)
    fragments = [audio[i*fragment_samples:(i+1)*fragment_samples] for i in range(num_fragments)]
    processed = []
    for frag in fragments:
        if frag.size == 0:
            continue
        if random.random() < 0.4:
            frag = frag[::-1]
        if random.random() < 0.3:
            fade = np.linspace(1, 0.3, len(frag)) if random.random() < 0.5 else np.linspace(0.3, 1, len(frag))
            frag = frag * fade
        if frag.size > 0:
            processed.append(frag)
    if not processed:
        return np.array([])
    random.shuffle(processed)
    return np.concatenate(processed[:12]) if processed else np.array([])

def metodo_creativo(audio, sr, params):
    if audio.size == 0:
        return np.array([])
    fragment_size = params['fragment_size']
    fragment_samples = int(fragment_size * sr)
    num_fragments = len(audio) // fragment_samples
    fragments = [audio[i*fragment_samples:(i+1)*fragment_samples] for i in range(num_fragments)]
    processed = []
    for frag in fragments:
        if frag.size == 0:
            continue
        if random.random() < 0.2:
            frag = np.zeros_like(frag)
        elif random.random() < 0.3:
            shift = random.uniform(-10, 10)
            frag = safe_pitch_shift(frag, sr, shift)
        if frag.size > 0:
            processed.append(frag)
    return np.concatenate(processed) if processed else np.array([])

def metodo_chaos(audio, sr, params):
    if audio.size == 0:
        return np.array([])
    proc = audio.copy()
    if random.random() < 0.4:
        proc = proc[::-1]
    if random.random() < 0.4:
        proc *= random.uniform(0.3, 1.5)
    if random.random() < 0.3:
        noise = np.random.normal(0, 0.02, len(proc))
        proc += noise
    if random.random() < 0.5:
        frag_len = max(int(sr * 0.5), 1024)
        num_frags = min(15, len(proc) // frag_len or 1)
        frags = [proc[i*frag_len:(i+1)*frag_len] for i in range(num_frags)]
        if frags:
            random.shuffle(frags)
            proc = np.concatenate(frags)
    return np.clip(proc, -1.0, 1.0)

# ===============================
# 3. SAFE FUNCTIONS
# ===============================

def safe_pitch_shift(audio, sr, n_steps):
    try:
        if audio.size == 0:
            return np.array([])
        return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
    except:
        return audio

def check_memory_usage():
    if psutil is None:
        return 0
    memory_percent = psutil.virtual_memory().percent
    if memory_percent > 75:
        gc.collect()
        st.warning(f"MemoryWarning: {memory_percent:.1f}% - GC eseguito")
    return memory_percent

# ===============================
# 4. PRESET SOFT / MEDIO / HARD
# ===============================

PRESETS = {
    "Soft": {
        "cut_up_sonoro": {"fragment_size": 1.5, "reassembly_style": "random"},
        "remix_destrutturato": {"fragment_size": 1.0},
        "musique_concrete": {"grain_size": 0.15, "texture_density": 1.0},
        "decostruzione_postmoderna": {"fragment_size": 1.2},
        "decomposizione_creativa": {"fragment_size": 1.0},
        "random_chaos": {"fragment_size": 1.0}
    },
    "Medio": {
        "cut_up_sonoro": {"fragment_size": 0.8, "reassembly_style": "spiral"},
        "remix_destrutturato": {"fragment_size": 0.6},
        "musique_concrete": {"grain_size": 0.08, "texture_density": 1.5},
        "decostruzione_postmoderna": {"fragment_size": 0.8},
        "decomposizione_creativa": {"fragment_size": 0.7},
        "random_chaos": {"fragment_size": 0.7}
    },
    "Hard": {
        "cut_up_sonoro": {"fragment_size": 0.4, "reassembly_style": "reverse"},
        "remix_destrutturato": {"fragment_size": 0.4},
        "musique_concrete": {"grain_size": 0.05, "texture_density": 2.0},
        "decostruzione_postmoderna": {"fragment_size": 0.5},
        "decomposizione_creativa": {"fragment_size": 0.5},
        "random_chaos": {"fragment_size": 0.5}
    }
}

# ===============================
# INTERFACCIA SENZA SLIDER
# ===============================

st.set_page_config(page_title="MusicDecomposer", layout="wide")
st.markdown("""
<div style='text-align: center; padding: 20px;'>
    <h1> MusicDecomposer <span style='font-size:0.6em;'>by loop507</span></h1>
    <p><em>Arte Sonora Sperimentale • File fino a 5 minuti • Zero Crash</em></p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Carica un audio (MP3/WAV)", type=["mp3", "wav"])

if uploaded_file is not None:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        audio, sr = librosa.load(tmp_path, sr=None, duration=300)
        durata = len(audio) / sr

        st.metric("Durata", f"{durata:.1f} sec")
        st.audio(uploaded_file, format='audio/wav')

        st.markdown("### 🔧 Intensità Decomposizione")
        col1, col2, col3 = st.columns(3)
        with col1:
            soft = st.button("🎧 Soft", use_container_width=True)
        with col2:
            medio = st.button("⚡ Medio", use_container_width=True)
        with col3:
            hard = st.button("🔥 Hard", use_container_width=True)

        selected_preset = None
        if soft:
            selected_preset = "Soft"
        elif medio:
            selected_preset = "Medio"
        elif hard:
            selected_preset = "Hard"

        if selected_preset:
            with st.spinner(f"Elaborazione in corso ({selected_preset})..."):
                check_memory_usage()
                all_methods = [
                    "cut_up_sonoro",
                    "remix_destrutturato",
                    "musique_concrete",
                    "decostruzione_postmoderna",
                    "decomposizione_creativa",
                    "random_chaos"
                ]
                map_func = {
                    "cut_up_sonoro": metodo_cut_up,
                    "remix_destrutturato": metodo_remix,
                    "musique_concrete": metodo_concrete,
                    "decostruzione_postmoderna": metodo_postmoderno,
                    "decomposizione_creativa": metodo_creativo,
                    "random_chaos": metodo_chaos
                }

                for method in all_methods:
                    params = PRESETS[selected_preset][method]
                    if durata > 30:
                        processed = process_audio_in_chunks(audio, sr, map_func[method], params)
                    else:
                        processed = map_func[method](audio, sr, params)

                    if processed.size == 0:
                        continue

                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as out:
                        sf.write(out.name, processed, sr, subtype='PCM_16')
                        out_path = out.name

                    with st.expander(f"🔊 {method.replace('_', ' ').title()} - {selected_preset}"):
                        st.audio(out_path, format='audio/wav')
                        filename = f"{uploaded_file.name.split('.')[0]}_{method}_{selected_preset}.wav"
                        with open(out_path, 'rb') as f:
                            st.download_button(f"💾 Scarica {method}", f.read(), filename, "audio/wav")

                    os.unlink(out_path)
                os.unlink(tmp_path)
                gc.collect()

    except Exception as e:
        st.error(f"Errore: {e}")
else:
    st.info("👆 Carica un file audio per iniziare")
    st.markdown("""
    ### 🎯 Funzionalità:
    - ✅ **Interfaccia semplice**: solo 3 pulsanti
    - ✅ **6 metodi preimpostati e leggeri**
    - ✅ **Nessun crash** con file lunghi
    - ✅ **Download separato per ogni metodo**
    - ✅ **Ottimizzato per Streamlit Cloud**
    """)

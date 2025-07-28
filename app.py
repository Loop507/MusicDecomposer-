import streamlit as st
import numpy as np
import librosa
import librosa.display
import soundfile as sf
import tempfile
import os
import random
from scipy import signal
# from sklearn.cluster import KMeans # Not used in current optimized versions
# from sklearn.preprocessing import StandardScaler # Not used in current optimized versions
import matplotlib.pyplot as plt
import io
import traceback
import gc

# Simplified placeholder for memory monitoring if psutil is not available
def check_memory_usage():
    """Placeholder for memory check without psutil."""
    gc.collect() # Always force garbage collection
    # st.info("Memory monitoring (psutil) is not enabled.") # Can uncomment for debugging
    return 0

# --- LIGHTWEIGHT AUDIO MANIPULATIONS (REPLACEMENTS FOR MEMORY-HEAVY LIBROSA.EFFECTS) ---

def lightweight_pitch_shift(audio, sr, n_steps, method='resample'):
    """
    Lightweight pitch shift using resampling.
    Much less memory-intensive than librosa.effects.pitch_shift for large files.
    """
    if audio.size == 0:
        return np.array([])
    
    # Calculate new sample rate based on pitch shift in semitones
    factor = 2**(n_steps / 12.0)
    new_sr = int(sr * factor)
    
    if new_sr <= 0: # Prevent division by zero or invalid sample rate
        return np.array([])

    try:
        # Use simple resampling for pitch shift effect
        # This will also change duration, but it's memory-efficient
        resampled_audio = librosa.resample(y=audio, orig_sr=sr, target_sr=new_sr)
        return np.nan_to_num(resampled_audio, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception as e:
        # st.warning(f"Lightweight pitch shift failed: {e}") # Debugging
        return np.array([]) # Return empty array on failure

def lightweight_time_stretch(audio, rate):
    """
    Lightweight time stretch using simple linear interpolation/decimation.
    Does not preserve pitch, but is very memory-efficient.
    """
    if audio.size == 0:
        return np.array([])
    
    if rate <= 0: # Avoid invalid rates
        return np.array([])

    new_length = int(len(audio) / rate)
    if new_length <= 0:
        return np.array([])

    try:
        indices = np.linspace(0, len(audio) - 1, new_length)
        stretched_audio = np.interp(indices, np.arange(len(audio)), audio)
        return np.nan_to_num(stretched_audio, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception as e:
        # st.warning(f"Lightweight time stretch failed: {e}") # Debugging
        return np.array([]) # Return empty array on failure


# --- OPTIMIZATIONS FOR ALL METHODS ---

# 1. PROCESSING IN CHUNKS - Elabora audio a pezzi invece che tutto insieme
def process_audio_in_chunks(audio, sr, processing_func, params, chunk_duration=7.0): # Default to 7s chunks
    """
    Elabora audio lunghi a pezzi per evitare problemi di memoria.
    Forza l'elaborazione a chunk per tutti i metodi per maggiore stabilità.
    """
    chunk_samples = int(chunk_duration * sr)
    
    if len(audio) <= chunk_samples * 1.5: # If audio is relatively short, process as one large chunk
        return processing_func(audio, sr, params)
    
    overlap_samples = int(0.1 * sr)  # Small overlap (0.1 sec)
    chunks_processed = []
    
    # Ensure chunk_samples is at least overlap_samples + a minimal processing unit
    min_processing_unit = 512 # A reasonable minimum for an audio chunk
    if chunk_samples <= overlap_samples + min_processing_unit:
        chunk_samples = overlap_samples + min_processing_unit

    for start in range(0, len(audio), chunk_samples - overlap_samples):
        end = min(start + chunk_samples, len(audio))
        chunk = audio[start:end]
        
        if chunk.size > 0:
            try:
                processed_chunk = processing_func(chunk, sr, params)
                if processed_chunk.size > 0:
                    chunks_processed.append(processed_chunk)
            except Exception as e:
                st.warning(f"Errore durante l'elaborazione di un chunk: {e}")
                # Optionally append original chunk if processing failed
                # chunks_processed.append(chunk) 
            
            # Force garbage collection after each chunk
            del chunk
            gc.collect()
    
    if chunks_processed:
        return merge_chunks_with_crossfade(chunks_processed, sr)
    else:
        return np.array([])

def merge_chunks_with_crossfade(chunks, sr, fade_duration=0.05): # Reduced fade duration
    """Unisce chunks con crossfade leggero per evitare click"""
    if not chunks:
        return np.array([])
    if len(chunks) == 1:
        return chunks[0]
    
    fade_samples = int(fade_duration * sr)
    if fade_samples <= 0: # Ensure positive fade samples
        fade_samples = 1 

    # Estimate total length to pre-allocate
    estimated_length = sum(len(c) for c in chunks) - (len(chunks) - 1) * fade_samples
    if estimated_length <= 0: # If estimate is bad, just concatenate for now
        estimated_length = sum(len(c) for c in chunks) 
    
    result = np.empty(estimated_length, dtype=chunks[0].dtype)
    current_pos = 0

    result[current_pos:current_pos + len(chunks[0])] = chunks[0]
    current_pos += len(chunks[0])

    for chunk in chunks[1:]:
        if chunk.size == 0:
            continue
            
        current_fade_len = min(fade_samples, result.size - current_pos, chunk.size)

        if current_fade_len > 0:
            # Overlap section of result and new chunk
            overlap_start_in_result = current_pos - current_fade_len
            
            fade_out = np.linspace(1, 0, current_fade_len)
            fade_in = np.linspace(0, 1, current_fade_len)
            
            # Apply fade to the end of the existing result
            if overlap_start_in_result >= 0:
                result[overlap_start_in_result:current_pos] *= fade_out
            
            # Blend the start of the new chunk
            blended_start = chunk[:current_fade_len] * fade_in
            
            # Add blended part to the end of result
            if overlap_start_in_result >= 0:
                result[overlap_start_in_result:current_pos] += blended_start
            else: # If result is too short for overlap, just prepend blended_start
                result = np.concatenate([blended_start, result]) # This case shouldn't happen often with pre-allocation logic

            # Append the rest of the new chunk
            if len(chunk) > current_fade_len:
                # Need to resize result if it's not large enough for the appended part
                required_size = current_pos + len(chunk) - current_fade_len
                if result.size < required_size:
                    new_result_array = np.empty(required_size, dtype=result.dtype)
                    new_result_array[:current_pos] = result[:current_pos]
                    result = new_result_array
                
                result[current_pos : current_pos + len(chunk) - current_fade_len] = chunk[current_fade_len:]
                current_pos += len(chunk) - current_fade_len
            
        else: # No overlap possible or too small
            # Need to resize result if it's not large enough for the appended part
            required_size = current_pos + len(chunk)
            if result.size < required_size:
                new_result_array = np.empty(required_size, dtype=result.dtype)
                new_result_array[:current_pos] = result[:current_pos]
                result = new_result_array

            result[current_pos:current_pos + len(chunk)] = chunk
            current_pos += len(chunk)
    
    # Trim to actual used length
    return result[:current_pos]

# --- OPTIMIZED DECOMPOSITION METHODS (INTERNAL, LIGHTWEIGHT VERSIONS) ---

def optimized_cut_up_sonoro_internal(audio, sr, params):
    """Internal lightweight version of cut_up_sonoro."""
    fragment_size = params['fragment_size']
    randomness = params.get('cut_randomness', 0.7)
    reassembly = params.get('reassembly_style', 'random')

    if audio.size == 0:
        return np.array([])

    fragment_samples = max(int(fragment_size * sr), 512) # Min 512 samples
    if fragment_samples <= 0:
        return audio # Fallback if fragment_samples becomes invalid

    fragment_indices = []
    for i in range(0, len(audio), fragment_samples):
        start = i
        end = min(i + fragment_samples, len(audio))
        if end > start:
            fragment_indices.append((start, end))
    
    processed_fragments = []
    for start, end in fragment_indices:
        fragment = audio[start:end].copy()
        
        if random.random() < randomness:
            # Lighter manipulation: simple volume change or reversal
            if random.random() < 0.5: # Reverse
                fragment = fragment[::-1]
            else: # Volume variation
                fragment *= random.uniform(0.5, 1.5)
        
        if fragment.size > 0:
            processed_fragments.append(fragment)
        del fragment # Free memory
    
    if not processed_fragments:
        return np.array([])
    
    if reassembly == 'random':
        random.shuffle(processed_fragments)
    elif reassembly == 'reverse':
        processed_fragments = [frag[::-1] for frag in processed_fragments if frag.size > 0]
        processed_fragments.reverse()
    elif reassembly == 'palindrome':
        valid_fragments = [frag for frag in processed_fragments if frag.size > 0]
        processed_fragments = valid_fragments + [frag for frag in valid_fragments[::-1]]
    elif reassembly == 'spiral':
        new_fragments = []
        start_idx, end_idx = 0, len(processed_fragments) - 1
        while start_idx <= end_idx:
            if start_idx < len(processed_fragments) and processed_fragments[start_idx].size > 0:
                new_fragments.append(processed_fragments[start_idx])
                start_idx += 1
            if start_idx <= end_idx and end_idx < len(processed_fragments) and processed_fragments[end_idx].size > 0 and start_idx <= end_idx:
                new_fragments.append(processed_fragments[end_idx])
                end_idx -= 1
            # Prevent infinite loops if fragments are empty
            if start_idx <= end_idx and (processed_fragments[start_idx].size == 0 or processed_fragments[end_idx].size == 0):
                if processed_fragments[start_idx].size == 0: start_idx += 1
                if start_idx <= end_idx and processed_fragments[end_idx].size == 0: end_idx -= 1
        processed_fragments = new_fragments
        
    total_length = sum(len(frag) for frag in processed_fragments)
    if total_length == 0:
        return np.array([])
    
    result = np.empty(total_length, dtype=audio.dtype)
    current_pos = 0
    for frag in processed_fragments:
        if frag.size > 0:
            result[current_pos:current_pos + len(frag)] = frag
            current_pos += len(frag)
    
    del processed_fragments # Free memory
    gc.collect()
    return result[:current_pos]


def optimized_remix_destrutturato_internal(audio, sr, params):
    """Internal lightweight version of remix_destrutturato."""
    fragment_size = params['fragment_size']
    beat_preservation = params.get('beat_preservation', 0.4)
    melody_fragmentation = params.get('melody_fragmentation', 1.5)

    if audio.size == 0:
        return np.array([])

    # Simplified "beat tracking" - use regular intervals or basic energy peaks
    # Avoiding librosa.beat.beat_track for memory efficiency and speed
    cut_points = [0]
    num_intervals = int(len(audio) / (sr * fragment_size))
    for i in range(1, num_intervals + 1):
        cut_points.append(min(len(audio), int(i * sr * fragment_size * random.uniform(0.8, 1.2))))
    
    # Add random cut points for more "fragmentation"
    num_extra_cuts = min(50, int(len(audio) / sr / 2)) # Max 50 extra cuts
    for _ in range(num_extra_cuts):
        if len(audio) > 0:
            cut_points.append(random.randint(0, len(audio)))
    
    cut_points = sorted(list(set(cut_points)))
    if cut_points[-1] != len(audio):
        cut_points.append(len(audio))

    fragments = []
    for i in range(len(cut_points) - 1):
        start = cut_points[i]
        end = cut_points[i + 1]
        if end <= start: continue
        fragment = audio[start:end].copy()

        if fragment.size > 0:
            # Lighter manipulations instead of pitch_shift/time_stretch
            if random.random() < melody_fragmentation / 5.0: # Reduced probability
                if random.random() < 0.5:
                    fragment = lightweight_pitch_shift(fragment, sr, random.uniform(-5, 5)) # Smaller shifts
                else:
                    fragment = lightweight_time_stretch(fragment, random.uniform(0.8, 1.2)) # Smaller stretch
        
        if fragment.size > 0:
            fragments.append(fragment)
        del fragment
    
    if not fragments:
        return np.array([])

    if beat_preservation > 0.3 and len(fragments) > 1:
        # Simple attempt to preserve some order
        num_to_preserve = int(len(fragments) * beat_preservation)
        shuffled_parts = random.sample(fragments, k=max(0, len(fragments) - num_to_preserve))
        
        # This is a highly simplified reassembly to prevent complex logic/memory
        final_fragments = []
        for i in range(len(fragments)):
            if i < num_to_preserve: # Keep first few fragments in order (simplified "preservation")
                final_fragments.append(fragments[i])
            else:
                if shuffled_parts:
                    final_fragments.append(shuffled_parts.pop(0))
                else:
                    final_fragments.append(fragments[i]) # Fallback
        fragments = final_fragments
    else:
        random.shuffle(fragments)
    
    total_length = sum(len(f) for f in fragments)
    if total_length == 0:
        return np.array([])

    result = np.empty(total_length, dtype=audio.dtype)
    current_pos = 0
    for frag in fragments:
        if frag.size > 0:
            result[current_pos:current_pos + len(frag)] = frag
            current_pos += len(frag)
    
    del fragments
    gc.collect()
    return result[:current_pos]


def optimized_musique_concrete_internal(audio, sr, params):
    """Internal lightweight version of musique_concrete."""
    grain_size = max(params.get('grain_size', 0.1), 0.05) # Min grain size 0.05s
    texture_density = min(params.get('texture_density', 1.0), 1.5) # Max density 1.5
    chaos_level = min(params['chaos_level'], 2.0) # Max chaos 2.0

    if audio.size == 0:
        return np.array([])

    grain_samples = max(int(grain_size * sr), 256) # Min 256 samples
    if grain_samples <= 0:
        return np.array([])

    grains = []
    # Reduced number of grains generated from source
    max_source_grains = min(500, len(audio) // grain_samples) # Limit source grains
    if max_source_grains <= 0 and len(audio) > 0:
        grains.append(audio.copy()) # Treat whole audio as one grain if very short
    else:
        for i in range(0, len(audio) - grain_samples + 1, int(grain_samples * 0.5)): # Overlap 50%
            if len(grains) >= max_source_grains: break # Stop generating if max reached
            grain = audio[i:i + grain_samples].copy()
            
            if grain.size > 0:
                try:
                    # Apply a light window to avoid clicks
                    window = signal.windows.hann(len(grain))
                    grain = grain * window
                except Exception:
                    pass # Ignore if windowing fails

                # Lighter manipulations
                if random.random() < chaos_level * 0.15: # Lower probability
                    grain = grain[::-1] # Only reversal

                if random.random() < chaos_level * 0.15:
                    grain *= random.uniform(0.3, 1.2) # Simple amplitude change
                
                if grain.size > 0:
                    grains.append(grain)
            del grain
    
    if not grains:
        return np.array([])
    
    # Limit output length and number of grains to use
    max_output_length = int(len(audio) * (1 + texture_density * 0.5))
    if max_output_length <= 0:
        return np.array([])

    num_grains_to_use = min(len(grains), int(len(grains) * texture_density * 0.8)) # Use fewer grains
    if num_grains_to_use <= 0:
        return np.array([])

    selected_grains = random.sample(grains, num_grains_to_use)
    
    result = np.zeros(max_output_length, dtype=audio.dtype)
    
    for grain in selected_grains:
        if grain.size == 0:
            continue
        
        # Ensure grain fits within result buffer
        if grain.size > max_output_length:
            grain = grain[:max_output_length]
        
        if grain.size > 0 and max_output_length - grain.size >= 0:
            start_pos = random.randint(0, max_output_length - grain.size)
            end_pos = start_pos + grain.size
            result[start_pos:end_pos] += grain * random.uniform(0.1, 0.6) # Reduced amplitude for blending
        del grain
    
    # Normalize result
    if result.size > 0:
        max_val = np.max(np.abs(result))
        if max_val > 0:
            result = result / max_val * 0.8 # Normalize to 80%
        else:
            result = np.array([])
    
    del selected_grains
    del grains
    gc.collect()
    return result


def optimized_decostruzione_postmoderna_internal(audio, sr, params):
    """Internal lightweight version of decostruzione_postmoderna."""
    irony_level = min(params.get('irony_level', 0.5), 0.8) # Capped irony
    context_shift = min(params.get('context_shift', 0.6), 0.8) # Capped context shift
    fragment_size = params['fragment_size']

    if audio.size == 0:
        return np.array([])

    fragment_samples = max(int(fragment_size * sr), 1024) # Min 1024 samples
    if fragment_samples <= 0:
        return audio

    # Simplified "important" fragment detection (using fixed intervals and simple RMS)
    fragments_with_type = []
    
    num_fixed_fragments = min(50, len(audio) // fragment_samples)
    for i in range(num_fixed_fragments):
        start = i * fragment_samples
        end = min(start + fragment_samples, len(audio))
        if end > start:
            frag = audio[start:end].copy()
            if frag.size > 0:
                fragments_with_type.append((frag, 'fixed'))
    
    # Add a few "important" fragments (just pick a few random high-amplitude ones)
    num_important_fragments = min(5, len(fragments_with_type) // 2)
    if num_important_fragments > 0:
        for _ in range(num_important_fragments):
            if fragments_with_type:
                idx = random.randint(0, len(fragments_with_type) - 1)
                fragments_with_type[idx] = (fragments_with_type[idx][0], 'important') # Mark as important

    if not fragments_with_type:
        return audio # Return original if no fragments could be generated

    processed_fragments = []
    
    for fragment, frag_type in fragments_with_type:
        if fragment.size == 0:
            continue
        
        processed_frag = fragment.copy()

        # Irony transformations (very light)
        if frag_type == 'important' and random.random() < irony_level * 0.3: # Lower probability
            transform_choice = random.choice(['reverse', 'volume_down'])
            if transform_choice == 'reverse':
                processed_frag = processed_frag[::-1]
            elif transform_choice == 'volume_down':
                processed_frag *= 0.2 # Very low volume

        # Context shift (very light)
        if processed_frag.size > 0 and random.random() < context_shift * 0.3: # Lower probability
            effect_choice = random.choice(['fade_in', 'light_noise'])
            if effect_choice == 'fade_in':
                fade = np.linspace(0.1, 1, len(processed_frag))
                processed_frag = processed_frag * fade
            elif effect_choice == 'light_noise':
                noise_level = 0.005 # Extremely light noise
                noise = np.random.normal(0, noise_level, len(processed_frag))
                processed_frag = processed_frag + noise
        
        if processed_frag.size > 0:
            processed_fragments.append(processed_frag)
        del fragment, processed_frag
    
    if not processed_fragments:
        return np.array([])

    random.shuffle(processed_fragments) # Random reassembly
    
    # Limit number of final fragments to prevent excessive length
    final_fragments_count = min(30, len(processed_fragments))
    final_fragments = processed_fragments[:final_fragments_count]

    total_length = sum(len(f) for f in final_fragments)
    if total_length == 0:
        return np.array([])
    
    result = np.empty(total_length, dtype=audio.dtype)
    current_pos = 0
    for frag in final_fragments:
        if frag.size > 0:
            result[current_pos:current_pos + len(frag)] = frag
            current_pos += len(frag)
    
    if result.size > 0:
        result[np.isnan(result)] = 0
        result[np.isinf(result)] = 0
        max_val = np.max(np.abs(result))
        if max_val > 1.0:
            result = result / max_val * 0.95
    
    del fragments_with_type, processed_fragments, final_fragments
    gc.collect()
    return result[:current_pos]


def optimized_decomposizione_creativa_internal(audio, sr, params):
    """Internal lightweight version of decomposizione_creativa."""
    discontinuity = min(params.get('discontinuity', 1.0), 1.5) # Capped
    emotional_shift = min(params.get('emotional_shift', 0.8), 1.5) # Capped
    fragment_size = params['fragment_size']
    chaos_level = min(params['chaos_level'], 2.0) # Capped

    if audio.size == 0:
        return np.array([])

    # Simplified onset detection: regular intervals plus a few random ones
    cut_points = [0]
    fixed_interval_samples = int(fragment_size * sr * random.uniform(0.8, 1.2))
    if fixed_interval_samples <= 0:
        fixed_interval_samples = int(sr * 0.5) # Min 0.5s if calculation fails

    for i in range(0, len(audio), fixed_interval_samples):
        cut_points.append(i)
    
    num_random_onsets = min(20, int(len(audio) / sr / 5)) # Max 20 random points
    for _ in range(num_random_onsets):
        if len(audio) > 0:
            cut_points.append(random.randint(0, len(audio) - 1))
    
    cut_points = sorted(list(set(cut_points)))
    if cut_points[-1] != len(audio):
        cut_points.append(len(audio))

    processed_fragments = []
    for i in range(len(cut_points) - 1):
        start_sample = cut_points[i]
        end_sample = cut_points[i+1]
        
        if end_sample <= start_sample:
            continue

        fragment = audio[start_sample:end_sample].copy()
        if fragment.size == 0:
            continue

        # Apply discontinuity
        if random.random() < discontinuity * 0.1:
            if random.random() < 0.5: fragment = np.zeros_like(fragment)
            else: continue # Skip fragment

        # Apply emotional shifts (lightweight replacements)
        if random.random() < emotional_shift * 0.3:
            if random.random() < 0.5: # Simple resampling for pitch-like effect
                fragment = lightweight_pitch_shift(fragment, sr, random.uniform(-6, 6)) # Smaller range
            else: # Simple interpolation for time-like effect
                fragment = lightweight_time_stretch(fragment, random.uniform(0.7, 1.3)) # Smaller range
        
        # Add a touch of chaos (reversal/volume)
        if fragment.size > 0 and random.random() < chaos_level * 0.1:
            if random.random() < 0.5:
                fragment = fragment[::-1]
            else:
                fragment *= random.uniform(0.5, 1.5)

        if fragment.size > 0:
            processed_fragments.append(fragment)
        del fragment
    
    if not processed_fragments:
        return np.array([])

    total_length = sum(len(f) for f in processed_fragments)
    if total_length == 0:
        return np.array([])
    
    result = np.empty(total_length, dtype=audio.dtype)
    current_pos = 0
    for frag in processed_fragments:
        if frag.size > 0:
            result[current_pos:current_pos + len(frag)] = frag
            current_pos += len(frag)

    if result.size > 0:
        max_val = np.max(np.abs(result))
        if max_val > 0:
            result = result / max_val * 0.95
    
    del processed_fragments
    gc.collect()
    return result[:current_pos]


def ultra_optimized_random_chaos_internal(audio, sr, params):
    """Internal lightweight version of random_chaos - NO HEAVY OPERATIONS."""
    chaos_level = min(params['chaos_level'], 2.0) # Capped chaos

    if audio.size == 0:
        return np.array([])
    
    processed_audio = audio.copy()
    current_sr = sr

    # 1. Reverse sections (VELOCE)
    if random.random() < 0.3 * chaos_level and len(processed_audio) > int(current_sr * 0.5):
        section_length = min(int(current_sr * 2), len(processed_audio) // 4) # Max 2 sec
        if section_length > 0 and len(processed_audio) >= section_length:
            start = random.randint(0, len(processed_audio) - section_length)
            processed_audio[start:start + section_length] = processed_audio[start:start + section_length][::-1]
    
    # 2. Volume variations (VELOCE)
    if random.random() < 0.4 * chaos_level and processed_audio.size > 0:
        num_sections = min(10, max(1, int(chaos_level * 5)))
        section_length = len(processed_audio) // num_sections
        for i in range(num_sections):
            start = i * section_length
            end = min(start + section_length, len(processed_audio))
            if end > start:
                processed_audio[start:end] *= random.uniform(0.3, 1.5)
            
    # 3. Simple noise (VELOCE)
    if random.random() < 0.2 * chaos_level and processed_audio.size > 0:
        noise_level = min(0.05, chaos_level * 0.02)
        noise = np.random.normal(0, noise_level, len(processed_audio))
        processed_audio += noise
    
    # 4. Fragment shuffling (MODERATO)
    if random.random() < 0.5 * chaos_level and processed_audio.size > 0:
        fragment_length = max(int(current_sr * 0.5), 512) # Min 0.5 sec, or 512 samples
        num_fragments = min(20, len(processed_audio) // fragment_length)
        if num_fragments <= 0 and processed_audio.size > 0: # If audio is very short, make it one fragment
            num_fragments = 1
            fragment_length = processed_audio.size

        if num_fragments > 0:
            fragments = []
            for i in range(num_fragments):
                start = i * fragment_length
                end = min(start + fragment_length, len(processed_audio))
                if end > start:
                    fragments.append(processed_audio[start:end].copy())
            
            if fragments:
                random.shuffle(fragments)
                total_length = sum(len(f) for f in fragments)
                if total_length > 0:
                    new_audio = np.empty(total_length, dtype=audio.dtype)
                    pos = 0
                    for frag in fragments:
                        if frag.size > 0:
                            new_audio[pos:pos + len(frag)] = frag
                            pos += len(frag)
                    processed_audio = new_audio[:pos]
                else:
                    processed_audio = np.array([])
            else:
                processed_audio = np.array([])
        
    # Normalizza
    if processed_audio.size > 0:
        max_val = np.max(np.abs(processed_audio))
        if max_val > 0:
            processed_audio = processed_audio / max_val * 0.95
        else:
            processed_audio = np.array([])
        
    return processed_audio

# --- MAIN WRAPPER - ALWAYS USES CHUNKS ---
def safe_process_audio(audio, sr, method, params):
    """
    Wrapper sicuro che usa processing in chunks per tutti i metodi.
    """
    # The chunk_duration is set in process_audio_in_chunks and applies universally.
    # No need for conditional chunking here.
    
    if method == "cut_up_sonoro":
        return process_audio_in_chunks(audio, sr, optimized_cut_up_sonoro_internal, params)
    
    elif method == "musique_concrete":
        return process_audio_in_chunks(audio, sr, optimized_musique_concrete_internal, params)
    
    elif method == "random_chaos":
        return process_audio_in_chunks(audio, sr, ultra_optimized_random_chaos_internal, params)
    
    elif method == "remix_destrutturato":
        return process_audio_in_chunks(audio, sr, optimized_remix_destrutturato_internal, params)
    
    elif method == "decostruzione_postmoderna":
        return process_audio_in_chunks(audio, sr, optimized_decostruzione_postmoderna_internal, params)
    
    elif method == "decomposizione_creativa":
        return process_audio_in_chunks(audio, sr, optimized_decomposizione_creativa_internal, params)
    
    else:
        st.error(f"Metodo sconosciuto: {method}")
        return np.array([])

# --- STREAMLIT UI ---

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
            "cut_up_sonoro": "Cut-up Sonoro",
            "remix_destrutturato": "Remix Destrutturato",
            "musique_concrete": "Musique Concrète",
            "decostruzione_postmoderna": "Decostruzione Postmoderna",
            "decomposizione_creativa": "Decomposizione Creativa",
            "random_chaos": "Random Chaos"
        }[x]
    )

    st.markdown("---")

    # Parametri generali (slightly adjusted ranges for better defaults/stability)
    fragment_size = st.slider("Dimensione Frammenti (sec)", 0.1, 3.0, 1.0, 0.1) # Max 3s for robustness
    chaos_level = st.slider("Livello di Chaos", 0.1, 2.0, 1.0, 0.1) # Max 2.0
    structure_preservation = st.slider("Conservazione Struttura", 0.0, 1.0, 0.3, 0.1)

    st.markdown("---")

    # Parametri specifici per metodo
    cut_randomness = None
    reassembly_style = None
    beat_preservation = None
    melody_fragmentation = None
    grain_size = None
    texture_density = None
    irony_level = None
    context_shift = None
    discontinuity = None
    emotional_shift = None

    if decomposition_method == "cut_up_sonoro":
        st.subheader("Cut-up Parameters")
        cut_randomness = st.slider("Casualità Tagli", 0.1, 1.0, 0.7, 0.1)
        reassembly_style = st.selectbox("Stile Riassemblaggio",
                                         ["random", "reverse", "palindrome", "spiral"])

    elif decomposition_method == "remix_destrutturato":
        st.subheader("Remix Parameters")
        beat_preservation = st.slider("Conserva Ritmo", 0.0, 1.0, 0.4, 0.1)
        melody_fragmentation = st.slider("Frammentazione Melodia", 0.1, 2.0, 1.0, 0.1) # Reduced max

    elif decomposition_method == "musique_concrete":
        st.subheader("Concrete Parameters")
        grain_size = st.slider("Dimensione Grani", 0.01, 0.3, 0.1, 0.01) # Reduced max
        texture_density = st.slider("Densità Texture", 0.1, 2.0, 1.0, 0.1) # Reduced max

    elif decomposition_method == "decostruzione_postmoderna":
        st.subheader("Postmodern Parameters")
        irony_level = st.slider("Livello Ironia", 0.1, 0.8, 0.5, 0.1) # Reduced max
        context_shift = st.slider("Shift di Contesto", 0.1, 0.8, 0.6, 0.1) # Reduced max
        st.info("🔧 Parametri ottimizzati per prestazioni migliori")

    elif decomposition_method == "decomposizione_creativa":
        st.subheader("Creative Parameters")
        discontinuity = st.slider("Discontinuità", 0.1, 1.5, 1.0, 0.1) # Reduced max
        emotional_shift = st.slider("Shift Emotivo", 0.1, 1.5, 0.8, 0.1) # Reduced max

# Upload file
uploaded_file = st.file_uploader(
    "Carica il tuo brano da decomporre",
    type=["mp3", "wav", "m4a", "flac", "ogg"],
    help="Supporta MP3, WAV, M4A, FLAC, OGG"
)

# Logica principale per processare l'audio
if uploaded_file is not None:
    # Ensure a temporary directory for safe temp file handling
    temp_dir = tempfile.mkdtemp()
    tmp_file_path = ""
    processed_tmp_path = ""

    try:
        # Load the file safely
        with open(os.path.join(temp_dir, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        tmp_file_path = os.path.join(temp_dir, uploaded_file.name)

        audio, sr = librosa.load(tmp_file_path, sr=None, mono=True) # Ensure mono loading
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Durata", f"{len(audio)/sr:.2f} sec")
        with col2:
            st.metric("Sample Rate", f"{sr} Hz")
        with col3:
            st.metric("Canali", "Mono")

        st.subheader("Audio Originale")
        st.audio(uploaded_file, format='audio/wav')

        if st.button("🎭 SCOMPONI E RICOMPONI", type="primary", use_container_width=True):
            with st.spinner(f"Applicando {decomposition_method}..."):
                check_memory_usage()

                params = {
                    'fragment_size': fragment_size,
                    'chaos_level': chaos_level,
                    'structure_preservation': structure_preservation
                }

                if decomposition_method == "cut_up_sonoro":
                    params.update({'cut_randomness': cut_randomness, 'reassembly_style': reassembly_style})
                elif decomposition_method == "remix_destrutturato":
                    params.update({'beat_preservation': beat_preservation, 'melody_fragmentation': melody_fragmentation})
                elif decomposition_method == "musique_concrete":
                    params.update({'grain_size': grain_size, 'texture_density': texture_density})
                elif decomposition_method == "decostruzione_postmoderna":
                    params.update({'irony_level': irony_level, 'context_shift': context_shift})
                elif decomposition_method == "decomposizione_creativa":
                    params.update({'discontinuity': discontinuity, 'emotional_shift': emotional_shift})
                
                processed_audio = safe_process_audio(audio, sr, decomposition_method, params)

                if processed_audio.size == 0:
                    st.error("❌ Elaborazione fallita - audio risultante vuoto. Prova a modificare i parametri o a usare un file diverso.")
                else:
                    processed_tmp_path = os.path.join(temp_dir, "processed_audio.wav")
                    sf.write(processed_tmp_path, processed_audio, sr)

                    st.success("✅ Decomposizione completata!")
                    
                    col1, col2, col3 = st.columns(3)
                    new_duration = len(processed_audio)/sr
                    original_duration = len(audio)/sr
                    with col1:
                        st.metric("Nuova Durata", f"{new_duration:.2f} sec", f"{(new_duration - original_duration):.2f} sec")
                    with col2:
                        original_rms = np.sqrt(np.mean(audio**2)) if audio.size > 0 else 0
                        processed_rms = np.sqrt(np.mean(processed_audio**2)) if processed_audio.size > 0 else 0
                        st.metric("RMS Energy", f"{processed_rms:.4f}", f"{(processed_rms - original_rms):.4f}")
                    with col3:
                        spectral_diff = 0.0
                        min_len = min(len(processed_audio), len(audio))
                        if min_len > 0:
                            spec_orig = np.abs(np.fft.fft(audio[:min_len]))
                            spec_proc = np.abs(np.fft.fft(processed_audio[:min_len]))
                            spectral_diff = np.mean(np.abs(spec_proc - spec_orig))
                        st.metric("Variazione Spettrale", f"{spectral_diff:.2e}")

                    st.subheader("Audio Decomposto e Ricomposto")
                    with open(processed_tmp_path, 'rb') as audio_file:
                        audio_bytes = audio_file.read()
                        st.audio(audio_bytes, format='audio/wav')

                    method_names = {
                        "cut_up_sonoro": "CutUp", "remix_destrutturato": "RemixDestrutturato", 
                        "musique_concrete": "MusiqueConcrete", "decostruzione_postmoderna": "DecostruzionePostmoderna",
                        "decomposizione_creativa": "DecomposizioneCreativa", "random_chaos": "RandomChaos"
                    }
                    filename = f"{uploaded_file.name.split('.')[0]}_{method_names[decomposition_method]}.wav"
                    
                    st.download_button(
                        label="💾 Scarica Audio Decomposto",
                        data=audio_bytes,
                        file_name=filename,
                        mime="audio/wav",
                        use_container_width=True
                    )

                    technique_descriptions = {
                        "cut_up_sonoro": """
                        Il metodo **"Cut-up Sonoro"** si ispira a una tecnica letteraria dove il testo viene frammentato e riassemblato. Il brano viene diviso in sezioni, che vengono poi **tagliate e riassemblate in un ordine casuale o predefinito** (come inversione o palindromo). Questo crea un effetto di collage sonoro, dove il significato originale è destrutturato per rivelare nuove connessioni e pattern imprevedibili. La musica diventa una forma di testo decostruito.
                        """,
                        "remix_destrutturato": """
                        Il **"Remix Destrutturato"** mira a mantenere alcuni elementi riconoscibili del brano originale (come battiti o frammenti melodici), ma li **ricontestualizza in un nuovo arrangiamento**. Vengono applicate manipolazioni leggere come semplici variazioni di pitch o tempo ai frammenti, che poi vengono riorganizzati per creare un'esperienza d'ascolto che è sia familiare che sorprendentemente nuova, quasi una reinterpretazione.
                        """,
                        "musique_concrete": """
                        La **"Musique Concrète"** si basa sui principi di manipolazione sonora. Questo metodo si concentra sulla **manipolazione di "grani" sonori** (piccolissimi frammenti dell'audio) attraverso tecniche come la sintesi granulare, l'inversione e le leggere variazioni di ampiezza. Il risultato è una texture sonora astratta, spesso non riconoscibile come musica nel senso tradizionale, che esplora le proprietà intrinseche del suono.
                        """,
                        "decostruzione_postmoderna": """
                        La **"Decostruzione Postmoderna"** applica un approccio critico al brano, **decostruendone il significato musicale originale** attraverso l'uso di ironia e spostamenti di contesto. Frammenti "importanti" vengono manipolati in modi inaspettati (es. volume ridotto, inversione), e vengono introdotti elementi di rottura o rumore. L'obiettivo è provocare una riflessione critica sull'opera e sulla sua percezione.
                        """,
                        "decomposizione_creativa": """
                        La **"Decomposizione Creativa"** si focalizza sulla creazione di **discontinuità e "shift emotivi"** intensi. Utilizzando l'analisi degli onset (punti di attacco del suono), il brano viene frammentato in modo dinamico. I frammenti vengono poi trasformati con variazioni pronunciate ma leggere di pitch e tempo, e alcuni possono essere silenziati o saltati per generare un'esperienza sonora ricca di espressività e rotture inattese.
                        """,
                        "random_chaos": """
                        Il metodo **"Random Chaos"** è progettato per produrre **risultati altamente imprevedibili e sperimentali**. Ogni esecuzione è unica. Vengono applicate operazioni casuali ed estreme come variazioni di volume, inversioni casuali di sezioni e l'aggiunta di rumore. Questo metodo esplora i limiti della manipolazione audio, portando a trasformazioni radicali e spesso disorientanti senza ricorrere a operazioni pesanti.
                        """
                    }
                    selected_method_description = technique_descriptions.get(decomposition_method, "Nessuna descrizione disponibile per questo metodo.")

                    st.subheader("Sintesi Artistica della Decomposizione")
                    artistic_summary = ""
                    if decomposition_method == "cut_up_sonoro":
                        artistic_summary = f"""
                        Con il metodo del **"Cut-up Sonoro"**, il brano originale è stato smembrato e ricombinato, trasformandosi in un'opera di arte sonora ispirata a tecniche di collage e frammentazione. Ogni frammento, lungo circa {fragment_size:.1f} secondi, è stato trattato come un elemento in una composizione decostruita.
                        Il livello di casualità dei tagli ({cut_randomness:.1f}) e lo stile di riassemblaggio ('{reassembly_style}') hanno permesso di **dislocare il significato musicale** originale, creando inaspettate giustapposizioni e ritmi frammentati. Il risultato è un collage sonoro che sfida la percezione tradizionale, invitando l'ascoltatore a trovare nuove narrazioni all'interno della frammentazione.
                        """
                    elif decomposition_method == "remix_destrutturato":
                        artistic_summary = f"""
                        Attraverso il **"Remix Destrutturato"**, l'essenza del brano originale è stata catturata e rielaborata in una forma nuova e sorprendente. Pur mantenendo una certa fedeltà al ritmo (conservazione del battito del {beat_preservation*100:.0f}%), la melodia è stata frammentata e manipolata con leggere variazioni di pitch e tempo.
                        I frammenti, di circa {fragment_size:.1f} secondi, hanno subito alterazioni (frammentazione melodia: {melody_fragmentation:.1f}), ricollocando gli elementi sonori in un **paesaggio acustico reinventato**. Questo remix non è una semplice variazione, ma una vera e propria decostruzione che riassembla gli ingredienti in un'esperienza d'ascolto che è al contempo familiare ed estranea.
                        """
                    elif decomposition_method == "musique_concrete":
                        artistic_summary = f"""
                        Con la tecnica della **"Musique Concrète"**, il brano è stato ridotto ai suoi "grani" sonori più elementari (dimensione dei grani: {grain_size:.2f} secondi). Questi micro-frammenti sono stati manipolati con inversioni e variazioni di ampiezza, per poi essere ricombinati con una densità ({texture_density:.1f}) che crea una nuova tessitura.
                        Il risultato è un'opera sonora astratta che **esplora le qualità timbriche intrinseche del suono**, al di beyond della sua organizzazione musicale originale. L'ascolto si trasforma in un viaggio attraverso paesaggi sonori inusuali, dove il timbro e la consistenza diventano i veri protagonisti.
                        """
                    elif decomposition_method == "decostruzione_postmoderna":
                        artistic_summary = f"""
                        La **"Decostruzione Postmoderna"** ha applicato un filtro concettuale al brano, interrogandone il significato e la percezione. Con frammenti di {fragment_size:.1f} secondi, abbiamo esplorato l'ironia ({irony_level:.1f}) e gli spostamenti di contesto ({context_shift:.1f}).
                        Elementi riconoscibili sono stati trattati in modo inaspettato (es. alterazioni di volume o inversioni rapide), e sono stati introdotti sottili elementi di rottura. L'obiettivo non è solo trasformare il suono, ma anche provocare una **riflessione critica sull'opera e sulla sua fruizione**, trasformando il familiare in qualcosa di leggermente destabilizzante ma affascinante.
                        """
                    elif decomposition_method == "decomposizione_creativa":
                        artistic_summary = f"""
                        Attraverso la **"Decomposizione Creativa"**, il brano è stato frammentato in base a punti di taglio generati dinamicamente, consentendo interventi mirati che generano forti discontinuità ({discontinuity:.1f}) e shift emotivi ({emotional_shift:.1f}).
                        I frammenti, di circa {fragment_size:.1f} secondi, sono stati soggetti a variazioni leggere di pitch e tempo, e alcuni sono stati deliberatamente silenziati o saltati. Il risultato è un'esperienza sonora **ricca di colpi di scena e improvvisi cambi di umore**, un flusso e riflusso di stati emotivi e tensioni acustiche.
                        """
                    elif decomposition_method == "random_chaos":
                        artistic_summary = f"""
                        Il **"Random Chaos"** ha spinto il brano originale nei suoi limiti, creando un'opera unica e imprevedibile. Con un livello di caos di {chaos_level:.1f}, l'audio è stato sottoposto a **trasformazioni radicali e casuali**, come variazioni di volume, inversioni di sezioni e l'introduzione di rumori.
                        Il risultato è un'esplorazione sonora che sfugge a qualsiasi classificazione, un viaggio in un **paesaggio acustico alieno** dove l'originale è appena un eco lontano, e ogni ascolto rivela nuove, sorprendenti anomalie.
                        """
                    st.markdown(artistic_summary)

                    st.markdown("---")
                    st.markdown(f"**Descrizione della Tecnica Applicata:**")
                    st.markdown(selected_method_description)

                    st.markdown("---")
                    st.markdown("### Riepilogo dei Cambiamenti Quantitativi:")

                    analysis_text = f"""
                    * **Durata:** L'audio originale, di **{original_duration:.2f} secondi**, è stato trasformato in un brano di **{new_duration:.2f} secondi**. Questo indica un {'allungamento' if new_duration > original_duration else 'accorciamento'} di **{abs(new_duration - original_duration):.2f} secondi**.
                    * **Energia RMS (Volume Percepito):** Il livello di energia RMS (Root Mean Square), che è un indicatore del volume percepito, ha avuto una variazione di **{processed_rms - original_rms:.4f}**. Questo significa che il suono risultante è generalmente {'più forte' if processed_rms > original_rms else 'più debole' if processed_rms < original_rms else 'simile'} in termini di volume medio.
                    * **Variazione Spettrale:** La variazione spettrale di **{spectral_diff:.2e}** quantifica quanto è cambiato il "colore" o la distribuzione delle frequenze rispetto all'originale. Un valore più alto indica una trasformazione più significativa del timbro e della texture sonora.

                    Questi cambiamenti riflettono l'impatto dei parametri scelti (`Dimensione Frammenti: {fragment_size}s`, `Livello di Chaos: {chaos_level}`, `Conservazione Struttura: {structure_preservation}`).
                    """
                    st.markdown(analysis_text)

                    with st.expander("📊 Confronto Forme d'Onda"):
                        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
                        time_orig = np.linspace(0, len(audio)/sr, len(audio))
                        ax1.plot(time_orig, audio, color='blue', alpha=0.7)
                        ax1.set_title("Forma d'Onda Originale")
                        ax1.set_xlabel("Tempo (sec)")
                        ax1.set_ylabel("Ampiezza")
                        ax1.grid(True, alpha=0.3)
                        
                        time_proc = np.linspace(0, len(processed_audio)/sr, len(processed_audio))
                        ax2.plot(time_proc, processed_audio, color='red', alpha=0.7)
                        ax2.set_title(f"Forma d'Onda Decomposta ({method_names[decomposition_method]})")
                        ax2.set_xlabel("Tempo (sec)")
                        ax2.set_ylabel("Ampiezza")
                        ax2.grid(True, alpha=0.3)
                        
                        plt.tight_layout()
                        st.pyplot(fig)

                    with st.expander("🎼 Analisi Spettrale"):
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                        
                        D_orig = librosa.amplitude_to_db(np.abs(librosa.stft(audio, n_fft=2048)), ref=np.max)
                        librosa.display.specshow(D_orig, y_axis='hz', x_axis='time', sr=sr, ax=ax1)
                        ax1.set_title("Spettrogramma Originale")
                        ax1.set_xlabel("Tempo (sec)")
                        ax1.set_ylabel("Frequenza (Hz)")
                        
                        D_proc = librosa.amplitude_to_db(np.abs(librosa.stft(processed_audio, n_fft=2048)), ref=np.max)
                        librosa.display.specshow(D_proc, y_axis='hz', x_axis='time', sr=sr, ax=ax2)
                        ax2.set_title("Spettrogramma Decomposto")
                        ax2.set_xlabel("Tempo (sec)")
                        ax2.set_ylabel("Frequenza (Hz)")
                        
                        plt.tight_layout()
                        st.pyplot(fig)
    except Exception as e:
        st.error(f"❌ Errore nel processamento: {str(e)}")
        st.error(f"Dettagli: {traceback.format_exc()}")
    finally:
        # Clean up temporary files and directory
        try:
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
            if os.path.exists(processed_tmp_path):
                os.unlink(processed_tmp_path)
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)
        except Exception as e:
            st.error(f"Errore durante la pulizia dei file temporanei: {e}")

else:
    st.info("👆 Carica un file audio per iniziare la decomposizione")
    
    with st.expander("📖 Come usare MusicDecomposer"):
        st.markdown("""
        ### Metodi di Decomposizione:

        **Cut-up Sonoro**
        - Ispirati a una tecnica letteraria di taglio e riassemblaggio
        - Taglia l'audio in frammenti e li riassembla casualmente
        - Ottimo per creare collage sonori sperimentali

        **Remix Destrutturato**
        - Mantiene elementi riconoscibili ma li ricontestualizza
        - Preserva parzialmente il ritmo originale con manipolazioni leggere
        - Ideale per remix creativi e riarrangiamenti

        **Musique Concrète**
        - Basato sui principi di manipolazione sonora
        - Utilizza granular synthesis e manipolazioni concrete leggere
        - Perfetto per texture sonore astratte

        **Decostruzione Postmoderna**
        - Decostruisce il significato musicale originale con effetti sottili
        - Applica ironia e spostamenti di contesto minimi
        - Crea riflessioni critiche sull'opera originale

        **Decomposizione Creativa**
        - Focus su discontinuità e shift emotivi controllati
        - Trasformazioni basate su punti di attacco e variazioni leggere
        - Genera variazioni espressive intense senza operazioni pesanti

        **Random Chaos**
        - Ogni esecuzione è completamente diversa ma con operazioni leggere
        - Operazioni casuali come inversioni, volume e rumore
        - Risultati imprevedibili e sperimentali mantenendo stabilità

        ### Parametri:
        - **Dimensione Frammenti**: Quanto grandi sono i pezzi tagliati
        - **Livello di Chaos**: Intensità delle trasformazioni (limitata per stabilità)
        - **Conservazione Struttura**: Quanto mantenere dell'originale
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

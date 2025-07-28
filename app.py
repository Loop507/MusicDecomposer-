mood_labels = np.array([])
    # Controlla se features è stato creato e non è vuoto
    if features.size > 0:
        try:
            features_scaled = StandardScaler().fit_transform(features.T) # Trasponi per avere (n_frames, n_features)
            # Controlla che features_scaled non sia vuoto dopo lo scaling
            if features_scaled.shape[0] == 0:
                st.warning("Scaled features are empty after scaling. Cannot perform KMeans clustering.")
            else:
                # Determina il numero di cluster
                n_clusters = min(8, features_scaled.shape[0] // 10)
                if n_clusters < 1:
                    n_clusters = 1 # Usa almeno 1 cluster se ci sono dati

                # Esegui KMeans
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                mood_labels = kmeans.fit_predict(features_scaled)
                # DEBUG: st.write(f"DEBUG: KMeans successful, n_clusters={n_clusters}, mood_labels.shape={mood_labels.shape}")

        except ValueError as ve:
            st.warning(f"KMeans clustering failed due to data issue (e.g., constant features): {ve}. Mood labels will be empty.")
            mood_labels = np.array([])
        except Exception as e:
            st.warning(f"KMeans clustering failed unexpectedly: {e}. Mood labels will be empty.")
            mood_labels = np.array([])
    else:
        st.warning("Features for clustering could not be generated (likely chroma or MFCC were empty). Mood labels will be empty.")

    # Gestisci il caso in cui mood_labels è vuoto
    if mood_labels.size == 0:
        st.warning("Mood labels could not be generated or are empty. Returning original audio as fallback in decomposizione_creativa.")
        return audio # Fallback se il clustering non produce etichette valide

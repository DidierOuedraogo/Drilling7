import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
import base64

# Configuration de la page
st.set_page_config(
    page_title="Analyse de Forages Miniers",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Titre de l'application
st.title("Analyse de Forages Miniers")

# Fonction pour convertir des dataframes en CSV téléchargeables
def get_csv_download_link(df, filename):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'Télécharger {filename}'
    return href

# Initialisation des variables de session
if 'collar_df' not in st.session_state:
    st.session_state.collar_df = None
if 'survey_df' not in st.session_state:
    st.session_state.survey_df = None
if 'lithology_df' not in st.session_state:
    st.session_state.lithology_df = None
if 'assays_df' not in st.session_state:
    st.session_state.assays_df = None

# Fonction pour calculer les coordonnées 3D des forages
def calculate_drillhole_coordinates(collar_df, survey_df, hole_id_col, depth_col, 
                                   collar_x_col, collar_y_col, collar_z_col,
                                   survey_hole_id_col, survey_depth_col, 
                                   survey_azimuth_col, survey_dip_col):
    result_df = pd.DataFrame()
    
    for hole_id in collar_df[hole_id_col].unique():
        # Récupérer les données du collar
        collar_row = collar_df[collar_df[hole_id_col] == hole_id].iloc[0]
        x0, y0, z0 = collar_row[collar_x_col], collar_row[collar_y_col], collar_row[collar_z_col]
        
        # Récupérer les données de survey pour ce trou
        hole_surveys = survey_df[survey_df[survey_hole_id_col] == hole_id].sort_values(by=survey_depth_col)
        
        if hole_surveys.empty:
            # Si pas de données de survey, on utilise un forage vertical
            st.warning(f"Pas de données de survey pour le forage {hole_id}. Un forage vertical sera utilisé.")
            max_depth = collar_df[collar_df[hole_id_col] == hole_id][depth_col].values[0]
            df = pd.DataFrame({
                'hole_id': [hole_id, hole_id],
                'depth': [0, max_depth],
                'x': [x0, x0],
                'y': [y0, y0],
                'z': [z0, z0 - max_depth],
            })
            result_df = pd.concat([result_df, df])
            continue
        
        # Initialiser les points
        points = [(0, x0, y0, z0)]
        
        # Pour chaque intervalle de survey
        for i in range(len(hole_surveys)):
            row = hole_surveys.iloc[i]
            depth = row[survey_depth_col]
            azimuth = row[survey_azimuth_col]
            dip = row[survey_dip_col]
            
            # Si ce n'est pas le premier point, calculer le segment
            if i > 0:
                prev_row = hole_surveys.iloc[i-1]
                prev_depth = prev_row[survey_depth_col]
                prev_azimuth = prev_row[survey_azimuth_col]
                prev_dip = prev_row[survey_dip_col]
                
                # Longueur du segment
                segment_length = depth - prev_depth
                
                # Convertir en radians
                azimuth_rad = np.radians(prev_azimuth)
                dip_rad = np.radians(prev_dip)
                
                # Calculer les composantes du vecteur direction
                dx = np.sin(azimuth_rad) * np.cos(dip_rad) * segment_length
                dy = np.cos(azimuth_rad) * np.cos(dip_rad) * segment_length
                dz = -np.sin(dip_rad) * segment_length  # Négatif car z diminue avec la profondeur
                
                # Ajouter le nouveau point
                last_x, last_y, last_z = points[-1][1], points[-1][2], points[-1][3]
                points.append((depth, last_x + dx, last_y + dy, last_z + dz))
            else:
                # Premier point du survey
                points.append((depth, x0, y0, z0))
        
        # Créer un dataframe pour ce trou
        df = pd.DataFrame(points, columns=['depth', 'x', 'y', 'z'])
        df['hole_id'] = hole_id
        df = df[['hole_id', 'depth', 'x', 'y', 'z']]
        
        result_df = pd.concat([result_df, df])
    
    return result_df

# Interface principale avec onglets
tabs = st.tabs(["Chargement des données", "Aperçu des données", "Statistiques", "Visualisation 3D"])

# Onglet 1 : Chargement des données
with tabs[0]:
    st.header("Chargement des fichiers CSV")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Chargement des collars
        st.subheader("Fichier Collars")
        collar_file = st.file_uploader("Chargez le fichier CSV des collars", type=['csv'])
        
        if collar_file is not None:
            try:
                temp_df = pd.read_csv(collar_file)
                st.success(f"Fichier chargé avec succès : {collar_file.name}")
                st.dataframe(temp_df.head(3))
                
                # Sélection des colonnes
                st.subheader("Correspondance des colonnes pour Collars")
                hole_id_col = st.selectbox("Colonne ID du forage (HOLE_ID)", temp_df.columns.tolist())
                collar_x_col = st.selectbox("Colonne X", temp_df.columns.tolist())
                collar_y_col = st.selectbox("Colonne Y", temp_df.columns.tolist())
                collar_z_col = st.selectbox("Colonne Z (élévation)", temp_df.columns.tolist())
                max_depth_col = st.selectbox("Colonne profondeur totale (EOH)", temp_df.columns.tolist())
                
                if st.button("Valider les collars"):
                    st.session_state.collar_df = temp_df
                    st.session_state.collar_cols = {
                        'hole_id': hole_id_col,
                        'x': collar_x_col,
                        'y': collar_y_col,
                        'z': collar_z_col,
                        'depth': max_depth_col
                    }
                    st.success("Données collars validées!")
            except Exception as e:
                st.error(f"Erreur lors du chargement du fichier : {e}")
        
        # Chargement des surveys
        st.subheader("Fichier Survey")
        survey_file = st.file_uploader("Chargez le fichier CSV des surveys", type=['csv'])
        
        if survey_file is not None:
            try:
                temp_df = pd.read_csv(survey_file)
                st.success(f"Fichier chargé avec succès : {survey_file.name}")
                st.dataframe(temp_df.head(3))
                
                # Sélection des colonnes
                st.subheader("Correspondance des colonnes pour Survey")
                survey_hole_id_col = st.selectbox("Colonne ID du forage", temp_df.columns.tolist())
                survey_depth_col = st.selectbox("Colonne profondeur", temp_df.columns.tolist())
                survey_azimuth_col = st.selectbox("Colonne azimut", temp_df.columns.tolist())
                survey_dip_col = st.selectbox("Colonne pendage (dip)", temp_df.columns.tolist())
                
                if st.button("Valider les surveys"):
                    st.session_state.survey_df = temp_df
                    st.session_state.survey_cols = {
                        'hole_id': survey_hole_id_col,
                        'depth': survey_depth_col,
                        'azimuth': survey_azimuth_col,
                        'dip': survey_dip_col
                    }
                    st.success("Données survey validées!")
            except Exception as e:
                st.error(f"Erreur lors du chargement du fichier : {e}")
    
    with col2:
        # Chargement des lithologies
        st.subheader("Fichier Lithologie")
        lithology_file = st.file_uploader("Chargez le fichier CSV des lithologies", type=['csv'])
        
        if lithology_file is not None:
            try:
                temp_df = pd.read_csv(lithology_file)
                st.success(f"Fichier chargé avec succès : {lithology_file.name}")
                st.dataframe(temp_df.head(3))
                
                # Sélection des colonnes
                st.subheader("Correspondance des colonnes pour Lithologie")
                litho_hole_id_col = st.selectbox("Colonne ID du forage pour lithologie", temp_df.columns.tolist())
                litho_from_col = st.selectbox("Colonne profondeur début (From)", temp_df.columns.tolist())
                litho_to_col = st.selectbox("Colonne profondeur fin (To)", temp_df.columns.tolist())
                litho_code_col = st.selectbox("Colonne code lithologique", temp_df.columns.tolist())
                
                if st.button("Valider les lithologies"):
                    st.session_state.lithology_df = temp_df
                    st.session_state.lithology_cols = {
                        'hole_id': litho_hole_id_col,
                        'from': litho_from_col,
                        'to': litho_to_col,
                        'code': litho_code_col
                    }
                    st.success("Données lithologie validées!")
            except Exception as e:
                st.error(f"Erreur lors du chargement du fichier : {e}")
        
        # Chargement des analyses
        st.subheader("Fichier Analyses (Assays)")
        assays_file = st.file_uploader("Chargez le fichier CSV des analyses", type=['csv'])
        
        if assays_file is not None:
            try:
                temp_df = pd.read_csv(assays_file)
                st.success(f"Fichier chargé avec succès : {assays_file.name}")
                st.dataframe(temp_df.head(3))
                
                # Sélection des colonnes
                st.subheader("Correspondance des colonnes pour Analyses")
                assay_hole_id_col = st.selectbox("Colonne ID du forage pour analyses", temp_df.columns.tolist())
                assay_from_col = st.selectbox("Colonne profondeur début pour analyses", temp_df.columns.tolist())
                assay_to_col = st.selectbox("Colonne profondeur fin pour analyses", temp_df.columns.tolist())
                
                # Sélection des éléments d'analyse
                assay_elements = st.multiselect("Éléments analysés", 
                                               [c for c in temp_df.columns if c not in [assay_hole_id_col, assay_from_col, assay_to_col]])
                
                if st.button("Valider les analyses"):
                    st.session_state.assays_df = temp_df
                    st.session_state.assay_cols = {
                        'hole_id': assay_hole_id_col,
                        'from': assay_from_col,
                        'to': assay_to_col,
                        'elements': assay_elements
                    }
                    st.success("Données analyses validées!")
            except Exception as e:
                st.error(f"Erreur lors du chargement du fichier : {e}")

# Onglet 2 : Aperçu des données
with tabs[1]:
    st.header("Aperçu des données")
    
    dataset_select = st.selectbox(
        "Sélectionnez un jeu de données à visualiser",
        ["Collars", "Survey", "Lithologie", "Analyses"]
    )
    
    if dataset_select == "Collars" and st.session_state.collar_df is not None:
        st.subheader("Données des collars")
        st.dataframe(st.session_state.collar_df)
        st.markdown(get_csv_download_link(st.session_state.collar_df, "collars_data"), unsafe_allow_html=True)
        
    elif dataset_select == "Survey" and st.session_state.survey_df is not None:
        st.subheader("Données des surveys")
        st.dataframe(st.session_state.survey_df)
        st.markdown(get_csv_download_link(st.session_state.survey_df, "survey_data"), unsafe_allow_html=True)
        
    elif dataset_select == "Lithologie" and st.session_state.lithology_df is not None:
        st.subheader("Données des lithologies")
        st.dataframe(st.session_state.lithology_df)
        st.markdown(get_csv_download_link(st.session_state.lithology_df, "lithology_data"), unsafe_allow_html=True)
        
    elif dataset_select == "Analyses" and st.session_state.assays_df is not None:
        st.subheader("Données des analyses")
        st.dataframe(st.session_state.assays_df)
        st.markdown(get_csv_download_link(st.session_state.assays_df, "assays_data"), unsafe_allow_html=True)
        
    else:
        st.info(f"Aucune donnée disponible pour {dataset_select}. Veuillez d'abord charger les fichiers.")

# Onglet 3 : Statistiques
with tabs[2]:
    st.header("Statistiques descriptives")
    
    if st.session_state.collar_df is not None and st.session_state.survey_df is not None:
        st.subheader("Statistiques des forages")
        
        # Nombre de forages
        num_holes = len(st.session_state.collar_df[st.session_state.collar_cols['hole_id']].unique())
        st.metric("Nombre de forages", num_holes)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Profondeur totale forée
            total_depth = st.session_state.collar_df[st.session_state.collar_cols['depth']].sum()
            st.metric("Profondeur totale forée (m)", f"{total_depth:.2f}")
        
        with col2:
            # Profondeur moyenne des forages
            avg_depth = st.session_state.collar_df[st.session_state.collar_cols['depth']].mean()
            st.metric("Profondeur moyenne des forages (m)", f"{avg_depth:.2f}")
            
        with col3:
            # Profondeur max des forages
            max_depth = st.session_state.collar_df[st.session_state.collar_cols['depth']].max()
            st.metric("Profondeur maximale des forages (m)", f"{max_depth:.2f}")
    
    if st.session_state.assays_df is not None and 'elements' in st.session_state.assay_cols:
        st.subheader("Statistiques des analyses")
        
        # Sélection de l'élément à analyser
        element = st.selectbox("Sélectionnez un élément", st.session_state.assay_cols['elements'])
        
        if element:
            # Statistiques descriptives
            stats = st.session_state.assays_df[element].describe()
            st.write(stats)
            
            # Histogramme
            fig = px.histogram(
                st.session_state.assays_df, 
                x=element, 
                nbins=50,
                title=f"Distribution de {element}"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Boxplot
            fig = px.box(
                st.session_state.assays_df, 
                y=element,
                title=f"Boxplot de {element}"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Statistiques lithologiques
    if st.session_state.lithology_df is not None:
        st.subheader("Statistiques des lithologies")
        
        litho_counts = st.session_state.lithology_df[st.session_state.lithology_cols['code']].value_counts()
        
        # Calculer les longueurs de chaque intervalle lithologique
        st.session_state.lithology_df['length'] = st.session_state.lithology_df[st.session_state.lithology_cols['to']] - st.session_state.lithology_df[st.session_state.lithology_cols['from']]
        
        # Grouper par code lithologique et sommer les longueurs
        litho_lengths = st.session_state.lithology_df.groupby(st.session_state.lithology_cols['code'])['length'].sum().sort_values(ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Nombre d'occurrences par lithologie")
            fig = px.bar(
                x=litho_counts.index,
                y=litho_counts.values,
                labels={'x': 'Code lithologique', 'y': 'Nombre d\'occurrences'},
                title="Fréquence des codes lithologiques"
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.write("Longueur totale par lithologie (m)")
            fig = px.bar(
                x=litho_lengths.index,
                y=litho_lengths.values,
                labels={'x': 'Code lithologique', 'y': 'Longueur totale (m)'},
                title="Longueur cumulée par lithologie"
            )
            st.plotly_chart(fig, use_container_width=True)

# Onglet 4 : Visualisation 3D
with tabs[3]:
    st.header("Visualisation 3D des forages")
    
    if st.session_state.collar_df is not None and st.session_state.survey_df is not None:
        try:
            # Calcul des coordonnées 3D des forages
            drillhole_coords = calculate_drillhole_coordinates(
                st.session_state.collar_df, 
                st.session_state.survey_df,
                st.session_state.collar_cols['hole_id'],
                st.session_state.collar_cols['depth'],
                st.session_state.collar_cols['x'],
                st.session_state.collar_cols['y'],
                st.session_state.collar_cols['z'],
                st.session_state.survey_cols['hole_id'],
                st.session_state.survey_cols['depth'],
                st.session_state.survey_cols['azimuth'],
                st.session_state.survey_cols['dip']
            )
            
            # Créer la figure 3D
            fig = go.Figure()
            
            # Options d'affichage
            col1, col2 = st.columns([3, 1])
            
            with col2:
                st.subheader("Options d'affichage")
                show_lithology = st.checkbox("Afficher la lithologie", value=False)
                show_assays = st.checkbox("Afficher les analyses", value=False)
                
                if show_assays and st.session_state.assays_df is not None:
                    assay_element = st.selectbox(
                        "Élément à afficher", 
                        st.session_state.assay_cols['elements'],
                        key="assay_element_3d"
                    )
                    
                    # Plage de couleurs pour les analyses
                    cmin = st.session_state.assays_df[assay_element].min()
                    cmax = st.session_state.assays_df[assay_element].max()
                    
                    col1_sub, col2_sub = st.columns(2)
                    with col1_sub:
                        cmin = st.number_input("Valeur min", value=float(cmin), format="%.3f")
                    with col2_sub:
                        cmax = st.number_input("Valeur max", value=float(cmax), format="%.3f")
            
            with col1:
                # Ajouter chaque forage à la figure
                for hole_id in drillhole_coords['hole_id'].unique():
                    hole_data = drillhole_coords[drillhole_coords['hole_id'] == hole_id]
                    
                    # Tracer le forage
                    fig.add_trace(go.Scatter3d(
                        x=hole_data['x'],
                        y=hole_data['y'],
                        z=hole_data['z'],
                        mode='lines',
                        name=hole_id,
                        line=dict(width=4, color='black'),
                        showlegend=True
                    ))
                    
                    # Si l'option lithologie est activée et que les données sont disponibles
                    if show_lithology and st.session_state.lithology_df is not None:
                        litho_data = st.session_state.lithology_df[
                            st.session_state.lithology_df[st.session_state.lithology_cols['hole_id']] == hole_id
                        ]
                        
                        if not litho_data.empty:
                            # Créer un dictionnaire de couleurs pour les codes lithologiques
                            litho_codes = litho_data[st.session_state.lithology_cols['code']].unique()
                            colors = px.colors.qualitative.Plotly[:len(litho_codes)]
                            color_map = dict(zip(litho_codes, colors))
                            
                            for _, row in litho_data.iterrows():
                                from_depth = row[st.session_state.lithology_cols['from']]
                                to_depth = row[st.session_state.lithology_cols['to']]
                                litho_code = row[st.session_state.lithology_cols['code']]
                                
                                # Interpoler les coordonnées pour cet intervalle lithologique
                                interval_coords = hole_data[
                                    (hole_data['depth'] >= from_depth) & 
                                    (hole_data['depth'] <= to_depth)
                                ]
                                
                                if len(interval_coords) > 1:
                                    fig.add_trace(go.Scatter3d(
                                        x=interval_coords['x'],
                                        y=interval_coords['y'],
                                        z=interval_coords['z'],
                                        mode='lines',
                                        line=dict(width=6, color=color_map[litho_code]),
                                        name=f"{hole_id} - {litho_code}",
                                        showlegend=False
                                    ))
                    
                    # Si l'option analyses est activée et que les données sont disponibles
                    if show_assays and st.session_state.assays_df is not None and 'assay_element_3d' in st.session_state:
                        assay_data = st.session_state.assays_df[
                            st.session_state.assays_df[st.session_state.assay_cols['hole_id']] == hole_id
                        ]
                        
                        if not assay_data.empty:
                            for _, row in assay_data.iterrows():
                                from_depth = row[st.session_state.assay_cols['from']]
                                to_depth = row[st.session_state.assay_cols['to']]
                                value = row[assay_element]
                                
                                # Interpoler les coordonnées pour cet intervalle d'analyse
                                interval_coords = hole_data[
                                    (hole_data['depth'] >= from_depth) & 
                                    (hole_data['depth'] <= to_depth)
                                ]
                                
                                if len(interval_coords) > 1:
                                    fig.add_trace(go.Scatter3d(
                                        x=interval_coords['x'],
                                        y=interval_coords['y'],
                                        z=interval_coords['z'],
                                        mode='lines',
                                        line=dict(width=8, color=px.colors.sequential.Viridis[int((value-cmin)/(cmax-cmin)*9)] if cmax > cmin else 'blue'),
                                        name=f"{hole_id} - {assay_element} = {value:.3f}",
                                        showlegend=False
                                    ))
                
                # Ajouter les marqueurs des collars
                fig.add_trace(go.Scatter3d(
                    x=st.session_state.collar_df[st.session_state.collar_cols['x']],
                    y=st.session_state.collar_df[st.session_state.collar_cols['y']],
                    z=st.session_state.collar_df[st.session_state.collar_cols['z']],
                    mode='markers',
                    marker=dict(size=5, color='red'),
                    name='Collars',
                    text=st.session_state.collar_df[st.session_state.collar_cols['hole_id']],
                    hoverinfo='text'
                ))
                
                # Configuration de la figure
                fig.update_layout(
                    scene=dict(
                        aspectmode='data',
                        xaxis_title='Est',
                        yaxis_title='Nord',
                        zaxis_title='Élévation',
                    ),
                    height=800,
                    margin=dict(l=0, r=0, b=0, t=30),
                    title="Visualisation 3D des forages"
                )
                
                # Ajouter une échelle de couleur pour les analyses
                if show_assays and st.session_state.assays_df is not None:
                    fig.update_layout(
                        coloraxis=dict(
                            colorscale='Viridis',
                            cmin=cmin,
                            cmax=cmax,
                            colorbar=dict(
                                title=assay_element,
                                thickness=20,
                                len=0.7,
                                x=0.95
                            )
                        )
                    )
                
                # Afficher la figure 3D
                st.plotly_chart(fig, use_container_width=True)
                
                # Légende des lithologies
                if show_lithology and st.session_state.lithology_df is not None:
                    st.subheader("Légende des lithologies")
                    litho_codes = st.session_state.lithology_df[st.session_state.lithology_cols['code']].unique()
                    colors = px.colors.qualitative.Plotly[:len(litho_codes)]
                    color_map = dict(zip(litho_codes, colors))
                    
                    legend_html = "
"
                    for code, color in color_map.items():
                        legend_html += f"
"
                        legend_html += f"
"
                        legend_html += f"
{code}
"
                        legend_html += "
"
                    legend_html += "
"
                    
                    st.markdown(legend_html, unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"Erreur lors de la création de la visualisation 3D : {e}")
    else:
        st.info("Veuillez d'abord charger les fichiers collar et survey pour pouvoir visualiser les forages en 3D.")

# Crédits et informations
st.markdown("---")
st.markdown("Développé pour l'analyse de données de forages miniers")
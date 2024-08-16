from io import BytesIO
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
import streamlit as st
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import tempfile

def main():
    # Initialize data
    if 'data' not in st.session_state:
        st.session_state.data = None

    uploaded_file = None  # Initialize uploaded_file
    st.title("Aplicación de Clustering con K-means")

    menu = ["Inicio", "Cuestionario", "Procesamiento de Datos Google", "Análisis K-means"]
    choice = st.sidebar.selectbox("Menú", menu)

    if choice == "Inicio":
        st.subheader("Inicio")
        st.write("Bienvenido a la aplicación de clustering. Usa el menú para navegar.")

    elif choice == "Cuestionario":
        st.subheader("Cuestionario")
        st.write("Aquí estará el cuestionario para recolectar información del usuario.")

        # Información personal
        zodiac_sign = st.selectbox("Selecciona tu signo zodiacal", 
                                ["Aries", "Tauro", "Géminis", "Cáncer", "Leo", "Virgo", 
                                    "Libra", "Escorpio", "Sagitario", "Capricornio", "Acuario", "Piscis"])
        age = st.number_input("¿Cuál es tu de edad?", min_value=12, max_value=100, value=22)
        gender = st.selectbox("¿Cuál es tu género?", ["Masculino", "Femenino"])
        
        # Slider de 1 a 10
        responsabilidad_disciplina_tareas_diarias = st.slider("Me considero una persona muy responsable y disciplinada en mis tareas diarias", 1, 10, 5)
        preferencia_horario_estructurado = st.slider("Prefiero seguir un horario estructurado en lugar de improvisar mis actividades diarias", 1, 10, 5)

        adaptabilidad_cambios_variedad = st.slider("Me adapto fácilmente a los cambios y disfruto de la variedad en mi vida", 1, 10, 5)
        comodidad_situaciones_nuevas = st.slider("Me siento cómodo en situaciones nuevas y desconocidas, y rápidamente encuentro la manera de ajustarme", 1, 10, 5)

        conexion_emocional_otros = st.slider("Siento una profunda conexión emocional con los demás y a menudo capto sus sentimientos", 1, 10, 5)
        afectado_emociones_ajenas = st.slider("A menudo, las emociones de otras personas me afectan fuertemente, incluso cuando no me lo dicen directamente", 1, 10, 5)

        motivacion_grandes_metas_liderazgo = st.slider("Me siento motivado a alcanzar grandes metas y disfruto liderar a otros hacia el éxito", 1, 10, 5)
        iniciativa_proyectos_grupales = st.slider("Me gusta tomar la iniciativa en proyectos grupales y siento que tengo la habilidad de dirigir a los demás", 1, 10, 5)

        creatividad_expresion_artistica = st.slider("Me considero una persona muy creativa y disfruto expresar mis ideas de manera artística", 1, 10, 5)
        placer_actividades_creativas = st.slider("Encuentro placer en actividades que me permiten ser creativo, como el arte, la música o la escritura", 1, 10, 5)

        importancia_estabilidad_seguridad = st.slider("Para mí, es fundamental tener estabilidad y seguridad en mi vida personal y profesional", 1, 10, 5)
        incomodidad_incertidumbre_cambios = st.slider("Me siento incómodo cuando enfrento incertidumbre o cambios inesperados en mi entorno", 1, 10, 5)

        compromiso_pasion_intensidad = st.slider("Cuando me comprometo con algo o alguien, lo hago con gran pasión e intensidad", 1, 10, 5)
        dificultad_indiferencia_desapasionada = st.slider("Me cuesta ser indiferente o desapasionado; lo que hago, lo hago con todo mi ser", 1, 10, 5)

        persona_optimista_lado_positivo = st.slider("Soy una persona optimista que siempre busca el lado positivo de las cosas", 1, 10, 5)
        entusiasmo_enfrentar_desafios = st.slider("Enfrento los desafíos con entusiasmo y creo que siempre hay una solución positiva para cada problema", 1, 10, 5)

        valoracion_independencia_libertad = st.slider("Valoro mi independencia y me siento incómodo cuando siento que pierdo mi libertad", 1, 10, 5)
        preferencia_decisiones_propias = st.slider("Prefiero tomar decisiones por mí mismo y evito depender demasiado de los demás", 1, 10, 5)

        detallista_precision_tareas = st.slider("Soy muy detallista y me preocupo por hacer las cosas de la manera más precisa posible", 1, 10, 5)
        perfeccionamiento_detalles_proyectos = st.slider("Puedo pasar mucho tiempo perfeccionando los detalles de un proyecto antes de considerarlo terminado", 1, 10, 5)
                
        if st.button("Enviar"):
            user_data = {
                "zodiac_sign": zodiac_sign,
                "age": age,
                "gender": gender,
                "responsabilidad_disciplina_tareas_diarias": responsabilidad_disciplina_tareas_diarias,
                "preferencia_horario_estructurado": preferencia_horario_estructurado,
                "adaptabilidad_cambios_variedad": adaptabilidad_cambios_variedad,
                "comodidad_situaciones_nuevas": comodidad_situaciones_nuevas,
                "conexion_emocional_otros": conexion_emocional_otros,
                "afectado_emociones_ajenas": afectado_emociones_ajenas,
                "motivacion_grandes_metas_liderazgo": motivacion_grandes_metas_liderazgo,
                "iniciativa_proyectos_grupales": iniciativa_proyectos_grupales,
                "creatividad_expresion_artistica": creatividad_expresion_artistica,
                "placer_actividades_creativas": placer_actividades_creativas,
                "importancia_estabilidad_seguridad": importancia_estabilidad_seguridad,
                "incomodidad_incertidumbre_cambios": incomodidad_incertidumbre_cambios,
                "compromiso_pasion_intensidad": compromiso_pasion_intensidad,
                "dificultad_indiferencia_desapasionada": dificultad_indiferencia_desapasionada,
                "persona_optimista_lado_positivo": persona_optimista_lado_positivo,
                "entusiasmo_enfrentar_desafios": entusiasmo_enfrentar_desafios,
                "valoracion_independencia_libertad": valoracion_independencia_libertad,
                "preferencia_decisiones_propias": preferencia_decisiones_propias,
                "detallista_precision_tareas": detallista_precision_tareas,
                "perfeccionamiento_detalles_proyectos": perfeccionamiento_detalles_proyectos
            }
            df = pd.DataFrame([user_data])
            # Exportar a CSV agregandolos al final del archivo
            with open("user_data.csv", "a") as f:
                df.to_csv(f, header=f.tell()==0, index=False, encoding='utf-8')
            st.success("Datos enviados con éxito.")

    elif choice == "Análisis K-means":
        st.subheader("Análisis K-means")
        st.write("Sube el archivo CSV con los datos recolectados para aplicar K-means.")

        uploaded_file = st.file_uploader("Sube tu archivo CSV", type=["csv"])

        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file, encoding='utf-8')  # Specify encoding
            datos_originales = data.copy()

            st.write("Vista previa de los datos:")
            st.dataframe(data.head())

            data = zodiac_sing_and_gender_to_num(data)

            st.dataframe(data.head())

            st.write("Selecciona las variables para el clustering:")

            # Initialize session state for selected columns
            if 'selected_columns' not in st.session_state:
                st.session_state.selected_columns = []

            # Multiselect for variable selection
            selected_columns = st.multiselect("Variables", data.columns.tolist())
            
            data_scaled = None

            if(selected_columns):
                data.dropna(inplace=True)
                scaler = StandardScaler() 
                data_t = scaler.fit_transform(data[selected_columns])
                data_scaled = pd.DataFrame(data_t, columns=selected_columns)

            st.write("Gráfico de Codo para seleccionar la cantidad de clusters optimos")
            mostrar_codo = st.checkbox("Mostrar gráfico del Método del Codo")
            if(mostrar_codo):
                if(len(selected_columns) < 2):
                    st.error("Por favor, selecciona al menos dos variables para el clustering.")
                else:
                    grafico_codo(data_t)





            # Slider for number of clusters
            num_clusters = st.slider("Número de clusters", min_value=2, max_value=12)
            st.session_state.num_clusters = num_clusters

            if len(selected_columns) > 1:
                try:
                    kmeans = KMeans(n_clusters=num_clusters)
                    kmeans.fit(data_t)
                    data['kmeans_cluster'] = kmeans.labels_
                    data_t_kmeans_excel = pd.DataFrame(data, columns=selected_columns + ['kmeans_cluster'])
                    st.session_state.data = data
                    st.session_state.selected_columns = selected_columns
                    st.session_state.data_t = data_t
                except Exception as e:
                    st.error(f"Error al aplicar K-means: {e}")
            else:
                st.session_state.data = None
                st.warning("Por favor, selecciona al menos una variable para el clustering.")


            # Display clustering results if they exist in session state
            if st.session_state.data is not None and st.session_state.data_t is not None:
                data_t = st.session_state.data_t
                data = st.session_state.data
                # Crear las dos figuras
                fig1, ax1 = plt.subplots()
                fig2, ax2 = plt.subplots()

                # Gráfico de dispersión con los clusters
                processed_columns = st.session_state.selected_columns

                data_pca = None

                if len(processed_columns) > 2:
                    st.warning("Más de dos columnas seleccionadas. Puedes aplicar PCA para reducir a dos dimensiones o seleccionar dos columnas.")
                    
                    if(st.checkbox("Aplicar PCA")):
                        pca = PCA(n_components=2)
                        data_pca = pca.fit_transform(data_t)

                        # Obtener la matriz de covarianza
                        data_t_df = pd.DataFrame(data_t)
                        cov_matrix = data_t_df.cov()

                        # Obtener los componentes principales (autovectores)
                        components = pca.components_
                        components_df = pd.DataFrame(components.T, columns=['PC1', 'PC2'])

                        # Obtener los autovalores
                        explained_variance = pca.explained_variance_

                        # Crear un DataFrame para los autovalores
                        eigenvalues_df = pd.DataFrame({
                            'Component': ['PC1', 'PC2'],
                            'Eigenvalue': explained_variance
                        })

                        # Mostrar los resultados en Streamlit
                        st.write("Componentes PCA:")
                        st.dataframe(components_df)

                        st.write("Autovalores:")
                        st.dataframe(eigenvalues_df)

                        # Imprimir la covarianza de las variables con cada componente
                        st.write("Covarianza de las variables con cada componente:")
                        st.write("PC1:")
                        st.dataframe(cov_matrix.dot(components_df['PC1']))
                        st.write("PC2:")
                        st.dataframe(cov_matrix.dot(components_df['PC2']))

                        st.write("PCA")
                        st.dataframe(data_pca)

                        # Preparar los datos para la gráfica
                        data_pca = pd.DataFrame(data_pca, columns=['PC1', 'PC2'])

                        # Crear y mostrar el gráfico
                        fig1, ax1 = plt.subplots()
                        fig1.set_size_inches(10, 6)
                        sns.scatterplot(x='PC1', y='PC2', hue=data['kmeans_cluster'], palette='dark', data=data_pca, ax=ax1)
                        ax1.set_title("PCA")

                        # Mostrar el gráfico en Streamlit
                        st.pyplot(fig1)
                    else:
                        columns = st.multiselect("Selecciona dos columnas para el gráfico de dispersión", processed_columns)
                        if len(columns) == 2:

                            if 'zodiac_sign' in columns:
                                data_kmeans = zodiac_sing_and_gender_from_num(data)
                            
                            fig1.set_size_inches(10, 6)
                            sns.scatterplot(x=columns[1], y=columns[0], hue=data['kmeans_cluster'], palette='dark', data=data_kmeans, ax=ax1)
                            ax1.set_title("K-means Clustering")
                            st.pyplot(fig1)
                elif len(processed_columns) == 2:
                    if 'zodiac_sign' in processed_columns:
                        data_kmeans = zodiac_sing_and_gender_from_num(data)
                    else:
                        data_kmeans = data.copy()
                    fig1.set_size_inches(10, 6)
                    sns.scatterplot(x=data_kmeans[processed_columns[1]], y=data_kmeans[processed_columns[0]], hue=data_kmeans['kmeans_cluster'], palette='dark', ax=ax1)
                    ax1.set_title("K-means Clustering")
                    st.pyplot(fig1)
                
                # Procesar con DBSSCAN
                st.write("Procesar con DBSCAN")
                eps = st.slider("Epsilon (Distancia para considerar un vecino)", 0.01, 1.0, 0.5, 0.01)
                min_samples = st.slider("Muestras mínimas (Puntos necesarios para considerar un nucleo)", 2, 10, 5)
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                data_scan = pd.DataFrame(data_scaled)
                data_dbscan = dbscan.fit(data_scan)
                dbscan_labels = data_dbscan.labels_
                df_dbscan = pd.DataFrame(data_t)
                df_dbscan['dbscan_cluster'] = dbscan_labels
                
                n_clusters_ = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
                n_noise_ = list(dbscan_labels).count(-1) 

                st.write(f"DBSCAN clustering, número de clusters estimado: {n_clusters_}")
                st.write(f"DBSCAN clustering, número de ruido estimado: {n_noise_}")

                unique_labels = set(dbscan_labels)
                core_samples_mask = np.zeros_like(dbscan_labels, dtype=bool)
                core_samples_mask[dbscan.core_sample_indices_] = True

                colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
                for k, col in zip(unique_labels, colors):
                    if k == -1:
                        col = [0, 0, 0, 1]

                    class_member_mask = (dbscan_labels == k)

                    xy = data_t[class_member_mask & core_samples_mask]
                    ax2.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=14)

                    xy = data_t[class_member_mask & ~core_samples_mask]
                    ax2.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=6)

                ax2.set_title('DBSCAN clustering, número de clusters estimado: %d' % n_clusters_)
                st.pyplot(fig2)

                st.write("Exportar resultados:")
                if st.button("Exportar a PDF"):
                    pdf = FPDF(orientation='L', unit='mm', format='A4')
                    pdf.add_page()
                    pdf.set_font("Arial", size=12)
                    pdf.ln(20)

                    # Guardar las gráficas como archivos temporales
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile1:
                        fig1.savefig(tmpfile1.name, format="png")
                        pdf.image(tmpfile1.name, x=10, y=30, w=180)

                    pdf.add_page()
                    pdf.ln(20)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile2:
                        fig2.savefig(tmpfile2.name, format="png")
                        pdf.image(tmpfile2.name, x=10, y=30, w=180)# Mostrar el gráfico en Streamlit
               

                    pdf.ln(100)  # Adjust this value based on the height of your image
                    pdf.set_font("Arial", size=6)  # Reduce font size further

                    # Calculate column width
                    page_width = pdf.w - 20  # Page width minus margins
                    col_width = page_width / 8  # 7 columns + 1 for the index column

                    # Function to add table header
                    def add_table_header(columns):
                        pdf.cell(col_width, 6, 'Index', 1)
                        for col in columns:
                            pdf.cell(col_width, 6, col, 1)
                        pdf.ln()

                    # Function to add table rows
                    def add_table_rows(columns):
                        row_height = 6  # Reduce row height
                        max_rows_per_page = int((pdf.h - 20) / row_height) - 2  # Adjust for margins and header
                        for i in range(len(data)):
                            if i % max_rows_per_page == 0 and i != 0:
                                pdf.add_page()
                                add_table_header(columns)
                            pdf.cell(col_width, 6, str(i), 1)
                            for col in columns:
                                pdf.cell(col_width, 6, str(data.iloc[i][col]), 1)
                            pdf.ln()

                    # Split columns into groups of 7
                    columns = data.columns.tolist()
                    for i in range(0, len(columns), 7):
                        pdf.add_page()
                        current_columns = columns[i:i+7]
                        add_table_header(current_columns)
                        add_table_rows(current_columns)

                    pdf_output = "resultados.pdf"
                    pdf.output(pdf_output)
                    st.success("Resultados exportados a resultados.pdf")

                if st.button("Exportar a Excel"):
                    # Exportar a Excel en diferentes hojas cada dataframe generado con K-means y DBSCAN
                    with pd.ExcelWriter('resultados.xlsx') as writer:
                        datos_originales.to_excel(writer, sheet_name='Datos Originales')
                        data_scaled.to_excel(writer, sheet_name='Datos Standard Scaler')

                        # Si data_t_kmeans_excel tiene zodiac_sign mapearlo a texto
                        if 'zodiac_sign' in data_t_kmeans_excel.columns:
                            data_t_kmeans_excel = zodiac_sing_and_gender_from_num(data_t_kmeans_excel)

                        data_t_kmeans_excel.to_excel(writer, sheet_name='K-means')

                        df_dbscan = pd.DataFrame(df_dbscan)
                        df_dbscan.columns = selected_columns + ['dbscan_cluster']
                        df_dbscan.to_excel(writer, sheet_name='DBSCAN')
                        if data_pca is not None:
                            data_pca.to_excel(writer, sheet_name='PCA')
                    st.success("Resultados exportados a resultados.xlsx")



def zodiac_sing_and_gender_to_num(data):
    # Transformar variables el signo zodical
    zodiac_signs = {
        "Aries": 1, "Tauro": 2, "Géminis": 3, "Cáncer": 4, "Leo": 5, "Virgo": 6,
        "Libra": 7, "Escorpio": 8, "Sagitario": 9, "Capricornio": 10, "Acuario": 11, "Piscis": 12
    }

    genders = {"Masculino": 0, "Femenino": 1}
    data['gender'] = data['gender'].map(genders)
    data['zodiac_sign'] = data['zodiac_sign'].map(zodiac_signs)

    return data

def zodiac_sing_and_gender_from_num(data):
    # Transformar variables el signo zodical
    if('zodiac_sign' in data.columns):
        zodiac_signs = {
            1: "Aries", 2: "Tauro", 3: "Géminis", 4: "Cáncer", 5: "Leo", 6: "Virgo",
            7: "Libra", 8: "Escorpio", 9: "Sagitario", 10: "Capricornio", 11: "Acuario", 12: "Piscis"
        }
        data['zodiac_sign'] = data['zodiac_sign'].map(zodiac_signs)

    if('gender' in data.columns):
        genders = {0: "Masculino", 1: "Femenino"}
        data['gender'] = data['gender'].map(genders)

    return data

def grafico_codo(data):
    elbow_fig, axElbow = plt.subplots()
    wcss = []
    for i in range(1, 12):
        kmeans = KMeans(n_clusters=i, random_state=0)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
    axElbow.plot(range(1, 12), wcss)
    axElbow.set_title('Método del Codo')
    axElbow.set_xlabel('Número de clústeres')
    axElbow.set_ylabel('Inercia')
    st.pyplot(elbow_fig)

if __name__ == "__main__":
    main()

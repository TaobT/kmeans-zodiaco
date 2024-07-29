from io import BytesIO
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from utils.export import export_to_excel
from fpdf import FPDF
import tempfile

def clean_unicode_errors(df):
    # Función para reemplazar caracteres a utf-8 ignorando el header ignorando errores
    for col in df.columns:
        df[col] = df[col].apply(lambda x: x.encode('raw_unicode_escape').decode('utf-8', errors='ignore'))
    return df

def map_google_form_columns(df):
    column_mapping = {
        "Nombre": "full_name",
        "Selecciona tu signo zodiacal": "zodiac_sign",
        "¿Cuál es tu rango de edad?": "age_range",
        "Selecciona tu genero": "gender",
        "¿Cuál es tu nivel de educación?": "education_level",
        "¿Te consideras una persona extrovertida o introvertida?": "extroversion",
        "¿Qué tan importante es para ti la puntualidad?": "punctuality",
        "¿Eres una persona organizada?": "organization",
        "¿Te gusta tomar riesgos?": "risk_taking",
        "¿Te consideras una persona creativa?": "creativity",
        "¿Qué tipo de música prefieres?": "music_preference",
        "¿Cuál es tu género de película favorito?": "movie_genre",
        "¿Qué tipo de deporte prefieres?": "sport_preference",
        "¿Qué tipo de comida prefieres?": "food_preference",
        "En promedio ¿Cuántas horas duermes por noche?": "sleep_hours",
        "¿Haces ejercicio regularmente?": "exercise",
        "¿Fumas?": "smoking",
        "¿Consumes alcohol?": "alcohol",
        "¿Te consideras una persona saludable?": "healthiness",
        "¿Crees en la astrología?": "astrology_belief",
        "¿Crees en la religión?": "religiosity",
        "¿Te consideras una persona espiritual?": "spirituality",
        "¿Qué tan importante es para ti la familia?": "family_importance",
        "¿Qué tan importante es para ti el trabajo?": "work_importance",
        "¿Hablas más de un idioma?": "languages",
        "¿Te consideras una persona tecnológica?": "tech_savvy",
        "¿Te gusta leer?": "reading",
        "¿Te consideras una persona sexualmente activa?": "sexually_active",
        "¿Estás satisfecho/a con tu vida sexual?": "sexual_satisfaction",
        "¿Consideras importante la comunicación en tu vida sexual?": "sexual_communication_importance"
    }
    
    df = df.rename(columns=column_mapping)
    
    # Seleccionar solo las columnas mapeadas y en el orden deseado
    columns_order = [
        "full_name", "zodiac_sign", "age_range", "gender", "education_level", "extroversion",
        "punctuality", "organization", "risk_taking", "creativity", "music_preference", 
        "movie_genre", "sport_preference", "food_preference", "sleep_hours", "exercise", 
        "smoking", "alcohol", "healthiness", "astrology_belief", "religiosity", "spirituality", 
        "family_importance", "work_importance", "languages", "tech_savvy", "reading", 
        "sexually_active", "sexual_satisfaction", "sexual_communication_importance"
    ]
    
    df = df[columns_order]
    return df

def main():
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
        full_name = st.text_input("Nombre completo")
        zodiac_sign = st.selectbox("Selecciona tu signo zodiacal", 
                                ["Aries", "Tauro", "Géminis", "Cáncer", "Leo", "Virgo", 
                                    "Libra", "Escorpio", "Sagitario", "Capricornio", "Acuario", "Piscis"])
        age_range = st.selectbox("¿Cuál es tu rango de edad?", 
                                ["Menor a 18 años", "18 a 25 años", "26 a 35 años", "36 a 45 años", "Más de 45 años"])
        gender = st.selectbox("¿Cuál es tu género?", ["Masculino", "Femenino", "No binario", "Prefiero no decirlo"])
        education_level = st.selectbox("¿Cuál es tu nivel de educación?", 
                                    ["Primaria", "Secundaria", "Preparatoria", "Universidad", "Postgrado"])

        # Personalidad y hábitos
        extroversion = st.selectbox("¿Te consideras una persona extrovertida o introvertida?", ["Extrovertida", "Introvertida"])
        punctuality = st.selectbox("¿Qué tan importante es para ti la puntualidad?", 
                                ["Muy importante", "Importante", "Poco importante", "Nada importante"])
        organization = st.selectbox("¿Te consideras una persona organizada?", ["Sí", "No"])
        risk_taking = st.selectbox("¿Te gusta tomar riesgos?", ["Sí", "No"])
        creativity = st.selectbox("¿Te consideras una persona creativa?", ["Sí", "No"])

        # Preferencias y gustos
        music_preference = st.selectbox("¿Qué tipo de música prefieres?", 
                                        ["Pop", "Rock", "Clásica", "Jazz", "Electrónica", "Otro"])
        movie_genre = st.selectbox("¿Cuál es tu género de película favorito?", 
                                ["Acción", "Comedia", "Drama", "Ciencia ficción", "Terror", "Romance"])
        sport_preference = st.selectbox("¿Qué tipo de deporte prefieres?", 
                                        ["Fútbol", "Baloncesto", "Natación", "Correr", "No practico deportes"])
        food_preference = st.selectbox("¿Qué tipo de comida prefieres?", 
                                    ["Italiana", "Mexicana", "China", "Japonesa", "Americana"])

        # Estilo de vida
        sleep_hours = st.selectbox("¿Cuántas horas duermes en promedio por noche?", 
                                ["Menos de 5 horas", "5 a 7 horas", "7 a 9 horas", "Más de 9 horas"])
        exercise = st.selectbox("¿Haces ejercicio regularmente?", ["Sí", "No"])
        smoking = st.selectbox("¿Fumas?", ["Sí", "No"])
        alcohol = st.selectbox("¿Consumes alcohol?", ["Sí", "No"])
        healthiness = st.selectbox("¿Te consideras una persona saludable?", ["Sí", "No"])

        # Creencias y valores
        astrology_belief = st.selectbox("¿Crees en la astrología?", ["Si", "No"])
        religiosity = st.selectbox("¿Te consideras una persona religiosa?", ["Si", "No"])
        spirituality = st.selectbox("¿Te consideras una persona espiritual?", ["Si", "No"])
        family_importance = st.selectbox("¿Qué tan importante es para ti la familia?", 
                                        ["Muy importante", "Importante", "Poco importante", "Nada importante"])
        work_importance = st.selectbox("¿Qué tan importante es para ti el trabajo?", 
                                    ["Muy importante", "Importante", "Poco importante", "Nada importante"])

        # Habilidades y conocimientos
        languages = st.selectbox("¿Hablas más de un idioma?", ["Si", "No"])
        tech_savvy = st.selectbox("¿Te consideras una persona tecnológica?", ["Si", "No"])
        reading = st.selectbox("¿Te gusta leer?", ["Sí", "No"])

        # Vida sexual
        sexually_active = st.selectbox("¿Te consideras una persona sexualmente activa?", ["Sí", "No"])
        sexual_satisfaction = st.selectbox("¿Estás satisfecho/a con tu vida sexual?", 
                                        ["Muy satisfecho/a", "Satisfecho/a", "Neutral", "Insatisfecho/a", "Muy insatisfecho/a"])
        sexual_communication_importance = st.selectbox("¿Consideras importante la comunicación en tu vida sexual?", 
                                                    ["Muy importante", "Importante", "Poco importante", "Nada importante"])
        if st.button("Enviar"):
            user_data = {
                "full_name": full_name,
                "zodiac_sign": zodiac_sign,
                "age_range": age_range,
                "gender": gender,
                "education_level": education_level,
                "extroversion": extroversion,
                "punctuality": punctuality,
                "organization": organization,
                "risk_taking": risk_taking,
                "creativity": creativity,
                "music_preference": music_preference,
                "movie_genre": movie_genre,
                "sport_preference": sport_preference,
                "food_preference": food_preference,
                "sleep_hours": sleep_hours,
                "exercise": exercise,
                "smoking": smoking,
                "alcohol": alcohol,
                "healthiness": healthiness,
                "astrology_belief": astrology_belief,
                "religiosity": religiosity,
                "spirituality": spirituality,
                "family_importance": family_importance,
                "work_importance": work_importance,
                "languages": languages,
                "tech_savvy": tech_savvy,
                "reading": reading,
                "sexually_active": sexually_active,
                "sexual_satisfaction": sexual_satisfaction,
                "sexual_communication_importance": sexual_communication_importance,
            }
            df = pd.DataFrame([user_data])
            # Exportar a CSV agregandolos al final del archivo
            with open("user_data.csv", "a") as f:
                df.to_csv(f, header=f.tell()==0, index=False, encoding='utf-8')
            st.success("Datos enviados con éxito.")

    if choice == "Procesamiento de Datos Google":
        st.subheader("Procesamiento de Datos Google")
        st.write("Sube un archivo CSV exportado de Google Forms para su procesamiento.")

        uploaded_file = st.file_uploader("Elige un archivo CSV", type="csv")

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write("Datos del archivo CSV subido:")
            st.write(df.head())

            # Limpiar errores de unicode
            # df = clean_unicode_errors(df)

            # Mapea las columnas del CSV al formato esperado
            df = map_google_form_columns(df)
            st.write("Datos después de mapear las columnas:")
            st.write(df.head())

            # Añade un botón para guardar el archivo CSV procesado
            if st.button("Guardar CSV Procesado"):
                buffer = BytesIO()
                df.to_csv(buffer, index=False)
                buffer.seek(0)
                st.download_button(label="Descargar CSV Procesado", data=buffer, file_name="datos_procesados.csv", mime="text/csv")

    elif choice == "Análisis K-means":
        st.subheader("Análisis K-means")
        st.write("Sube el archivo CSV con los datos recolectados para aplicar K-means.")

        uploaded_file = st.file_uploader("Sube tu archivo CSV", type=["csv"])

        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file, encoding='utf-8')  # Specify encoding

            st.write("Vista previa de los datos:")
            st.dataframe(data.head())

            # Transformar variables categóricas y ordinales
            one_hot_columns = ['zodiac_sign', 'gender', 'music_preference', 'movie_genre', 'sport_preference', 'food_preference']
            data = pd.get_dummies(data, columns=one_hot_columns, drop_first=True)

            label_columns = ['education_level', 'extroversion', 'organization', 'risk_taking', 'creativity', 'exercise', 'smoking', 'alcohol', 'healthiness', 'astrology_belief', 'religiosity', 'spirituality', 'languages', 'tech_savvy', 'reading', 'sexually_active']
            label_encoder = LabelEncoder()
            for col in label_columns:
                data[col] = label_encoder.fit_transform(data[col])

            ordinal_columns = {
                'age_range': ['Menor a 18 años', '18 a 25 años', '26 a 35 años', '36 a 45 años', 'Más de 45 años'],
                'punctuality': ['Nada importante', 'Poco importante', 'Importante', 'Muy importante'],
                'sleep_hours': ['Menos de 5 horas', '5 a 7 horas', '7 a 9 horas', 'Más de 9 horas'],
                'family_importance': ['Nada importante', 'Poco importante', 'Importante', 'Muy importante'],
                'work_importance': ['Nada importante', 'Poco importante', 'Importante', 'Muy importante'],
                'sexual_satisfaction': ['Muy insatisfecho/a', 'Insatisfecho/a', 'Neutral', 'Satisfecho/a', 'Muy satisfecho/a'],
                'sexual_communication_importance': ['Nada importante', 'Poco importante', 'Importante', 'Muy importante']
            }

            for col, categories in ordinal_columns.items():
                ordinal_encoder = OrdinalEncoder(categories=[categories])
                data[col] = ordinal_encoder.fit_transform(data[[col]])

            st.write("Vista previa de los datos transformados:")
            st.dataframe(data.head())

            st.write("Selecciona las variables para el clustering:")

            # Initialize session state for selected columns
            if 'selected_columns' not in st.session_state:
                st.session_state.selected_columns = []

            # Checkbox to select all variables
            select_all = st.checkbox("Seleccionar todas las variables", value=len(st.session_state.selected_columns) == len(data.columns))

            # Update selected columns based on checkbox
            if select_all:
                st.session_state.selected_columns = data.columns.tolist()
            else:
                if len(st.session_state.selected_columns) == len(data.columns):
                    st.session_state.selected_columns = []

            # Multiselect for variable selection
            selected_columns = st.multiselect("Variables", data.columns.tolist(), default=st.session_state.selected_columns)

            # Update session state based on multiselect
            st.session_state.selected_columns = selected_columns

            # Update checkbox based on multiselect
            if len(selected_columns) == len(data.columns):
                select_all = True
            else:
                select_all = False

            # Initialize session state for clustering results
            if 'clusters' not in st.session_state:
                st.session_state.clusters = None
            if 'num_clusters' not in st.session_state:
                st.session_state.num_clusters = 3
            if 'centers' not in st.session_state:
                st.session_state.centers = None
            
            # Seleccion las variables en los ejes
            st.write("Selecciona las variables para los ejes X y Y:")
            if len(selected_columns) > 2:
                x_axis = st.selectbox("Eje X", selected_columns, index=0)
                y_axis = st.selectbox("Eje Y", selected_columns, index=1)
            # Slider for number of clusters
            num_clusters = st.slider("Número de clusters", min_value=2, max_value=10, value=st.session_state.num_clusters)
            st.session_state.num_clusters = num_clusters

            # Button to apply K-means
            aplicar_kmeans = st.button("Aplicar K-means")
            if aplicar_kmeans:
                if selected_columns:
                    try:
                        kmeans = KMeans(n_clusters=num_clusters)
                        data['Cluster'] = kmeans.fit_predict(data[selected_columns])
                        st.session_state.clusters = data['Cluster']
                        st.session_state.centers = kmeans.cluster_centers_
                    except Exception as e:
                        st.error(f"Error al aplicar K-means: {e}")
                else:
                    st.warning("Por favor, selecciona al menos una variable para el clustering.")

            # Display clustering results if they exist in session state
            if st.session_state.clusters is not None and st.session_state.centers is not None:
                data['Cluster'] = st.session_state.clusters
                st.write("Resultados de K-means:")
                st.dataframe(data.head())

                # Crear las dos figuras
                fig1, ax1 = plt.subplots()
                fig2, ax2 = plt.subplots()

                # Graficar los puntos de datos y los centros de clústeres en fig1
                scatter = ax1.scatter(data[x_axis], data[y_axis], c=data['Cluster'], cmap='viridis')
                ax1.set_xlabel(x_axis)
                ax1.set_ylabel(y_axis)

                if 'centers' in st.session_state:
                    centers = st.session_state.centers
                    ax1.scatter(centers[:, data.columns.get_loc(x_axis)], centers[:, data.columns.get_loc(y_axis)], c='red', s=100, alpha=0.5, marker='X', label='Centroides')
                    ax1.legend()

                # Mostrar la gráfica en Streamlit
                st.pyplot(fig1)

                # Gráfico del método del codo en fig2
                wcss = []
                for i in range(1, 11):
                    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
                    kmeans.fit(data[selected_columns])
                    wcss.append(kmeans.inertia_)

                ax2.plot(range(1, 11), wcss)
                ax2.set_title('Método del Codo')
                ax2.set_xlabel('Número de clústeres')
                ax2.set_ylabel('WCSS')

                # Mostrar la gráfica en Streamlit
                st.pyplot(fig2)

                st.write("Exportar resultados:")
                if st.button("Exportar a PDF"):
                    pdf = FPDF(orientation='L', unit='mm', format='A4')
                    pdf.add_page()
                    pdf.set_font("Arial", size=12)
                    pdf.cell(200, 10, txt="Resultados de K-means", ln=True, align='C')
                    pdf.ln(20)

                    # Guardar las gráficas como archivos temporales
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile1:
                        fig1.savefig(tmpfile1.name, format="png")
                        pdf.image(tmpfile1.name, x=10, y=30, w=180)

                    pdf.add_page()
                    pdf.cell(200, 10, txt="Gráfico del Método del Codo", ln=True, align='C')
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

if __name__ == "__main__":
    main()

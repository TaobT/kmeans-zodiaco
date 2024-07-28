from io import BytesIO
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from utils.clustering import apply_kmeans
from utils.export import export_to_excel
from fpdf import FPDF
import tempfile

def main():
    uploaded_file = None  # Initialize uploaded_file
    st.title("Aplicación de Clustering con K-means")

    menu = ["Inicio", "Cuestionario", "Análisis K-means"]
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
                                ["Menos de 18 años", "18-25 años", "26-35 años", "36-45 años", "Más de 45 años"])
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
                                ["Menos de 5 horas", "5-7 horas", "7-9 horas", "Más de 9 horas"])
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
        sexual_partners = st.selectbox("¿Cuántas parejas sexuales has tenido en el último año?", 
                                    ["Ninguna", "1", "2-3", "4-5", "Más de 5"])
        sexual_frequency = st.selectbox("¿Con qué frecuencia tienes relaciones sexuales?", 
                                        ["Diariamente", "Semanalmente", "Mensualmente", "Raramente", "Nunca"])
        sexual_satisfaction = st.selectbox("¿Estás satisfecho/a con tu vida sexual?", 
                                        ["Muy satisfecho/a", "Satisfecho/a", "Neutral", "Insatisfecho/a", "Muy insatisfecho/a"])
        sexual_communication_importance = st.selectbox("¿Consideras importante la comunicación en tu vida sexual?", 
                                                    ["Muy importante", "Importante", "Poco importante", "Nada importante"])
        sexual_comfort = st.selectbox("¿Te sientes cómodo/a hablando de temas sexuales con tu(s) pareja(s)?", ["Sí", "No"])
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
                "sexual_partners": sexual_partners,
                "sexual_frequency": sexual_frequency,
                "sexual_satisfaction": sexual_satisfaction,
                "sexual_communication_importance": sexual_communication_importance,
                "sexual_comfort": sexual_comfort,
            }
            df = pd.DataFrame([user_data])
            # Exportar a CSV agregandolos al final del archivo
            with open("user_data.csv", "a") as f:
                df.to_csv(f, header=f.tell()==0, index=False, encoding='latin1')
            st.success("Datos enviados con éxito.")

    elif choice == "Análisis K-means":
        st.subheader("Análisis K-means")
        st.write("Sube el archivo CSV con los datos recolectados para aplicar K-means.")

        uploaded_file = st.file_uploader("Sube tu archivo CSV", type=["csv"])

        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file, encoding='latin1')  # Specify encoding

            st.write("Vista previa de los datos:")
            st.dataframe(data.head())

            # Transformar variables categóricas y ordinales
            one_hot_columns = ['zodiac_sign', 'gender', 'music_preference', 'movie_genre', 'sport_preference', 'food_preference']
            data = pd.get_dummies(data, columns=one_hot_columns, drop_first=True)

            label_columns = ['education_level', 'extroversion', 'organization', 'risk_taking', 'creativity', 'exercise', 'smoking', 'alcohol', 'healthiness', 'astrology_belief', 'religiosity', 'spirituality', 'languages', 'tech_savvy', 'reading', 'sexually_active', 'sexual_comfort']
            label_encoder = LabelEncoder()
            for col in label_columns:
                data[col] = label_encoder.fit_transform(data[col])

            ordinal_columns = {
                'age_range': ['Menos de 18 años', '18-25 años', '26-35 años', '36-45 años', 'Más de 45 años'],
                'punctuality': ['Nada importante', 'Poco importante', 'Importante', 'Muy importante'],
                'sleep_hours': ['Menos de 5 horas', '5-7 horas', '7-9 horas', 'Más de 9 horas'],
                'family_importance': ['Nada importante', 'Poco importante', 'Importante', 'Muy importante'],
                'work_importance': ['Nada importante', 'Poco importante', 'Importante', 'Muy importante'],
                'sexual_partners': ['Ninguna', '1', '2-3', '4-5', 'Más de 5'],
                'sexual_frequency': ['Nunca', 'Raramente', 'Mensualmente', 'Semanalmente', 'Diariamente'],
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

            # Slider for number of clusters
            num_clusters = st.slider("Número de clusters", min_value=2, max_value=10, value=st.session_state.num_clusters)
            st.session_state.num_clusters = num_clusters

            # Button to apply K-means
            aplicar_kmeans = st.button("Aplicar K-means")
            if aplicar_kmeans:
                if selected_columns:
                    try:
                        kmeans = KMeans(n_clusters=num_clusters)
                        clusters = kmeans.fit_predict(data[selected_columns])
                        data['Cluster'] = clusters
                        st.session_state.clusters = clusters
                    except Exception as e:
                        st.error(f"Error al aplicar K-means: {e}")
                else:
                    st.warning("Por favor, selecciona al menos una variable para el clustering.")

            # Display clustering results if they exist in session state
            if st.session_state.clusters is not None:
                data['Cluster'] = st.session_state.clusters
                st.write("Resultados de K-means:")
                st.dataframe(data.head())

                fig, ax = plt.subplots()
                sns.scatterplot(data=data, x=selected_columns[0], y=selected_columns[1], hue='Cluster', palette='viridis', ax=ax)
                plt.title("Clusters")
                st.pyplot(fig)

                st.write("Exportar resultados:")
                if st.button("Exportar a PDF"):
                    pdf = FPDF(orientation='L', unit='mm', format='A4')  # Set orientation to Landscape
                    pdf.add_page()
                    pdf.set_font("Arial", size=12)
                    pdf.cell(200, 10, txt="Resultados de K-means", ln=True, align='C')
                    pdf.ln(20)
                    
                    # Save plot to a temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                        plt.savefig(tmpfile.name, format="png")
                        pdf.image(tmpfile.name, x=10, y=30, w=180)
                    
                    pdf.ln(100)  # Adjust this value based on the height of your image
                    pdf.add_page()  # Add a new page for the table
                    pdf.set_font("Arial", size=8)  # Reduce font size
                    
                    # Calculate column width
                    page_width = pdf.w - 20  # Page width minus margins
                    col_width = page_width / (len(data.columns) + 1)  # +1 for the index column
                    
                    # Add table header
                    pdf.cell(col_width, 10, 'Index', 1)
                    for col in data.columns:
                        pdf.cell(col_width, 10, col, 1)
                    pdf.ln()
                    
                    # Add table rows
                    row_height = 10
                    max_rows_per_page = int((pdf.h - 20) / row_height) - 2  # Adjust for margins and header
                    for i in range(len(data)):
                        if i % max_rows_per_page == 0 and i != 0:
                            pdf.add_page()
                            pdf.cell(col_width, 10, 'Index', 1)
                            for col in data.columns:
                                pdf.cell(col_width, 10, col, 1)
                            pdf.ln()
                        pdf.cell(col_width, 10, str(i), 1)
                        for col in data.columns:
                            pdf.cell(col_width, 10, str(data.iloc[i][col]), 1)
                        pdf.ln()
                    
                    pdf_output = "resultados.pdf"
                    pdf.output(pdf_output)
                    st.success("Resultados exportados a resultados.pdf")

if __name__ == "__main__":
    main()

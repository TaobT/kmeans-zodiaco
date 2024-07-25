import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from utils.clustering import apply_kmeans
from utils.export import export_to_excel, export_to_pdf

def main():
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
        astrology_belief = st.selectbox("¿Crees en la astrología?", ["Sí", "No"])
        religiosity = st.selectbox("¿Te consideras una persona religiosa?", ["Sí", "No"])
        spirituality = st.selectbox("¿Te consideras una persona espiritual?", ["Sí", "No"])
        family_importance = st.selectbox("¿Qué tan importante es para ti la familia?", 
                                        ["Muy importante", "Importante", "Poco importante", "Nada importante"])
        work_importance = st.selectbox("¿Qué tan importante es para ti el trabajo?", 
                                    ["Muy importante", "Importante", "Poco importante", "Nada importante"])

        # Habilidades y conocimientos
        languages = st.selectbox("¿Hablas más de un idioma?", ["Sí", "No"])
        tech_savvy = st.selectbox("¿Te consideras una persona tecnológica?", ["Sí", "No"])
        reading = st.selectbox("¿Te gusta leer?", ["Sí", "No"])

        # Vida sexual
        sexually_active = st.selectbox("¿Te consideras una persona sexualmente activa?", ["Sí", "No"])
        sexual_partners = st.selectbox("¿Cuántas parejas sexuales has tenido en el último año?", 
                                    ["Ninguna", "1", "2-3", "4-5", "Más de 5"])
        sexual_frequency = st.selectbox("¿Con qué frecuencia tienes relaciones sexuales?", 
                                        ["Diariamente", "Semanalmente", "Mensualmente", "Raramente", "Nunca"])
        sexual_satisfaction = st.selectbox("¿Estás satisfecho/a con tu vida sexual?", 
                                        ["Muy satisfecho/a", "Satisfecho/a", "Neutral", "Insatisfecho/a", "Muy insatisfecho/a"])
        contraceptive_use = st.selectbox("¿Usas métodos anticonceptivos durante las relaciones sexuales?", 
                                        ["Siempre", "A menudo", "A veces", "Raramente", "Nunca"])
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
            "contraceptive_use": contraceptive_use,
            "sexual_communication_importance": sexual_communication_importance,
            "sexual_comfort": sexual_comfort,
        }
        df = pd.DataFrame([user_data])
        st.write(df)
        st.success("Datos enviados con éxito.")

    elif choice == "Análisis K-means":
        st.subheader("Análisis K-means")
    st.write("Selecciona las variables de interés y aplica el algoritmo K-means.")

    uploaded_file = st.file_uploader("Cargar archivo CSV", type=["csv"])
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.write("Datos cargados:", data.head())

        if st.checkbox("Mostrar todas las columnas"):
            st.write(data.columns.tolist())

        selected_columns = st.multiselect("Selecciona las variables para el clustering", data.columns.tolist())
        if st.button("Aplicar K-means"):
            if selected_columns:
                clusters, centroids = apply_kmeans(data[selected_columns])
                st.write("Clusters generados:")
                st.write(clusters)

                # Visualización de resultados
                st.write("Visualización de los clusters:")
                sns.pairplot(data[selected_columns + ['cluster']])
                st.pyplot()

                # Exportar resultados
                if st.button("Exportar resultados a Excel"):
                    export_to_excel(data, clusters, "resultados.xlsx")
                    st.success("Resultados exportados a resultados.xlsx")

                if st.button("Exportar resultados a PDF"):
                    export_to_pdf(data, clusters, "resultados.pdf")
                    st.success("Resultados exportados a resultados.pdf")
            else:
                st.warning("Por favor, selecciona al menos una variable.")

if __name__ == "__main__":
    main()

# Demarrage

cd groupe-02-C2-JVX

.\venv\Scripts\activate

# Lancer l'interface Streamlit :

streamlit run src/ui/app.py


# Générer les données et indexer :

python data/generate_samples.py

python src/main.py ingest --source data/samples/
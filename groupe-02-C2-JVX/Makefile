.PHONY: install generate-data ingest ui test evaluate clean help

help:
	@echo "FinRAG — Système RAG pour l'Analyse Financière"
	@echo ""
	@echo "Commandes disponibles:"
	@echo "  make install        Installe les dépendances Python"
	@echo "  make generate-data  Génère les données d'exemple"
	@echo "  make ingest         Indexe les documents dans ChromaDB"
	@echo "  make ui             Lance l'interface Streamlit"
	@echo "  make test           Lance les tests unitaires"
	@echo "  make evaluate       Lance l'évaluation RAGAS"
	@echo "  make clean          Supprime les données générées"

install:
	pip install -r requirements.txt

generate-data:
	python data/generate_samples.py

ingest:
	python src/main.py ingest --source data/samples/

ui:
	streamlit run src/ui/app.py

test:
	pytest tests/ -v --tb=short

evaluate:
	python src/main.py evaluate

clean:
	rm -rf data/chroma_db/*
	rm -rf data/embedding_cache/*
	rm -rf logs/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true

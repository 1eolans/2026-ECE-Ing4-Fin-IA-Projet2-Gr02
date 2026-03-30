# -- Ce module definit la configuration centralisee du systeme FinRAG.
# -- Il utilise pydantic-settings pour charger les parametres depuis des variables
# -- d'environnement ou un fichier .env, avec validation automatique des types.
# -- Cela permet de centraliser tous les reglages (cles API, parametres RAG,
# -- chemins de stockage, etc.) en un seul endroit, facilitant la maintenance.
"""Configuration centralisée du système FinRAG via pydantic-settings."""

# -- Import de BaseSettings : classe de base qui lit automatiquement les variables
# -- d'environnement et le fichier .env pour hydrater les champs de configuration.
# -- Import de SettingsConfigDict : dictionnaire type pour configurer le comportement
# -- de BaseSettings (chemin du .env, encodage, sensibilite a la casse, etc.).
from pydantic_settings import BaseSettings, SettingsConfigDict

# -- Import de Field : permet de definir des valeurs par defaut, des descriptions
# -- et des contraintes de validation pour chaque parametre de configuration.
from pydantic import Field

# -- Import de Path : classe utilitaire pour manipuler les chemins de fichiers
# -- de maniere portable (fonctionne sous Windows, Linux, macOS).
from pathlib import Path


# -- Classe principale de configuration. Elle herite de BaseSettings, ce qui
# -- signifie que chaque attribut de classe peut etre surcharge par une variable
# -- d'environnement portant le meme nom (insensible a la casse).
# -- Exemple : l'attribut OPENAI_API_KEY peut etre defini via la variable
# -- d'environnement OPENAI_API_KEY ou dans le fichier .env.
class Settings(BaseSettings):
    """Configuration globale du système FinRAG.

    Toutes les valeurs peuvent être surchargées via variables d'environnement
    ou le fichier .env à la racine du projet.
    """

    # -- FIX ELEVE : dans pydantic-settings v2, l'ancienne syntaxe "class Config"
    # -- est deprecie. On utilise desormais model_config avec SettingsConfigDict
    # -- pour definir le comportement de chargement de la configuration.
    # FIX ÉLEVÉ : utilisation de model_config au lieu de class Config (déprécié en pydantic-settings v2)
    model_config = SettingsConfigDict(
        # -- Chemin vers le fichier .env contenant les variables d'environnement.
        # -- Ce fichier est lu automatiquement au demarrage pour hydrater les champs.
        env_file=".env",
        # -- Encodage du fichier .env. UTF-8 est le standard pour supporter
        # -- les caracteres speciaux (accents, etc.) dans les valeurs.
        env_file_encoding="utf-8",
        # -- Si False, les noms de variables d'environnement sont insensibles
        # -- a la casse (ex: openai_api_key == OPENAI_API_KEY).
        case_sensitive=False,
        # -- "ignore" signifie que les variables d'environnement presentes dans
        # -- le .env mais non declarees dans cette classe seront ignorees
        # -- silencieusement, sans lever d'erreur de validation.
        extra="ignore",       # ignore les variables d'env inconnues
    )

    # =========================================================================
    # === LLM (Large Language Model) ==========================================
    # =========================================================================
    # -- Cette section regroupe les parametres lies aux modeles de langage
    # -- utilises pour la generation de texte et le calcul d'embeddings.

    # -- Cle API OpenAI : necessaire pour utiliser les modeles d'embeddings
    # -- d'OpenAI (ex: text-embedding-3-small). Si vide, le systeme bascule
    # -- automatiquement sur un modele local (voir FALLBACK_EMBEDDING_MODEL).
    OPENAI_API_KEY: str = Field(default="", description="Clé API OpenAI pour les embeddings")

    # -- Cle API Anthropic : necessaire pour utiliser les modeles de generation
    # -- de texte d'Anthropic (Claude). C'est le LLM principal qui produit les
    # -- reponses aux questions des utilisateurs a partir des documents retrouves.
    ANTHROPIC_API_KEY: str = Field(default="", description="Clé API Anthropic pour la génération")

    # -- Identifiant du modele LLM utilise pour la generation de reponses.
    # -- Claude Sonnet 4 est un modele performant et rapide, adapte aux taches
    # -- de question-reponse sur des documents financiers.
    LLM_MODEL: str = Field(default="claude-sonnet-4-20250514", description="Modèle LLM pour la génération")

    # -- Modele d'embeddings OpenAI utilise pour convertir le texte en vecteurs
    # -- numeriques. "text-embedding-3-small" offre un bon compromis entre
    # -- qualite des embeddings et cout d'utilisation de l'API.
    EMBEDDING_MODEL: str = Field(default="text-embedding-3-small", description="Modèle d'embeddings OpenAI")

    # -- Modele d'embeddings local utilise en cas d'indisponibilite de la cle
    # -- OpenAI. "all-MiniLM-L6-v2" est un modele Sentence-Transformers leger
    # -- qui tourne en local sans necessite de connexion API, mais avec une
    # -- qualite d'embeddings legerement inferieure.
    FALLBACK_EMBEDDING_MODEL: str = Field(
        default="all-MiniLM-L6-v2",
        description="Modèle d'embeddings local (fallback sans API OpenAI)",
    )

    # =========================================================================
    # === RAG (Retrieval-Augmented Generation) ================================
    # =========================================================================
    # -- Cette section contient les hyperparametres du pipeline RAG, qui combine
    # -- la recherche de documents pertinents (retrieval) avec la generation de
    # -- reponses par le LLM (generation). Ces parametres influencent directement
    # -- la qualite et la pertinence des reponses produites.

    # -- Taille maximale de chaque chunk (fragment de document) en nombre de tokens.
    # -- 512 tokens est un bon compromis : assez grand pour conserver le contexte
    # -- semantique, assez petit pour permettre une recherche precise.
    CHUNK_SIZE: int = Field(default=512, description="Taille des chunks en tokens")

    # -- Chevauchement (overlap) entre deux chunks consecutifs, en nombre de tokens.
    # -- Fixe a ~10% de CHUNK_SIZE (51 tokens). Ce chevauchement assure qu'une
    # -- information situee a la frontiere de deux chunks ne soit pas perdue,
    # -- car elle apparaitra dans les deux fragments adjacents.
    CHUNK_OVERLAP: int = Field(default=51, description="Overlap entre chunks (10%)")

    # -- Nombre de chunks recuperes apres la fusion RRF (Reciprocal Rank Fusion).
    # -- RRF combine les resultats des recherches dense et sparse (BM25) pour
    # -- produire un classement unique. 10 chunks offrent un bon pool de candidats
    # -- avant l'etape de re-ranking.
    TOP_K_RETRIEVAL: int = Field(default=10, description="Nombre de chunks récupérés après RRF")

    # -- Nombre de chunks conserves apres l'etape de re-ranking.
    # -- Le re-ranking reordonne les TOP_K_RETRIEVAL chunks en utilisant un modele
    # -- plus precis (cross-encoder) pour ne garder que les 5 plus pertinents.
    # -- Ces 5 chunks sont ensuite injectes dans le prompt du LLM comme contexte.
    TOP_K_RERANK: int = Field(default=5, description="Nombre de chunks après re-ranking")

    # -- Coefficient alpha pour la recherche hybride (fusion dense + sparse).
    # -- alpha = 0.0 : uniquement recherche sparse (BM25, mots-cles exacts).
    # -- alpha = 1.0 : uniquement recherche dense (embeddings, similarite semantique).
    # -- alpha = 0.7 : favorise la recherche dense (70%) tout en integrant la
    # -- recherche par mots-cles (30%), ce qui donne de bons resultats en pratique.
    HYBRID_ALPHA: float = Field(default=0.7, description="Poids dense vs sparse (0=sparse, 1=dense)")

    # -- Facteur de decroissance temporelle : penalise les documents anciens
    # -- en reduisant leur score de pertinence proportionnellement a leur age.
    # -- Une valeur de 0.1 applique une penalite moderee, privilegiant les
    # -- documents recents tout en conservant les anciens s'ils sont tres pertinents.
    TIME_DECAY_FACTOR: float = Field(default=0.1, description="Facteur de décroissance temporelle")

    # -- Nombre de resultats recuperes par la recherche BM25 (sparse) avant
    # -- la fusion RRF. 20 candidats BM25 sont combines avec les 20 candidats
    # -- denses pour produire le classement fusionne final.
    BM25_TOP_K: int = Field(default=20, description="Top-k pour BM25 avant fusion")

    # -- Nombre de resultats recuperes par la recherche dense (embeddings)
    # -- avant la fusion RRF. Symetrique a BM25_TOP_K pour equilibrer les
    # -- deux sources de resultats lors de la fusion.
    DENSE_TOP_K: int = Field(default=20, description="Top-k pour dense retrieval avant fusion")

    # =========================================================================
    # === BM25 (Best Matching 25) =============================================
    # =========================================================================
    # -- Parametres specifiques a l'algorithme BM25, qui effectue une recherche
    # -- par mots-cles (recherche sparse / lexicale). BM25 est complementaire
    # -- a la recherche dense basee sur les embeddings.

    # -- FIX ELEVE : limite le nombre de documents charges en memoire pour
    # -- l'index BM25. Sans ce plafond, un corpus tres volumineux peut provoquer
    # -- un depassement de memoire (Out Of Memory / OOM). 10 000 documents
    # -- representent un bon compromis entre couverture et consommation memoire.
    # FIX ÉLEVÉ : plafond mémoire BM25 pour éviter OOM sur gros corpus
    BM25_MAX_DOCS: int = Field(default=10_000, description="Nombre max de docs chargés en RAM pour BM25")

    # =========================================================================
    # === ChromaDB (Base de donnees vectorielle) ==============================
    # =========================================================================
    # -- ChromaDB est la base de donnees vectorielle utilisee pour stocker et
    # -- rechercher les embeddings des chunks de documents. Elle permet une
    # -- recherche par similarite cosinus efficace et persistante sur disque.

    # -- Repertoire ou ChromaDB persiste ses donnees sur disque.
    # -- Par defaut, les donnees sont stockees dans ./data/chroma_db relatif
    # -- au repertoire de travail. Ce dossier contient les fichiers SQLite
    # -- et les index vectoriels (fichiers .bin).
    CHROMA_PERSIST_DIR: str = Field(default="./data/chroma_db", description="Répertoire de persistance ChromaDB")

    # -- Prefixe utilise pour nommer les collections ChromaDB.
    # -- Chaque collection stocke les embeddings d'un type de document.
    # -- Le prefixe "finrag" permet de distinguer les collections de ce projet
    # -- d'eventuelles autres collections dans la meme instance ChromaDB.
    COLLECTION_PREFIX: str = Field(default="finrag", description="Préfixe des collections ChromaDB")

    # =========================================================================
    # === UI (Interface utilisateur) ==========================================
    # =========================================================================
    # -- Parametres lies a l'interface utilisateur (Streamlit ou autre).

    # -- Nombre maximum de messages conserves dans l'historique de conversation.
    # -- Au-dela de 50 messages, les plus anciens sont supprimes pour eviter
    # -- une consommation excessive de memoire et garder le contexte pertinent.
    MAX_HISTORY_MESSAGES: int = Field(default=50, description="Nombre max de messages dans l'historique")

    # -- Active ou desactive le streaming des reponses du LLM.
    # -- Si True, les reponses s'affichent progressivement (mot par mot) dans
    # -- l'interface, offrant une meilleure experience utilisateur.
    # -- Si False, la reponse complete est affichee d'un coup apres generation.
    STREAM_RESPONSE: bool = Field(default=True, description="Active le streaming des réponses")

    # =========================================================================
    # === Performance =========================================================
    # =========================================================================
    # -- Parametres lies aux performances et a l'optimisation du systeme.

    # -- Delai d'attente maximal (en secondes) pour les requetes HTTP vers les
    # -- APIs externes (OpenAI, Anthropic). Si une requete depasse 30 secondes,
    # -- elle est annulee et une erreur est levee pour eviter de bloquer le systeme.
    REQUEST_TIMEOUT: int = Field(default=30, description="Timeout des requêtes en secondes")

    # -- Repertoire de cache pour les embeddings deja calcules.
    # -- Stocker les embeddings en cache evite de recalculer (et donc de payer)
    # -- les memes embeddings plusieurs fois pour un meme texte. Cela accelere
    # -- considerablement le traitement lors de re-indexations partielles.
    EMBEDDING_CACHE_DIR: str = Field(default="./data/embedding_cache", description="Cache des embeddings")

    # =========================================================================
    # === Logging (Journalisation) ============================================
    # =========================================================================
    # -- Parametres de journalisation pour le suivi et le debogage du systeme.

    # -- Niveau de log : controle la verbosity des messages journalises.
    # -- DEBUG : tous les messages (tres verbeux, utile pour le debogage).
    # -- INFO : messages informatifs + warnings + erreurs (niveau recommande).
    # -- WARNING : uniquement les avertissements et erreurs.
    # -- ERROR : uniquement les erreurs critiques.
    LOG_LEVEL: str = Field(default="INFO", description="Niveau de log (DEBUG, INFO, WARNING, ERROR)")

    # -- Chemin du fichier de log ou sont ecrits les messages journalises.
    # -- Le dossier ./logs/ doit exister ou etre cree au demarrage de l'application.
    LOG_FILE: str = Field(default="./logs/finrag.log", description="Fichier de log")

    # =========================================================================
    # === Proprietes calculees ================================================
    # =========================================================================
    # -- Ces proprietes (decorateur @property) ne sont pas des parametres
    # -- configurables mais des valeurs derivees, calculees dynamiquement a
    # -- partir des champs ci-dessus. Elles simplifient les verifications
    # -- recurrentes dans le reste du code.

    @property
    def use_openai_embeddings(self) -> bool:
        # -- Verifie si la cle API OpenAI est renseignee (non vide).
        # -- Retourne True si la cle est presente, ce qui signifie que le systeme
        # -- peut utiliser le modele d'embeddings OpenAI (EMBEDDING_MODEL).
        # -- Retourne False si la cle est absente, auquel cas le systeme doit
        # -- basculer sur le modele local (FALLBACK_EMBEDDING_MODEL).
        """Retourne True si la clé OpenAI est disponible."""
        return bool(self.OPENAI_API_KEY)

    @property
    def use_anthropic(self) -> bool:
        # -- Verifie si la cle API Anthropic est renseignee (non vide).
        # -- Retourne True si la cle est presente, ce qui signifie que le systeme
        # -- peut utiliser le LLM Anthropic (Claude) pour generer des reponses.
        # -- Si False, le systeme ne pourra pas generer de reponses et devra
        # -- signaler l'absence de configuration a l'utilisateur.
        """Retourne True si la clé Anthropic est disponible."""
        return bool(self.ANTHROPIC_API_KEY)

    @property
    def chroma_persist_path(self) -> Path:
        # -- Convertit le chemin relatif CHROMA_PERSIST_DIR en chemin absolu
        # -- en utilisant Path.resolve(). Cela garantit que le chemin est toujours
        # -- correct quel que soit le repertoire de travail courant du processus.
        # -- Utile pour ChromaDB qui a besoin d'un chemin absolu fiable pour
        # -- persister ses donnees de maniere coherente entre les executions.
        """Retourne le chemin absolu du répertoire ChromaDB."""
        return Path(self.CHROMA_PERSIST_DIR).resolve()


# -- Instance unique (singleton) de la configuration, creee au chargement du module.
# -- Tous les autres modules importent cette instance pour acceder aux parametres :
# --   from config import settings
# -- Cela garantit qu'une seule instance de configuration existe dans tout le projet,
# -- avec des valeurs coherentes chargees une seule fois depuis l'environnement.
settings = Settings()

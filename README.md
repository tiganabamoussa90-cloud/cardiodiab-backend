# CardioDiab Predict — Backend

API REST (FastAPI + Python) de la plateforme prédictive CardioDiab. Gère
l'authentification, la gestion des dossiers patients, les consultations
cliniques, l'inférence des modèles de Machine Learning (risque
cardiovasculaire et risque diabétique) et leur explicabilité via SHAP.

## Stack technique

- **Framework** : FastAPI (Python), serveur ASGI Uvicorn
- **Base de données** : MySQL (via `mysql-connector-python`)
- **Authentification** : JWT (PyJWT) + hachage des mots de passe (bcrypt)
- **Machine Learning** : scikit-learn (régression logistique — risque
  cardiovasculaire), LightGBM (risque diabétique)
- **Explicabilité** : SHAP (`shap.Explainer` sur `predict_proba` pour le
  modèle cardio, `shap.TreeExplainer` pour le modèle diabète)
- **Validation des schémas** : Pydantic

## Démarrage en local

```bash
pip install -r requirements.txt --break-system-packages
cp .env.example .env   # renseigner les valeurs locales
uvicorn main:app --reload
```

L'API est alors disponible sur `http://127.0.0.1:8000`, avec la
documentation interactive Swagger sur `http://127.0.0.1:8000/docs`.

## Variables d'environnement

| Variable | Rôle |
|---|---|
| `DB_HOST`, `DB_USER`, `DB_PASSWORD`, `DB_NAME`, `DB_PORT` | Connexion à la base MySQL |
| `SECRET_KEY` | Clé de signature des badges JWT (générer une valeur forte en production, jamais réutiliser celle de développement) |

Ces variables sont chargées via `python-dotenv` (`load_dotenv()` dans
`database.py` et `auth.py`) et ne doivent jamais être commitées — seul
`.env.example` (sans valeurs réelles) est versionné.

## Arborescence

```
main.py                 Point d'entrée FastAPI, toutes les routes de l'API
auth.py                 Fabrication/vérification des badges JWT, hachage bcrypt
database.py             Connexion MySQL (get_db_connection)
schemas_pydantic.py      Modèles de validation des requêtes/réponses
models/
  pipeline_cardio_reglog.pkl     Pipeline scikit-learn (régression logistique)
  pipeline_diabete_model.pkl     Pipeline LightGBM
```

## Rôles et espaces de l'API

L'authentification (`POST /auth/login`) retourne un badge JWT contenant
l'identifiant et le rôle de l'utilisateur. Pour un médecin, le rôle renvoyé
est en réalité sa **spécialité** (`CARDIOLOGUE` ou `DIABETOLOGUE` — seules
valeurs valides, il n'existe pas de rôle générique), ce qui détermine
automatiquement quel modèle IA est sollicité et quelles données sont
restituées.

| Espace | Préfixe de routes | Protection |
|---|---|---|
| Authentification | `/auth/*` | Public |
| Agent d'admission | `/agent/*` | `verifier_badge_agent` |
| Médecin | `/medecin/*` | `verifier_badge_medecin` |
| Patient | `/patient/*` | `verifier_badge_patient` |
| Administrateur | `/admin/*` | `verifier_badge_admin` |

### Flux principal d'une consultation

1. L'agent d'admission crée le dossier patient (`POST /agent/creer-patient`),
   génère son IPP (Identifiant Permanent du Patient) et son mot de passe
   temporaire.
2. Le médecin lie le patient à sa liste via l'IPP
   (`POST /medecin/lier-patient`).
3. Le médecin enregistre une consultation (`POST /medecin/consultation`) :
   calcul de l'IMC, préparation des variables cliniques selon la spécialité,
   inférence du modèle correspondant, calcul et persistance de la matrice
   SHAP associée.
4. Le patient consulte l'évolution de ses scores et télécharge son rapport
   (`GET /patient/dashboard/*`, `GET /patient/consultation/{id}/rapport`).

## Explicabilité SHAP

- **Modèle cardiologie** (régression logistique) : l'explainer est construit
  sur `model_lr.predict_proba(X)[:, 1]` plutôt que sur la classe brute
  prédite, afin que les contributions SHAP restent exprimées dans la même
  unité (probabilité de risque) que le score affiché au médecin.
- **Modèle diabétologie** (LightGBM) : `shap.TreeExplainer`, calcul exact et
  rapide propre aux modèles à base d'arbres.
- Les variables catégorielles étant encodées en one-hot par le
  `ColumnTransformer` du pipeline, une fonction dédiée
  (`regrouper_valeurs_shap`) recompose les contributions par variable
  clinique d'origine, pour rester lisible côté médecin.

## Sécurité

- Mots de passe hachés avec bcrypt, jamais stockés en clair.
- Sessions sans état via JWT, expiration après 2 heures.
- Contrôle d'accès par rôle sur chaque route sensible (`Depends(...)`).
- Isolation des données par utilisateur (un patient ne peut accéder qu'à ses
  propres consultations, un médecin qu'aux patients avec lien actif).
- Journalisation de chaque action sensible dans `audit_log` (utilisateur,
  action, détails, adresse IP, horodatage).

## Déploiement

Backend déployé sur **Railway** (conteneur Python persistant — préféré à un
hébergement serverless du fait du poids des dépendances scientifiques :
scikit-learn, lightgbm, shap, pandas). Commande de démarrage :

```
uvicorn main:app --host 0.0.0.0 --port $PORT
```

Base de données MySQL hébergée séparément sur **Clever Cloud**. Le CORS
(`main.py`) doit inclure le domaine du frontend Vercel en production, en plus
de `http://localhost:5173` pour le développement local.

## Points connus en suspens

- `enregistrer_audit()` utilise la colonne `ip_adress` (faute de frappe),
  alors que le schéma SQL déclare `ip_address` dans certaines versions —
  vérifier la cohérence exacte entre le code et le schéma actuellement en
  base avant toute nouvelle installation.
- `PatientCreateRequest` déclare `mot_de_passe` et `ipp` comme champs requis,
  alors qu'ils sont générés côté serveur dans `creer_dossier_patient` — à
  aligner (champs optionnels/supprimés côté schéma, et l'INSERT doit utiliser
  la variable `ipp` générée plutôt que `data.ipp`).
- `GET /admin/medecins/en-attente` contient une virgule manquante dans le
  `SELECT` (`m.code_inpe u.email`) — erreur de syntaxe SQL à corriger.
- `GET /admin/agents/en-attente` sélectionne des colonnes inexistantes sur
  `agent_admission` (`nom`/`prenom` au lieu de `nom_agent`/`prenom_agent`,
  pas de colonne `specialite` pour les agents).
- Deux fonctions Python homonymes `valider_compte_medecin` (une pour les
  médecins, une pour les agents) — à renommer pour lever toute ambiguïté.

## Bibliothèques principales

```
fastapi, uvicorn[standard], pydantic[email], mysql-connector-python,
bcrypt, pyjwt[crypto], python-multipart, numpy, pandas, scikit-learn,
lightgbm, shap, python-dotenv
```

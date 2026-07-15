from database import get_db_connection
import uuid # Bibliothèque native pour générer desd chaînes de caractères uniques(universally unique identifier)
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from auth import fabriquer_badge, verifier_badge_medecin, verifier_badge_patient, verifier_badge_admin,verifier_badge_agent, hash_password, verifier_mot_de_passe
#from fastapi.security import OAuth2PasswordRequestForm
from datetime import datetime
from schemas_pydantic import *
import os
import json
import joblib
import shap
import pandas as pd

from fastapi.middleware.cors import CORSMiddleware




app = FastAPI(
    title="Plateforme Prédictive CardioDiab - API",
    description="Backend de gestion des diagnostics cliniques et d'intégration de l'IA",
    version="1.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://cardiodiab-frontend.vercel.app",  # votre vraie URL Vercel
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


#pour la gestion globale des exceptions qui n'ont pas été gérées
#gestionnaire global d'exceptions dans FastAPI
@app.exception_handler(Exception)
async def gestion_erreurs_globales(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": f"Erreur interne : {str(exc)}"},
    )

def enregistrer_audit(cursor, id_user, action, details, ip_adress):
    """Insère une trace d'audit dans la même transaction que l'action effectuée."""
    cursor.execute(
        "INSERT INTO audit_log (id_user, action, details, ip_adress) VALUES (%s, %s, %s, %s)",
        (id_user, action, details, ip_adress),
    )


# --- CHARGEMENT DES PIPELINES ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CARDIO_MODEL_PATH = os.path.join(BASE_DIR, "models", "pipeline_cardio_reglog.pkl")
DIABETE_MODEL_PATH = os.path.join(BASE_DIR, "models", "pipeline_diabete_model.pkl")

pipeline_cardio = joblib.load(CARDIO_MODEL_PATH) if os.path.exists(CARDIO_MODEL_PATH) else None
pipeline_diabete = joblib.load(DIABETE_MODEL_PATH) if os.path.exists(DIABETE_MODEL_PATH) else None

# --- INITIALISATION DES EXPLAINERS SHAP ---
explainer_cardio = None
explainer_diabete = None

if pipeline_cardio:
    # Pour la Régression Logistique (Modèle linéaire)
    # 1. On extrait le préprocesseur et le modèle linéaire du pipeline
    preprocessor_c = pipeline_cardio.named_steps['columntransformer']
    model_lr = pipeline_cardio.named_steps['logisticregression']
    
    # 2. On crée un faux background dataset (masker) de la bonne taille.
    # Ta régression logistique attend le nombre de caractéristiques APRÈS transformation.
    # On crée une ligne de zéros pour initialiser le mécanisme de SHAP.
    import numpy as np
    num_features_transformed = model_lr.n_features_in_
    dummy_background = np.zeros((1, num_features_transformed))
    
    # 3. On initialise l'explainer avec la fonction de décision et le masker requis
    explainer_cardio = shap.Explainer( lambda X: model_lr.predict_proba(X)[:, 1], masker=shap.maskers.Independent(dummy_background))

if pipeline_diabete:
    # Pour LightGBM (Modèle d'arbres), on utilise l'ultra-rapide TreeExplainer
    model_lgb = pipeline_diabete.named_steps['lgbmclassifier']
    explainer_diabete = shap.TreeExplainer(model_lgb)

REPARTITION_SMOKER = {
    "never": "never_smoked", "current": "current_smoker", "former": "former_smoker",
    "ever": "former_smoker", "not current": "former_smoker", "No Info": "unknown"
}

# --- FONCTION DE REGROUPEMENT MAGIQUE POUR LE ONE-HOT ENCODING ---
def regrouper_valeurs_shap(feature_names, shap_values, variables_origines):
    """
    Regroupe et additionne les contributions SHAP des variables encodées (One-Hot)
    pour revenir aux variables cliniques d'origine compréhensibles par le médecin.
    """
    dictionnaire_regroupe = {var: 0.0 for var in variables_origines}
    
    for name, val in zip(feature_names, shap_values):
        trouve = False
        # On cherche si le nom de la colonne transformée contient l'un de nos noms d'origine
        for var in variables_origines:
            if var in name:
                dictionnaire_regroupe[var] += float(val)
                trouve = True
                break
        # Si la variable n'est pas dans notre liste, on la garde brute au cas où
        if not trouve:
            dictionnaire_regroupe[name] = float(val)
            
    return dictionnaire_regroupe


#---------------------------------------------------------------
#Module Authentification et onboarding(integration(sign on))(Accès public)
#------------------------------------------------------------------

# --- 1. ROUTE D'AUTHENTIFICATION COMPATIBLE HERITAGE SQL ---
@app.post("/auth/login", summary="S'authentifier pour obtenir un badge")
def login(form_data: LoginRequest,request: Request):
    connection = get_db_connection()
    if not connection:
        raise HTTPException(status_code=500, detail="Base de données inaccessible.")
    cursor = connection.cursor(dictionary=True)
    
    try:
        # Étape A : On cherche dans la table parente 'utilisateur'
        query_user = "SELECT id_user, mot_de_passe, role FROM utilisateur WHERE email = %s"
        cursor.execute(query_user, (form_data.email,))
        user = cursor.fetchone()
        
        if not user or not verifier_mot_de_passe(form_data.mot_de_passe, user["mot_de_passe"]):
            raise HTTPException(status_code=400, detail="Email ou mot de passe incorrect.")
        
        id_utilisateur = user["id_user"]
        role_global = user["role"].upper()
        
        role_final = role_global
        id_final = id_utilisateur

        # Étape B : Si c'est un Médecin, on va chercher son profil dans la table enfant 'medecin'
        if role_global == "MEDECIN":
            query_medecin = "SELECT id_medecin, specialite, statut FROM medecin WHERE id_user = %s"
            cursor.execute(query_medecin, (id_utilisateur,))
            medecin_info = cursor.fetchone()
            
            if not medecin_info:
                raise HTTPException(status_code=404, detail="Profil médecin introuvable.")
            
            # Gestion du statut selon le ENUM SQL ('EN_ATTENTE', 'APPROUVE', 'REJETE')
            if medecin_info["statut"] == "EN_ATTENTE":
                raise HTTPException(status_code=403, detail="Compte en attente d'approbation par l'administrateur.")
            elif medecin_info["statut"] == "REJETE":
                raise HTTPException(status_code=403, detail="Compte rejeté par l'administrateur.")
            
            role_final = medecin_info["specialite"].upper() # Ex: 'CARDIOLOGUE'
            id_final = medecin_info["id_medecin"] # On prend le vrai id_medecin pour les consultations

        # Étape C : Si c'est un Patient, on prend son id_patient
        elif role_global == "PATIENT":
            query_patient = "SELECT id_patient FROM patient WHERE id_user = %s"
            cursor.execute(query_patient, (id_utilisateur,))
            patient_info = cursor.fetchone()
            if patient_info:
                id_final = patient_info["id_patient"]
            
        # Étape C : Si c'est un Agent d'admission, on prend son id_agent
        elif role_global == "AGENT":
            query_agent = "SELECT id_agent,statut FROM agent_admission WHERE id_user = %s"
            cursor.execute(query_agent, (id_utilisateur,))
            agent_info = cursor.fetchone()

            if not agent_info:
                raise HTTPException(status_code=404, detail="Profil Agent d'admission introuvable.")
            
            # Gestion du statut selon le ENUM SQL ('EN_ATTENTE', 'APPROUVE', 'REJETE')
            if agent_info["statut"] == "EN_ATTENTE":
                raise HTTPException(status_code=403, detail="Compte en attente d'approbation par l'administrateur.")
            elif agent_info["statut"] == "REJETE":
                raise HTTPException(status_code=403, detail="Compte rejeté par l'administrateur.")
            
            id_final = agent_info["id_agent"]

        # Fabrication du badge sécurisé
        badge = fabriquer_badge(user_id=id_final, role=role_final)
        
        enregistrer_audit(cursor, id_final, "CONNEXION", f"Connexion réussie ({role_final})", request.client.host)
        connection.commit()
        return {
            "access_token": badge,
            "token_type": "bearer",
            "role": role_final,
            "user_id": id_final
        }
    finally:
        cursor.close()
        connection.close()


#-----2 
@app.post("/auth/register-agent",summary="Inscription publique d'un nouveau agent d'admission")
def register_agent(data:AgentRegisterRequest,request:Request):
    connection=get_db_connection()
    if not connection:
        raise HTTPException(status_code=500,detail="Base de données inaccessible.")
    cursor=connection.cursor()

    try:
        query_user="INSERT INTO utilisateur(email, mot_de_passe, telephone, role) VALUES(%s,%s,%s,'AGENT')"
        cursor.execute(query_user,(data.email,hash_password(data.mot_de_passe),data.telephone))
        user_id=cursor.lastrowid

        query_agent="INSERT INTO agent_admission(id_user,matricule,nom_agent,prenom_agent,service_affectation,statut) VALUES(%s,%s,%s,%s,%s,'EN_ATTENTE')"
        cursor.execute(query_agent,(user_id,data.matricule,data.nom_agent,data.prenom_agent,data.service_affectation))

        enregistrer_audit(cursor,user_id,"INSCRIPTION AGENT",f"Demande d'inscription ({data.service_affectation})",request.client.host)
        connection.commit()
        return {"statut": "Succès", "message": f"Le compte du Agent {data.nom_agent} a été créé. En attente de validation."}

    except Exception as e:
        connection.rollback()
        if "Duplicate entry" in str(e):
            raise HTTPException(status_code=400, detail="Cette adresse email est déjà utilisée.")
        raise HTTPException(status_code=500, detail=f"Erreur : {str(e)}")
    finally:
        cursor.close()
        connection.close()


#-------3
@app.post("/auth/register-medecin", summary="Inscription publique d'un nouveau médecin")
def register_medecin(data: MedecinRegisterRequest,request: Request):
    if data.specialite not in [ 'CARDIOLOGUE', 'DIABETOLOGUE']:
        raise HTTPException(status_code=400, detail="Spécialité médicale invalide.")
        
    connection = get_db_connection()
    if not connection:
        raise HTTPException(status_code=500, detail="Base de données inaccessible.")
    cursor = connection.cursor()
    
    try:
        query_user = """
            INSERT INTO utilisateur (email, mot_de_passe, telephone, role) 
            VALUES (%s, %s, %s, 'MEDECIN')
        """
        cursor.execute(query_user, (data.email, hash_password(data.mot_de_passe), data.telephone))
        user_id = cursor.lastrowid
        
        # Correction ici : utilisation de 'statut' au lieu de 'statut_approbation'
        query_medecin = """
            INSERT INTO medecin (id_user, nom, prenom,num_ordre_cnom,code_inpe, specialite, statut) 
            VALUES (%s, %s, %s, %s,%s,%s, 'EN_ATTENTE')
        """
        cursor.execute(query_medecin, (user_id, data.nom, data.prenom,data.num_ordre_cnom,data.code_inpe, data.specialite))
        
        enregistrer_audit(cursor, user_id, "INSCRIPTION_MEDECIN", f"Demande d'inscription ({data.specialite})", request.client.host)
        connection.commit()
        return {"statut": "Succès", "message": f"Le compte du Dr {data.nom} a été créé. En attente de validation."}
    except Exception as e:
        connection.rollback()
        if "Duplicate entry" in str(e):
            raise HTTPException(status_code=400, detail="Cette adresse email est déjà utilisée.")
        raise HTTPException(status_code=500, detail=f"Erreur : {str(e)}")
    finally:
        cursor.close()
        connection.close()


#---------------------------------------------------------------------
#Espace Agent d'admission(Sécurisé par verifier_badge_agent)
#-------------------------------------------------------------------------

#1
@app.post("/agent/creer-patient", summary="Créer un dossier patient et générer son ipp(identifiant permanent du patient)")
def creer_dossier_patient(data: PatientCreateRequest, request: Request,agent_connecte: dict = Depends(verifier_badge_agent)):
    connection = get_db_connection()
    if not connection:
        raise HTTPException(status_code=500, detail="Base de données inaccessible.")
    cursor = connection.cursor()
    
    try:
        id_agent_authentifie = agent_connecte["user_id"]
        
        # 1. Génération des accès temporaires
        ipp = f"IPP-{str(uuid.uuid4())[:8].upper()}"
        
        # 2. Insertion dans la table mère utilisateur (id_user est AUTO_INCREMENT)
        query_user = """
            INSERT INTO utilisateur (email, mot_de_passe, telephone, role) 
            VALUES (%s, %s, %s, 'PATIENT')
        """
        cursor.execute(query_user, (data.email, hash_password(ipp), data.telephone))
        id_user_genere = cursor.lastrowid # Récupère l'id_user créé
        
        # 3. Insertion dans la table enfant patient (Conforme à ton SQL !)
        query_patient = """
            INSERT INTO patient (id_user, ipp, nom, prenom, cin, couverture_medicale, date_naissance, gender) 
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        cursor.execute(query_patient, (
            id_user_genere, ipp, data.nom, data.prenom, data.cin, data.couverture_medicale, data.date_naissance, data.gender))
        id_patient_genere = cursor.lastrowid # Récupère l'id_patient pour la table de liaison
        

        enregistrer_audit(cursor, id_agent_authentifie, "CREATION_PATIENT", f"Création du dossier patient #{id_patient_genere}", request.client.host)
        connection.commit()
        return {
            "statut": "Succès",
            "message": "Dossier patient créer avec succès.",
            "details": {
                "id_patient": id_patient_genere,
                "id_user": id_user_genere,
                "mot_de_passe": ipp,
                "ipp_genere": ipp
            }
        }
    except Exception as e:
        connection.rollback()
        if "Duplicate entry" in str(e):
            raise HTTPException(status_code=400, detail="Cet email ou ce code d'activation existe déjà.")
        raise HTTPException(status_code=500, detail=f"Erreur lors de la création du dossier : {str(e)}")
    finally:
        cursor.close()
        connection.close()

#---------------------------------------------------------------------
#Espace Médecin(Sécurisé par verifier_badge_medecin)
#-------------------------------------------------------------------------

#1
# ---  CRÉER UN LIEN AVEC UN PATIENT (Via son ipp) ---
@app.post("/medecin/lier-patient", summary="Associer un patient au médecin via son ipp(identifiant permanent du patient)")
def lier_patient_code(data: LiaisonRequest,request: Request, medecin_connecte: dict = Depends(verifier_badge_medecin)):
    connection = get_db_connection()
    if not connection:
        raise HTTPException(status_code=500, detail="Base de données inaccessible.")
    cursor = connection.cursor(dictionary=True)
    
    try:
        id_medecin = medecin_connecte["user_id"]
        
        # 1. Vérifier si l'ipp correspond à un patient valide
        cursor.execute("SELECT id_patient FROM patient WHERE ipp = %s", (data.ipp,))
        patient = cursor.fetchone()
        
        if not patient:
            raise HTTPException(status_code=404, detail="ipp(identifiant permanent du patient) invalide. Patient introuvable.")
        
        id_patient = patient["id_patient"]
        
        # 2. Vérifier si un lien existe déjà entre ce médecin et ce patient
        cursor.execute("SELECT id_lien, statut FROM lien_medecin_patient WHERE id_medecin = %s AND id_patient = %s", (id_medecin, id_patient))
        lien_existant = cursor.fetchone()
        
        if lien_existant:
            if lien_existant["statut"] == "ACTIF":
                return {"statut": "Info", "message": "Ce patient fait déjà partie de votre liste active."}
            else:
                # Réactiver le lien s'il avait été révoqué
                cursor.execute("UPDATE lien_medecin_patient SET statut = 'ACTIF', date_liaison = CURRENT_TIMESTAMP WHERE id_lien = %s", (lien_existant["id_lien"],))
        else:
            # Créer une nouvelle liaison active
            query_insert = "INSERT INTO lien_medecin_patient (id_medecin, id_patient, statut) VALUES (%s, %s, 'ACTIF')"
            cursor.execute(query_insert, (id_medecin, id_patient))

        enregistrer_audit(cursor, id_medecin, "LIAISON_PATIENT", f"Liaison avec le patient #{id_patient}", request.client.host)    
        connection.commit()
        return {"statut": "Succès", "message": "Patient associé avec succès."}
        
    finally:
        cursor.close()
        connection.close()

#3.
# ---  OBTENIR LA LISTE GÉNÉRALE DES PATIENTS DU MÉDECIN ---
@app.get("/medecin/patients", summary="Obtenir tous les patients liés au médecin connecté", response_model=list[PatientListItem])
def obtenir_patients_medecin(medecin_connecte: dict = Depends(verifier_badge_medecin)):
    connection = get_db_connection()
    if not connection:
        raise HTTPException(status_code=500, detail="Base de données inaccessible.")
    cursor = connection.cursor(dictionary=True)
    
    try:
        id_medecin = medecin_connecte["user_id"]
        
        # Requête pour lister les patients actifs liés au médecin via la table de liaison
        query = """
            SELECT p.id_patient,p.ipp, p.nom, p.prenom, DATE_FORMAT(p.date_naissance, '%Y-%m-%d') as date_naissance, 
            p.gender, p.cin, p.couverture_medicale, l.statut as statut_lien FROM patient p
            JOIN lien_medecin_patient l ON p.id_patient = l.id_patient
            WHERE l.id_medecin = %s AND l.statut = 'ACTIF'
        """
        cursor.execute(query, (id_medecin,))
        patients = cursor.fetchall()
        return patients
    finally:
        cursor.close()
        connection.close()


#4.
# --- . OBTENIR L'HISTORIQUE DES CONSULTATIONS D'UN PATIENT ---
@app.get("/medecin/patient/{id_patient}/consultations", summary="Historique des consultations d'un patient spécifique", response_model=list[ConsultationHistoryItem])
def historique_consultations_patient(id_patient: int, medecin_connecte: dict = Depends(verifier_badge_medecin)):
    connection = get_db_connection()
    if not connection:
        raise HTTPException(status_code=500, detail="Base de données inaccessible.")
    cursor = connection.cursor(dictionary=True)
    
    try:
        id_medecin = medecin_connecte["user_id"]
        
        # Sécurité : On vérifie d'abord si ce médecin a bien le droit de voir ce patient (lien ACTIF)
        cursor.execute("SELECT id_lien FROM lien_medecin_patient WHERE id_medecin = %s AND id_patient = %s AND statut = 'ACTIF'", (id_medecin, id_patient))
        if not cursor.fetchone():
            raise HTTPException(status_code=403, detail="Accès refusé : Vous n'avez pas d'autorisation active pour ce patient.")
        
        # Récupération de l'historique des examens du plus récent au plus ancien
        query = """
            SELECT id_consultation, DATE_FORMAT(date_consultation, '%Y-%m-%d %H:%M') as date_consultation, 
                   type_consultation, score_cardio, score_diabete
            FROM consultation
            WHERE id_patient = %s AND id_medecin = %s
            ORDER BY date_consultation DESC
        """
        cursor.execute(query, (id_patient, id_medecin))
        historique = cursor.fetchall()
        return historique
    finally:
        cursor.close()
        connection.close()


#5.
@app.post("/medecin/consultation",summary="Enregistrer une consultation (Sécurisé)")
def enregistrer_consultation(data: ConsultationRequest,request: Request, medecin_connecte: dict = Depends(verifier_badge_medecin) ):
    if data.height <= 0:
        raise HTTPException(status_code=400, detail="La taille doit être supérieure à 0.")
    
    bmi_calcule = round(data.weight / (data.height ** 2), 2)
    diff_ps_pd_calcule = data.ap_hi - data.ap_lo
    
        # --- Dérivation du type de consultation à partir de la spécialité ---
    specialite = data.flux_prediction.upper()
    if specialite == "CARDIOLOGUE":
        type_consultation_calcule = "CARDIOLOGUE"
    elif specialite == "DIABETOLOGUE":
        type_consultation_calcule = "DIABETOLOGUE"
    else:
        raise HTTPException(status_code=400, detail="flux_prediction invalide : doit être CARDIOLOGUE ou DIABETOLOGUE.")
    

    connection = get_db_connection()
    if not connection:
        raise HTTPException(status_code=500, detail="Base de données inaccessible.")
    cursor = connection.cursor(dictionary=True)
    
    try:
        id_medecin_authentifie = medecin_connecte["user_id"]
        # Récupération des infos du patient
        cursor.execute("SELECT date_naissance, gender FROM patient WHERE id_patient = %s", (data.id_patient,))
        patient_info = cursor.fetchone()
        
        age_patient, gender_brut = 0, 1
        if patient_info:
            gender_brut = patient_info["gender"]
            if patient_info["date_naissance"]:
                age_patient = datetime.now().year - patient_info["date_naissance"].year
                
        # Préparation des DataFrames
        df_cardio = pd.DataFrame([{
            "age": age_patient, "diff_PS_PD": diff_ps_pd_calcule, "diastolic_pressure": data.ap_lo,
            "bmi": bmi_calcule, "cholesterol": data.cholesterol, "glucose": data.gluc,
            "gender": gender_brut, "smoking_status": data.smoke, "alcohol_status": data.alco, "physical_activity": data.active
        }])

        status_smoker_diabete = REPARTITION_SMOKER.get(data.smoking_history, "unknown")
        df_diabete = pd.DataFrame([{
            "age": float(age_patient), "bmi": bmi_calcule, "HbA1c_level": data.HbA1c_level,
            "blood_glucose_level": data.blood_glucose_level, "hypertension": data.antecedent_hypertension,
            "heart_disease": data.antecedent_heart_disease, "gender": "Male" if gender_brut == 2 else "Female",
            "smoking_status": status_smoker_diabete
        }])

        # Inférence
        score_cardio, score_diabete = 0.00, 0.00
        specialite = data.flux_prediction.upper()

        if specialite == "CARDIOLOGUE" and pipeline_cardio:
            score_cardio = round(pipeline_cardio.predict_proba(df_cardio)[0][1] * 100, 2)
        if specialite == "DIABETOLOGUE" and pipeline_diabete:
            score_diabete = round(pipeline_diabete.predict_proba(df_diabete)[0][1] * 100, 2)

        # Enregistrement clinique
        query_consult = """
            INSERT INTO consultation (
                id_patient, id_medecin, type_consultation, weight, height,antecedent_hypertension,antecedent_heart_disease, ap_hi, ap_lo, cholesterol, gluc, 
                smoke, alco, active, smoking_history, HbA1c_level, blood_glucose_level,
                score_cardio, score_diabete, commentaire_medecin
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,%s,%s,%s, %s, %s, %s, %s, %s, %s)
        """
        values_consult = (
            data.id_patient, id_medecin_authentifie,type_consultation_calcule, data.weight, data.height,data.antecedent_hypertension,data.antecedent_heart_disease, 
            data.ap_hi, data.ap_lo,data.cholesterol, data.gluc, data.smoke, data.alco, data.active, REPARTITION_SMOKER.get(data.smoking_history, "unknown"),
            data.HbA1c_level, data.blood_glucose_level, score_cardio, score_diabete, data.commentaire_medecin
        )
        cursor.execute(query_consult, values_consult)
        consultation_id = cursor.lastrowid

        # --- CALCUL SHAP AVANCÉ (AVEC REGROUPEMENT) ---
        explicabilite_finale = {}

        # 1. Traitement SHAP pour le modèle Cardio
        if specialite =="CARDIOLOGUE" and explainer_cardio:
            preprocessor_c = pipeline_cardio.named_steps['columntransformer']
            X_trans_c = preprocessor_c.transform(df_cardio)
            feature_names_c = preprocessor_c.get_feature_names_out()
            shap_vals_c = explainer_cardio.shap_values(X_trans_c)
            
            # Liste de tes variables d'origine pour le tri
            vars_origines_c = ['age', 'diff_PS_PD', 'diastolic_pressure', 'bmi', 'cholesterol', 'glucose', 'gender', 'smoking_status', 'alcohol_status', 'physical_activity']
            explicabilite_finale["cardio"] = regrouper_valeurs_shap(feature_names_c, shap_vals_c[0], vars_origines_c)

        # 2. Traitement SHAP pour le modèle Diabète (LightGBM)
        if specialite =="DIABETOLOGUE" and explainer_diabete:
            preprocessor_d = pipeline_diabete.named_steps['columntransformer']
            X_trans_d = preprocessor_d.transform(df_diabete)
            feature_names_d = preprocessor_d.get_feature_names_out()
            
            # Pour LightGBM (classification binaire), TreeExplainer renvoie une liste de deux matrices [classe_0, classe_1]
            # On prend l'indice [1] pour expliquer le risque positif (Présence de diabète)
            shap_vals_d = explainer_diabete.shap_values(X_trans_d)
            shap_vals_d_classe1 = shap_vals_d[1][0] if isinstance(shap_vals_d, list) else shap_vals_d[0]
            
            vars_origines_d = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'hypertension', 'heart_disease', 'gender', 'smoking_status']
            explicabilite_finale["diabete"] = regrouper_valeurs_shap(feature_names_d, shap_vals_d_classe1, vars_origines_d)

        # Sauvegarde de la structure XAI propre au format JSON
        if explicabilite_finale:
            query_shap = "INSERT INTO explicabilite (id_consultation, valeurs_shap_json) VALUES (%s, %s)"
            cursor.execute(query_shap, (consultation_id, json.dumps(explicabilite_finale)))

        enregistrer_audit(
        cursor, id_medecin_authentifie, "CONSULTATION",
        f"Consultation #{consultation_id} pour patient #{data.id_patient} (cardio={score_cardio}%, diabète={score_diabete}%)",
        request.client.host,
        )
        connection.commit()
        return {
            "statut": "Succès",
            "id_consultation": consultation_id,
            "ia_predictions": {"score_cardio_pourcent": score_cardio, "score_diabete_pourcent": score_diabete},
            "xai_status": "SHAP calculé et regroupé avec succès pour les variables One-Hot."
        }

    except Exception as e:
        connection.rollback()
        raise HTTPException(status_code=500, detail=f"Erreur XAI : {str(e)}")
    finally:
        cursor.close()
        connection.close()



#6.
# --- . OBTENIR LE DÉTAIL COMPLET ET SHAP D'UNE ANCIENNE CONSULTATION ---
@app.get("/medecin/consultation/{id_consultation}", summary="Détails cliniques et matrices SHAP d'un examen passé")
def details_consultation_passee(id_consultation: int, medecin_connecte: dict = Depends(verifier_badge_medecin)):
    connection = get_db_connection()
    if not connection:
        raise HTTPException(status_code=500, detail="Base de données inaccessible.")
    cursor = connection.cursor(dictionary=True)
    
    try:
        id_medecin = medecin_connecte["user_id"]
        role_medecin = medecin_connecte["role"] # CARDIOLOGUE ou DIABETOLOGUE
        
        # 1. Récupération de la consultation
        query_consult = "SELECT * FROM consultation WHERE id_consultation = %s AND id_medecin = %s"
        cursor.execute(query_consult, (id_consultation, id_medecin))
        consultation = cursor.fetchone()
        
        if not consultation:
            raise HTTPException(status_code=404, detail="Consultation introuvable ou non associée à votre compte.")
        
        # Formatage de la date en chaîne de caractères
        if consultation["date_consultation"]:
            consultation["date_consultation"] = consultation["date_consultation"].strftime('%Y-%m-%d %H:%M')

        # 2. Récupération de la matrice d'explicabilité SHAP associée
        query_shap = "SELECT valeurs_shap_json FROM explicabilite WHERE id_consultation = %s"
        cursor.execute(query_shap, (id_consultation,))
        shap_record = cursor.fetchone()
        
        valeurs_shap = json.loads(shap_record["valeurs_shap_json"]) if shap_record else {}

        # 3. Logique de filtrage d'affichage selon les profils (A et B) de tes spécifications :
        if role_medecin == "CARDIOLOGUE":
            # Filtrage expert : On masque le bloc diabète
            consultation["score_diabete"] = None
            if "diabete" in valeurs_shap: del valeurs_shap["diabete"]
            
        else: 
            # Filtrage expert : On masque le bloc cardio
            consultation["score_cardio"] = None
            if "cardio" in valeurs_shap: del valeurs_shap["cardio"]
            

        return {
            "donnees_cliniques": consultation,
            "explicabilite_shap": valeurs_shap
        }
    finally:
        cursor.close()
        connection.close()



#7.
# ---  RÉDIGER / MODIFIER LES CONSEILS ET RECOMMANDATIONS ---
@app.put("/medecin/consultation/{id_consultation}/commentaire", summary="Rédiger ou modifier le commentaire clinique d'une consultation")
def mettre_a_jour_commentaire(id_consultation: int, data: CommentaireUpdateRequest,request: Request, medecin_connecte: dict = Depends(verifier_badge_medecin)):
    connection = get_db_connection()
    if not connection:
        raise HTTPException(status_code=500, detail="Base de données inaccessible.")
    cursor = connection.cursor(dictionary=True)
    
    try:
        id_medecin = medecin_connecte["user_id"]
        
        # Vérifier que la consultation existe et appartient bien à ce médecin
        cursor.execute("SELECT id_consultation FROM consultation WHERE id_consultation = %s AND id_medecin = %s", (id_consultation, id_medecin))
        if not cursor.fetchone():
            raise HTTPException(status_code=403, detail="Modification interdite : Cette consultation ne vous est pas associée.")
        
        # Mise à jour du commentaire rédigé par le médecin
        query_update = "UPDATE consultation SET commentaire_medecin = %s WHERE id_consultation = %s"
        cursor.execute(query_update, (data.commentaire_medecin, id_consultation))
        
        enregistrer_audit(cursor, id_medecin, "MAJ_RECOMMANDATIONS", f"Recommandations mises à jour pour la consultation #{id_consultation}", request.client.host)

        connection.commit()
        return {"statut": "Succès", "message": "Conseils et recommandations mis à jour avec succès."}
        
    finally:
        cursor.close()
        connection.close()


# --- 8. TÉLÉCHARGER LE RAPPORT CLINIQUE (Export Données) ---
@app.get("/medecin/consultation/{id_consultation}/rapport", summary="Générer un rapport structuré pour impression")
def generer_donnees_rapport(id_consultation: int, medecin_connecte: dict = Depends(verifier_badge_medecin)):
    connection = get_db_connection()
    if not connection:
        raise HTTPException(status_code=500, detail="Base de données inaccessible.")
    cursor = connection.cursor(dictionary=True)
    
    try:
        id_medecin = medecin_connecte["user_id"]
        role_medecin = medecin_connecte["role"]

        query = """
            SELECT c.*, p.nom as patient_nom, p.prenom as patient_prenom, 
            p.date_naissance as patient_naissance_brute, p.gender as patient_gender,
            p.cin, p.couverture_medicale FROM consultation c
            JOIN patient p ON c.id_patient = p.id_patient
            WHERE c.id_consultation = %s AND c.id_medecin = %s
        """
        cursor.execute(query, (id_consultation, id_medecin))
        rapport_data = cursor.fetchone()
        
        if not rapport_data:
            raise HTTPException(status_code=404, detail="Rapport introuvable ou accès non autorisé.")
            
        if rapport_data["date_consultation"]:
            rapport_data["date_consultation"] = rapport_data["date_consultation"].strftime('%Y-%m-%d %H:%M')

        # Calcul de l'âge à la date de la consultation, et formatage de la naissance
        date_naissance_brute = rapport_data["patient_naissance_brute"]
        age_patient = (datetime.now().year - date_naissance_brute.year) if date_naissance_brute else None
        patient_naissance_str = date_naissance_brute.strftime('%Y-%m-%d') if date_naissance_brute else None

        bmi = round(rapport_data["weight"] / (rapport_data["height"] ** 2), 2) if rapport_data["height"] > 0 else 0

        # --- Bloc commun (utilisé par les 2 modèles) ---
        constantes = {
            "age": age_patient,
            "gender": rapport_data["patient_gender"],
            "poids_kg": float(rapport_data["weight"]),
            "taille_m": float(rapport_data["height"]),
            "imc_calcule": bmi,
        }
        conclusions = {}

        if role_medecin == "CARDIOLOGUE":
            constantes.update({
                "pression_systolique": rapport_data["ap_hi"],
                "pression_diastolique": rapport_data["ap_lo"],
                "cholesterol_classe": rapport_data["cholesterol"],
                "glucose_classe": rapport_data["gluc"],
                "smoke": rapport_data["smoke"],
                "alcool": rapport_data["alco"],
                "activite_physique": rapport_data["active"],
            })
            conclusions["risque_cardiovasculaire"] = f"{rapport_data['score_cardio']}%"

        elif role_medecin == "DIABETOLOGUE":
            constantes.update({
                "antecedent_hypertension": rapport_data["antecedent_hypertension"],
                "antecedent_heart_disease": rapport_data["antecedent_heart_disease"],
                "smoking_history": rapport_data["smoking_history"],
                "taux_hemoglobine_glyquee": float(rapport_data["HbA1c_level"]) if rapport_data["HbA1c_level"] else None,
                "glycemie_a_jeun": rapport_data["blood_glucose_level"],
            })
            conclusions["risque_diabete"] = f"{rapport_data['score_diabete']}%"

        return {
            "entete_etablissement": "Plateforme d'Aide au Diagnostic CardioDiab",
            "specialite": role_medecin,
            "date_edition": datetime.now().strftime("%Y-%m-%d"),
            "date_examen": rapport_data["date_consultation"],
            "dossier_patient": {
                "nom": rapport_data["patient_nom"],
                "prenom": rapport_data["patient_prenom"],
                "date_naissance": patient_naissance_str,
                "cin": rapport_data["cin"],
                "couverture_medicale": rapport_data["couverture_medicale"]
            },
            "constantes_biologiques": constantes,
            "conclusions_ia": conclusions,
            "recommandations_therapeutiques": rapport_data["commentaire_medecin"] if rapport_data["commentaire_medecin"] else "Aucun commentaire écrit pour le moment."
        }
    finally:
        cursor.close()
        connection.close()

#--------------------------------------------------------------------------
#Espace Patient (Sécurisé par verifier_badge_patient)
#-------------------------------------------------------------------------
#1.
# --- . TIMELINE : HISTORIQUE ÉVOLUTIF DES SCORES ---
@app.get("/patient/dashboard/timeline", summary="Récupérer l'historique chronologique des scores du patient")
def obtenir_timeline_patient(patient_connecte: dict = Depends(verifier_badge_patient)):
    connection = get_db_connection()
    if not connection:
        raise HTTPException(status_code=500, detail="Base de données inaccessible.")
    cursor = connection.cursor(dictionary=True)
    
    try:
        id_patient = patient_connecte["user_id"]
        
        # Sélection chronologique (de la plus ancienne à la plus récente pour le graphique linéaire)
        query = """
            SELECT id_consultation,
                   DATE_FORMAT(date_consultation, '%Y-%m-%d') as date_examen,
                   type_consultation, score_cardio, score_diabete
            FROM consultation
            WHERE id_patient = %s
            ORDER BY date_consultation ASC
        """
        cursor.execute(query, (id_patient,))
        historique = cursor.fetchall()
        return historique
    finally:
        cursor.close()
        connection.close()


@app.get("/patient/dashboard/Indicateurs", summary="Indicateurs de corpulence (IMC), habitudes de vie, et évolution des scores cardio/diabète séparément")
def obtenir_indicateurs_sante(patient_connecte: dict = Depends(verifier_badge_patient)):
    connection = get_db_connection()
    if not connection:
        raise HTTPException(status_code=500, detail="Base de données inaccessible.")
    cursor = connection.cursor(dictionary=True)
    
    try:
        id_patient = patient_connecte["user_id"]

        # --- 1. PARTIE GÉNÉRALE : corpulence + habitudes, sur les 2 dernières consultations (tout type confondu) ---
        cursor.execute("""
            SELECT weight, height, smoke, alco, active
            FROM consultation 
            WHERE id_patient = %s 
            ORDER BY date_consultation DESC LIMIT 2
        """, (id_patient,))
        consultations_generales = cursor.fetchall()

        if not consultations_generales:
            raise HTTPException(status_code=404, detail="Aucun indicateur disponible. En attente d'une première consultation.")

        c_actuelle = consultations_generales[0]
        imc_actuel = round(c_actuelle["weight"] / (c_actuelle["height"] ** 2), 2) if c_actuelle["height"] > 0 else 0

        imc_precedent = None
        evolution_imc = "STABLE"
        if len(consultations_generales) > 1:
            c_precedente = consultations_generales[1]
            imc_precedent = round(c_precedente["weight"] / (c_precedente["height"] ** 2), 2) if c_precedente["height"] > 0 else 0
            if imc_actuel > imc_precedent:
                evolution_imc = "HAUSSE"
            elif imc_actuel < imc_precedent:
                evolution_imc = "BAISSE"

        # --- 2. FONCTION UTILITAIRE : calcule actuel/précédent/évolution pour UN type donné ---
        def calculer_evolution_score(type_cible: str, colonne_score: str):
            cursor.execute(f"""
                SELECT {colonne_score} as score
                FROM consultation
                WHERE id_patient = %s AND type_consultation = %s
                ORDER BY date_consultation DESC LIMIT 2
            """, (id_patient, type_cible))
            rows = cursor.fetchall()

            if not rows:
                return {"score_actuel": None, "score_prec": None, "evolution": "AUCUNE_DONNEE"}

            score_actuel = rows[0]["score"]
            score_prec = None
            evolution = "STABLE"

            if len(rows) > 1:
                score_prec = rows[1]["score"]
                if score_actuel > score_prec:
                    evolution = "HAUSSE"
                elif score_actuel < score_prec:
                    evolution = "BAISSE"

            return {"score_actuel": score_actuel, "score_prec": score_prec, "evolution": evolution}

        score_cardio_bloc = calculer_evolution_score("CARDIOLOGUE", "score_cardio")
        score_diabete_bloc = calculer_evolution_score("DIABETOLOGUE", "score_diabete")

        return {
            "corpulence": {
                "poids_actuel_kg": float(c_actuelle["weight"]),
                "taille_m": float(c_actuelle["height"]),
                "imc_actuel": imc_actuel,
                "imc_precedent": imc_precedent,
                "evolution": evolution_imc
            },
            "score_cardio": score_cardio_bloc,
            "score_diabete": score_diabete_bloc,
            "badges_hygiene_vie": {
                "activite_physique": "ACTIVE" if c_actuelle["active"] == 1 else "SEDENTAIRE",
                "tabagisme": "NON_FUMEUR" if c_actuelle["smoke"] == 0 else "FUMEUR",
                "alcool": "NON_CONSOMMATEUR" if c_actuelle["alco"] == 0 else "CONSOMMATEUR"
            }
        }
    finally:
        cursor.close()
        connection.close()


# --- 3. OBTENIR CONSEILS ET RECOMMANDATIONS MÉDICALES DIRECTES ---
@app.get("/patient/dashboard/recommandations", summary="Dernières recommandations textuelles du médecin")
def obtenir_recommandations_medecin(patient_connecte: dict = Depends(verifier_badge_patient)):
    connection = get_db_connection()
    if not connection:
        raise HTTPException(status_code=500, detail="Base de données inaccessible.")
    cursor = connection.cursor(dictionary=True)
    
    try:
        id_patient = patient_connecte["user_id"]
        
        # On va chercher le commentaire de la toute dernière consultation, ainsi que le nom du médecin qui l'a rédigé
        query = """
            SELECT c.commentaire_medecin, DATE_FORMAT(c.date_consultation, '%Y-%m-%d') as date_conseil,
                   m.nom as medecin_nom, m.prenom as medecin_prenom, m.specialite
            FROM consultation c
            JOIN medecin m ON c.id_medecin = m.id_medecin
            WHERE c.id_patient = %s 
              AND c.commentaire_medecin IS NOT NULL 
              AND c.commentaire_medecin != ''
              AND c.date_consultation = (
                  SELECT MAX(c2.date_consultation)
                  FROM consultation c2
                  JOIN medecin m2 ON c2.id_medecin = m2.id_medecin
                  WHERE c2.id_patient = %s AND m2.specialite = m.specialite
              )
        """
        cursor.execute(query, (id_patient, id_patient))
        recommandation = cursor.fetchone()
        
        if not recommandation or not recommandation["commentaire_medecin"]:
            return {"message": "Aucun conseil personnalisé n'a encore été rédigé pour le moment."}
            
        return recommandation
    finally:
        cursor.close()
        connection.close()


# --- 4. TÉLÉCHARGER UN RAPPORT CLINIQUE (Côté Patient) ---
@app.get("/patient/consultation/{id_consultation}/rapport", summary="Permettre au patient de télécharger son propre rapport clinique")
def patient_telecharger_rapport(id_consultation: int, patient_connecte: dict = Depends(verifier_badge_patient)):
    connection = get_db_connection()
    if not connection:
        raise HTTPException(status_code=500, detail="Base de données inaccessible.")
    cursor = connection.cursor(dictionary=True)
    
    try:
        id_patient_token = patient_connecte["user_id"]
        
        query = """
            SELECT c.*, p.nom as patient_nom, p.prenom as patient_prenom, 
            p.date_naissance as patient_naissance_brute, p.gender as patient_gender
            FROM consultation c
            JOIN patient p ON c.id_patient = p.id_patient
            WHERE c.id_consultation = %s AND c.id_patient = %s
        """
        cursor.execute(query, (id_consultation, id_patient_token))
        rapport_data = cursor.fetchone()
        
        if not rapport_data:
            raise HTTPException(
                status_code=404, 
                detail="Rapport introuvable ou vous n'avez pas l'autorisation d'accéder à ce document."
            )
            
        if rapport_data["date_consultation"]:
            rapport_data["date_consultation"] = rapport_data["date_consultation"].strftime('%Y-%m-%d %H:%M')

        date_naissance_brute = rapport_data["patient_naissance_brute"]
        age_patient = (datetime.now().year - date_naissance_brute.year) if date_naissance_brute else None
        patient_naissance_str = date_naissance_brute.strftime('%Y-%m-%d') if date_naissance_brute else None

        bmi = round(rapport_data["weight"] / (rapport_data["height"] ** 2), 2) if rapport_data["height"] > 0 else 0

        constantes = {
            "age": age_patient,
            "gender": rapport_data["patient_gender"],
            "poids_kg": float(rapport_data["weight"]),
            "taille_m": float(rapport_data["height"]),
            "imc_calcule": bmi,
        }

        conclusions = {}
        type_c = rapport_data["type_consultation"]

        if type_c == "CARDIOLOGUE":
            constantes.update({
                "pression_systolique": rapport_data["ap_hi"],
                "pression_diastolique": rapport_data["ap_lo"],
                "cholesterol_classe": rapport_data["cholesterol"],
                "glucose_classe": rapport_data["gluc"],
            })
            conclusions["risque_cardiovasculaire"] = f"{rapport_data['score_cardio']}%"

        elif type_c == "DIABETOLOGUE":
            constantes.update({
                "antecedent_hypertension": rapport_data["antecedent_hypertension"],
                "antecedent_heart_disease": rapport_data["antecedent_heart_disease"],
                "taux_hemoglobine_glyquee": float(rapport_data["HbA1c_level"]) if rapport_data["HbA1c_level"] else None,
                "glycemie_a_jeun": rapport_data["blood_glucose_level"],
            })
            conclusions["risque_diabete"] = f"{rapport_data['score_diabete']}%"

        return {
            "entete_etablissement": "Plateforme d'Aide au Diagnostic CardioDiab",
            "type_consultation": type_c,
            "date_edition": datetime.now().strftime("%Y-%m-%d"),
            "date_examen": rapport_data["date_consultation"],
            "dossier_patient": {
                "nom": rapport_data["patient_nom"],
                "prenom": rapport_data["patient_prenom"],
                "date_naissance": patient_naissance_str
            },
            "constantes_biologiques": constantes,
            "conclusions_ia": conclusions,
            "recommandations_therapeutiques": rapport_data["commentaire_medecin"] if rapport_data["commentaire_medecin"] else "Aucun commentaire écrit pour le moment.",
        }
    finally:
        cursor.close()
        connection.close()

# --- 5. MODIFIER LE MOT DE PASSE (Côté Patient) ---
@app.post("/patient/modifier-mot-de-passe", summary="Permettre au patient connecté de modifier son mot de passe")
def modifier_mot_de_passe(
    donnees: ChangementMotDePasse, 
    patient_connecte: dict = Depends(verifier_badge_patient)
):
    connection = get_db_connection()
    if not connection:
        raise HTTPException(status_code=500, detail="Base de données inaccessible.")
    cursor = connection.cursor(dictionary=True)
    
    try:
        id_patient = patient_connecte["user_id"]
        
        # 1. Récupérer l'id_user et le mot de passe actuel depuis la table utilisateur via une jointure
        query_sel = """
            SELECT u.id_user, u.mot_de_passe 
            FROM utilisateur u
            INNER JOIN patient p ON u.id_user = p.id_user
            WHERE p.id_patient = %s
        """
        cursor.execute(query_sel, (id_patient,))
        compte = cursor.fetchone()
        
        if not compte:
            raise HTTPException(status_code=404, detail="Patient ou compte utilisateur introuvable.")
            
        id_user_reel = compte["id_user"]
        mot_de_passe_stocke = compte["mot_de_passe"]

        # 2. Vérifier si l'ancien mot de passe saisi correspond à celui de la BDD
        ancien_valide = verifier_mot_de_passe(
            donnees.ancien_mot_de_passe, 
            mot_de_passe_stocke
        )
        
        if not ancien_valide:
            raise HTTPException(
                status_code=401, 
                detail="Le mot de passe actuel est incorrect."
            )
            
        # 3. Hacher le nouveau mot de passe
        nouveau_hachage = hash_password(donnees.nouveau_mot_de_passe)
        
        # 4. Mettre à jour la table utilisateur (où est stocké le mot de passe)
        query_upd = "UPDATE utilisateur SET mot_de_passe = %s WHERE id_user = %s"
        cursor.execute(query_upd, (nouveau_hachage, id_user_reel))
        connection.commit()  # Très important pour valider la modification sur MySQL
        
        return {"message": "Votre mot de passe a été modifié avec succès !"}
        
    except HTTPException as he:
        # On laisse remonter les erreurs HTTP (401, 404) directement sans rollback si inutile,
        # ou avec rollback pour être totalement sécurisé
        connection.rollback()
        raise he
    except Exception as e:
        # En cas d'erreur SQL ou autre exception inattendue, on annule la transaction
        connection.rollback()
        raise HTTPException(status_code=500, detail=f"Erreur interne lors de la modification : {str(e)}")
        
    finally:
        cursor.close()
        connection.close()


#--------------------------------------------------------------------------
#Espace Admin(Sécurisé par verifier_badge_admin)
#-------------------------------------------------------------------------


# --- 1. LISTER LES MÉDECINS EN ATTENTE DE VALIDATION ---
@app.get("/admin/medecins/en-attente", summary="Lister les médecins inscrits dont le compte attend une approbation")
def lister_medecins_en_attente(admin_connecte: dict = Depends(verifier_badge_admin)):
    connection = get_db_connection()
    if not connection:
        raise HTTPException(status_code=500, detail="Base de données inaccessible.")
    cursor = connection.cursor(dictionary=True)
    
    try:
        # Récupère les profils des médecins au statut 'EN_ATTENTE'
        query = """
            SELECT m.id_medecin, u.id_user, m.nom, m.prenom,m.num_ordre_cnom,m.code_inpe, u.email, m.specialite, m.statut
            FROM medecin m
            JOIN utilisateur u ON m.id_user = u.id_user
            WHERE m.statut = 'EN_ATTENTE'
        """
        cursor.execute(query)
        return cursor.fetchall()
    finally:
        cursor.close()
        connection.close()


# --- 2. LISTER LES AGENTS EN ATTENTE DE VALIDATION ---
@app.get("/admin/agents/en-attente", summary="Lister les agents inscrits dont le compte attend une approbation")
def lister_agents_en_attente(admin_connecte: dict = Depends(verifier_badge_admin)):
    connection = get_db_connection()
    if not connection:
        raise HTTPException(status_code=500, detail="Base de données inaccessible.")
    cursor = connection.cursor(dictionary=True)
    
    try:
        # Récupère les profils des agents au statut 'EN_ATTENTE'
        query = """
            SELECT a.id_agent, u.id_user, a.matricule, a.nom_agent, a.prenom_agent,a.service_affectation, u.email, a.statut
            FROM agent_admission a
            JOIN utilisateur u ON a.id_user = u.id_user
            WHERE a.statut = 'EN_ATTENTE'
        """
        cursor.execute(query)
        return cursor.fetchall()
    finally:
        cursor.close()
        connection.close()



# --- 3. APPROUVER OU REJETER UN COMPTE MÉDECIN ---
@app.post("/admin/medecins/valider", summary="Approuver ou rejeter la demande d'inscription d'un professionnel")
def valider_compte_medecin(data: ApprobationMedecinRequest,request: Request, admin_connecte: dict = Depends(verifier_badge_admin)):
    if data.action not in ["APPROUVE", "REJETE"]:
        raise HTTPException(status_code=400, detail="Action invalide. Choisissez 'APPROUVE' ou 'REJETE'.")
        
    connection = get_db_connection()
    if not connection:
        raise HTTPException(status_code=500, detail="Base de données inaccessible.")
    cursor = connection.cursor(dictionary=True)
    
    try:
        # Vérifier si le médecin existe
        cursor.execute("SELECT id_medecin FROM medecin WHERE id_medecin = %s", (data.id_medecin,))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Médecin introuvable.")
            
        # Mise à jour du statut d'approbation
        query_update = "UPDATE medecin SET statut = %s WHERE id_medecin = %s"
        cursor.execute(query_update, (data.action, data.id_medecin))
        
        enregistrer_audit(cursor, admin_connecte["user_id"], "VALIDATION_MEDECIN", f"Médecin #{data.id_medecin} → {data.action}", request.client.host)

        connection.commit()
        return {"statut": "Succès", "message": f"Le compte du médecin a été mis à jour avec le statut : {data.action}."}
    finally:
        cursor.close()
        connection.close()


# --- 4. APPROUVER OU REJETER UN COMPTE AGENT ---
@app.post("/admin/agents/valider", summary="Approuver ou rejeter la demande d'inscription d'un agent d'admission")
def valider_compte_agent(data: ApprobationAgentRequest,request: Request, admin_connecte: dict = Depends(verifier_badge_admin)):
    if data.action not in ["APPROUVE", "REJETE"]:
        raise HTTPException(status_code=400, detail="Action invalide. Choisissez 'APPROUVE' ou 'REJETE'.")
        
    connection = get_db_connection()
    if not connection:
        raise HTTPException(status_code=500, detail="Base de données inaccessible.")
    cursor = connection.cursor(dictionary=True)
    
    try:
        # Vérifier si l'agent existe
        cursor.execute("SELECT id_agent FROM agent_admission WHERE id_agent = %s", (data.id_agent,))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Médecin introuvable.")
            
        # Mise à jour du statut d'approbation
        query_update = "UPDATE agent_admission SET statut = %s WHERE id_agent = %s"
        cursor.execute(query_update, (data.action, data.id_agent))
        
        enregistrer_audit(cursor, admin_connecte["user_id"], "VALIDATION_AGENT", f"Agent #{data.id_agent} → {data.action}", request.client.host)

        connection.commit()
        return {"statut": "Succès", "message": f"Le compte du l'agent a été mis à jour avec le statut : {data.action}."}
    finally:
        cursor.close()
        connection.close()



# --- 5. SURVEILLER LES LOGS D'AUDIT (Sécurité du système) ---
@app.get("/admin/logs/audit", summary="Consulter l'historique des actions critiques effectuées sur la plateforme")
def consulter_logs_audit(limit: int = 50, admin_connecte: dict = Depends(verifier_badge_admin)):
    connection = get_db_connection()
    if not connection:
        raise HTTPException(status_code=500, detail="Base de données inaccessible.")
    cursor = connection.cursor(dictionary=True)
    
    try:
        # Récupération des traces de sécurité (connexions, modifications, calculs d'IA)
        query = """
            SELECT id_log, id_user, action, details, ip_adress, 
                   DATE_FORMAT(date_action, '%Y-%m-%d %H:%M:%S') as date_action
            FROM audit_log
            ORDER BY date_action DESC
            LIMIT %s
        """
        cursor.execute(query, (limit,))
        return cursor.fetchall()
    finally:
        cursor.close()
        connection.close()


#6.
@app.put("/admin/medecins/{id_medecin}/suspendre", summary="Suspendre un médecin ou un agent en passant son statut à REJETE")
def suspendre_medecin(id_medecin: int,request: Request, admin_connecte: dict = Depends(verifier_badge_admin)):
    connection = get_db_connection()
    if not connection:
        raise HTTPException(status_code=500, detail="Base de données inaccessible.")
    cursor = connection.cursor(dictionary=True)
        
    try:
        query = "UPDATE medecin SET statut = 'REJETE' WHERE id_medecin = %s"
        cursor.execute(query, (id_medecin,))
        
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Médecin introuvable.")

        enregistrer_audit(cursor, admin_connecte["user_id"], "SUSPENSION_MEDECIN", f"Médecin #{id_medecin} suspendu", request.client.host)
 
        connection.commit()

        return {"statut": "Succès", "message": "Le compte du médecin a été suspendu (Statut: REJETE)."}
    finally:
        cursor.close()
        connection.close()







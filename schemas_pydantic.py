from pydantic import BaseModel, EmailStr,Field


# --- MODÈLES DE VALIDATION PYDANTIC ---
#connection login
class LoginRequest(BaseModel):
    """Structure attendue lors d'une tentative de connexion"""
    email: EmailStr
    mot_de_passe: str


class MedecinRegisterRequest(BaseModel):
    """Structure requise pour l'inscription d'un nouveau médecin"""
    email: EmailStr
    mot_de_passe: str
    telephone: str
    nom: str
    prenom: str
    num_ordre_cnom:str
    code_inpe:str
    specialite: str  #  'CARDIOLOGUE' ou 'DIABETOLOGUE'


class AgentRegisterRequest(BaseModel):
    """Structure requise pour l'inscription d'un nouveau agent d'admission"""
    nom_agent: str
    prenom_agent: str
    matricule:str  #Identifiant RH unique pour l'inscription et la vérification
    service_affectation:str  #Ex: Consultations Externes
    email: EmailStr
    mot_de_passe: str
    telephone: str

 

class PatientCreateRequest(BaseModel):
    """Structure requise pour qu'un médecin crée un nouveau dossier patient"""
    email: EmailStr
    telephone: str
    nom: str
    prenom: str
    cin:str
    couverture_medicale:str #'AMO', 'MUTUELLE', 'SANS_COUVERTURE'
    date_naissance: str  # Format attendu : YYYY-MM-DD
    gender: int          # 1 pour Femme, 2 pour Homme (aligné sur l'IA)
    


class ConsultationRequest(BaseModel):
    """
    Structure regroupant l'ensemble des constantes cliniques et biologiques
    requises pour l'historique médical et l'inférence des modèles d'IA (Cardio & Diabète).
    """
    id_patient: int
    
    # Guidage du choix des modèles IA
    flux_prediction: str # 'CARDIOLOGUE' ou 'DIABETOLOGUE'

    # Données Morphologiques
    weight: float
    height: float
   

    # Paramètres Cardiovasculaires (Dataset Cardio)
    ap_hi: int          # Pression artérielle systolique
    ap_lo: int          # Pression artérielle diastolique
    cholesterol: int    # 1: Normal, 2: Élevé, 3: Très élevé
    gluc: int           # 1: Normal, 2: Élevé, 3: Très élevé
    smoke: int          # 0: Non, 1: Oui
    alco: int          # 0: Non, 1: Oui
    active: int         # 0: Non, 1: Oui
    
    # Paramètres Biologiques (Dataset Diabète)
    antecedent_hypertension: int  # 0 ou 1
    antecedent_heart_disease: int # 0 ou 1
    smoking_history: str # 'never', 'current', 'former', etc.
    HbA1c_level: float   # Taux d'hémoglobine glyquée
    blood_glucose_level: int # Glycémie à jeun
    
    # Note libre
    commentaire_medecin: str|None = None


# Modèle pour la liste des patients du médecin
class PatientListItem(BaseModel):
    id_patient: int
    ipp:str
    nom: str
    prenom: str
    date_naissance: str|None = None
    gender: int
    cin :str
    couverture_medicale :str
    statut_lien: str


# Modèle pour l'historique des consultations d'un patient
class ConsultationHistoryItem(BaseModel):
    id_consultation: int
    date_consultation: str
    type_consultation: str
    score_cardio: float|None = None
    score_diabete: float|None = None


# Modèle pour lier un patient via son code d'activation
class LiaisonRequest(BaseModel):
    ipp: str

# Modèle pour mettre à jour le commentaire clinique
class CommentaireUpdateRequest(BaseModel):
    commentaire_medecin: str


# Modèles Pydantic pour les requêtes Admin
class ApprobationMedecinRequest(BaseModel):
    id_medecin: int
    action: str # 'APPROUVE' ou 'REJETE'


class ApprobationAgentRequest(BaseModel):
    id_agent: int
    action: str # 'APPROUVE' ou 'REJETE'

# --- Schéma de validation pour la requête ---
class ChangementMotDePasse(BaseModel):
    ancien_mot_de_passe: str = Field(..., description="Le mot de passe actuel du patient")
    nouveau_mot_de_passe: str = Field(..., min_length=6, description="Le nouveau mot de passe (min 6 caractères)")
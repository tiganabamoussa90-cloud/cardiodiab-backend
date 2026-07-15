import os
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone
import jwt
from fastapi import HTTPException, status, Depends
from fastapi.security import OAuth2PasswordBearer
import bcrypt

load_dotenv()

SECRET_KEY = os.environ.get("SECRET_KEY")
ALGORITHM = "HS256"

if not SECRET_KEY:
    raise RuntimeError("SECRET_KEY manquant : vérifiez votre fichier .env ou les variables d'environnement du serveur.")

# ... le reste du fichier ne change pas (hash_password, verifier_badge_medecin, etc.)

def hash_password(mot_de_passe: str) -> str:
    """Hash un mot de passe en clair avec bcrypt avant stockage en base."""
    return bcrypt.hashpw(mot_de_passe.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

def verifier_mot_de_passe(mot_de_passe_clair: str, mot_de_passe_hash: str) -> bool:
    """Vérifie un mot de passe en clair contre son hash bcrypt stocké."""
    try:
        return bcrypt.checkpw(mot_de_passe_clair.encode("utf-8"), mot_de_passe_hash.encode("utf-8"))
    except ValueError:
        # Compte historique encore en clair (avant ce correctif) : compatibilité temporaire.
        return mot_de_passe_clair == mot_de_passe_hash
    
    

# Cette ligne indique à FastAPI où chercher le badge (dans les en-têtes des requêtes)
# c'est comme tu dis à FastAPI les routes protégées utiliseront un token Bearer obtenu via /auth/login
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

def fabriquer_badge(user_id:int,role:str)-> str:
    """
    Crée un Token (badge) contenant l'ID et le Rôle de l'utilisateur.
    Le badge expire automatiquement après 2 heures.
    """
    temps_expiration=datetime.now(timezone.utc) + timedelta(hours=2)
    contenu_badge = {
        "sub": str(user_id),
        "role": role,
        "exp": temps_expiration
    }

    # On signe le badge avec notre clé secrète
    token_genere = jwt.encode(contenu_badge,SECRET_KEY,algorithm=ALGORITHM)
    return token_genere

def verifier_badge_medecin(token:str=Depends(oauth2_scheme))-> dict:
    """
    Intercepteur qui lit le badge. Si le badge est faux, expiré 
    ou si ce n'est pas un médecin, il bloque l'accès immédiatement.
    """
    erreur_authentification=HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Badge invalide ou session expirée. Veuillez vous reconnecter."
    )

    try:
        #on décode et vérifie la signature du badge
        payload=jwt.decode(token,SECRET_KEY,algorithms=[ALGORITHM])
        user_id=payload.get("sub")
        role=payload.get("role")

        if user_id is None or role is None:
            raise erreur_authentification
        
        # Sécurité : On s'assure que c'est bien un médecin (Généraliste ou Spécialiste)
        if role not in ["CARDIOLOGUE", "DIABETOLOGUE"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Accès refusé : Seuls les médecins peuvent effectuer cette action."
            )
        
        return {"user_id":int(user_id),"role":role}

    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401,detail="Votre session a expiré.")
    except jwt.PyJWTError:
        raise erreur_authentification



def verifier_badge_patient(token: str = Depends(oauth2_scheme)) -> dict:
    """Intercepteur qui valide le badge et vérifie que c'est un patient connecté"""
    erreur_auth = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Badge invalide ou session expirée. Veuillez vous reconnecter."
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        role = payload.get("role")
        
        if user_id is None or role is None:
            raise erreur_auth
            
        if role != "PATIENT":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Accès refusé : Cette zone est réservée aux patients."
            )
            
        return {"user_id": int(user_id), "role": role}
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401,detail="Votre session a expiré.")
        
    except jwt.PyJWTError:
        raise erreur_auth


def verifier_badge_admin(token: str = Depends(oauth2_scheme)) -> dict:
    """Intercepteur qui valide le badge et vérifie que l'utilisateur est un Administrateur"""
    erreur_auth = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Session admin invalide ou expirée."
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        role = payload.get("role")
        
        if user_id is None or role is None:
            raise erreur_auth
            
        if role != "ADMIN":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Accès interdit : Cette zone nécessite des droits d'administration."
            )
            
        return {"user_id": int(user_id), "role": role}
        
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401,detail="Votre session a expiré.")
        
    except jwt.PyJWTError:
        raise erreur_auth



def verifier_badge_agent(token: str = Depends(oauth2_scheme)) -> dict:
    """Intercepteur qui valide le badge et vérifie que l'utilisateur est un Agent d'admission"""
    erreur_auth = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Session agent invalide ou expirée."
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        role = payload.get("role")
        
        if user_id is None or role is None:
            raise erreur_auth
            
        if role != "AGENT":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Accès interdit : Cette zone nécessite des droits d'agent d'admission de l'accueil."
            )
            
        return {"user_id": int(user_id), "role": role}
        
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401,detail="Votre session a expiré.")
        
    except jwt.PyJWTError:
        raise erreur_auth



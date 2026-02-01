import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import time
import torch

# --- 1. CONFIGURATION & CSS (DESIGN GEMINI) ---
st.set_page_config(page_title="SQuAD Chat", page_icon="✨", layout="wide")

st.markdown("""
<style>
    /* IMPORT FONTS */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    .stApp { background-color: #F8FAFC; font-family: 'Inter', sans-serif; }

    /* SIDEBAR BLEUE */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0F172A 0%, #1E40AF 100%);
        color: white;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] label, [data-testid="stSidebar"] p {
        color: #E2E8F0 !important;
    }
    .stTextArea textarea {
        border-radius: 10px;
        background-color: rgba(255,255,255,0.95);
        color: #1e293b;
        font-size: 0.9rem;
    }

    /* CHAT BUBBLES */
    .user-bubble {
        background-color: #EFF6FF; color: #1E293B; padding: 15px 20px;
        border-radius: 20px 20px 5px 20px; margin-bottom: 15px; text-align: right;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05); max-width: 80%; margin-left: auto;
    }
    
    /* INPUT & BOUTON ROND */
    .stTextInput input {
        border-radius: 30px !important; padding: 15px 25px !important;
        border: 1px solid #CBD5E1 !important;
    }
    div.stButton > button {
        border-radius: 50% !important; width: 55px; height: 55px;
        background: linear-gradient(135deg, #2563EB 0%, #4F46E5 100%);
        color: white; border: none; display: flex; align-items: center; justify-content: center;
        font-size: 24px; padding: 0 !important;
    }
    div.stButton > button:hover { transform: scale(1.1); }
</style>
""", unsafe_allow_html=True)

# --- 2. DONNÉES DE DÉMO (BASE DE CONNAISSANCE) ---
knowledge_base = {
    "✍️ Personnalisé (Vide)": "",
    "🎓 Université UVSQ": """L'Université de Versailles-Saint-Quentin-en-Yvelines (UVSQ) est une université française créée en 1991. Elle est située dans le département des Yvelines et, depuis 2002, dans celui des Hauts-de-Seine. Elle compte environ 19 000 étudiants, répartis en cinq UFR, trois instituts et une école d'ingénieurs.""",
    "🐍 Langage Python": """Python est un langage de programmation interprété, multiparadigme et multiplateformes. Il favorise la programmation impérative structurée, fonctionnelle et orientée objet. Il a été créé par Guido van Rossum et publié pour la première fois en 1991.""",
    "🗼 Tour Eiffel": """La tour Eiffel est une tour de fer puddlé de 330 m de hauteur située à Paris. Construite par Gustave Eiffel et ses collaborateurs pour l'Exposition universelle de Paris de 1889, elle est devenue le symbole de la capitale française."""
}

# --- 3. FONCTIONS DE CHARGEMENT ---
@st.cache_resource
def load_model(path):
    try:
        # Détection automatique du GPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSeq2SeqLM.from_pretrained(path).to(device)
        return tokenizer, model, device, None
    except Exception as e:
        return None, None, "cpu", str(e)

# --- 4. SIDEBAR : CONTEXTE ET MODÈLE ---
with st.sidebar:
    st.title("⚙️ Configuration")
    
    # A. SÉLECTION DU MODÈLE
    st.markdown("### 1. Moteur IA")
    model_options = {
        "T5": "/models",
        "🧪 Mon Modèle Fine-tuné": "./models/t5-squad-v2-final",
        "📉 T5-Small (Base)": "t5-small"
    }
    selected_name = st.selectbox("Version du modèle", list(model_options.keys()), label_visibility="collapsed")
    
    # Chargement
    with st.spinner("Chargement du cerveau..."):
        tokenizer, model, device, err = load_model(model_options[selected_name])
    
    if err:
        st.error("Modèle introuvable. Utilisez la démo.")
    else:
        st.success(f"Connecté ({device.upper()})")

    st.markdown("---")
    
    # B. GESTION DU CONTEXTE (AUTO-REMPLISSAGE)
    st.markdown("### 2. Document Source")
    
    # Sélecteur de sujet (Le petit plus pour la démo !)
    selected_topic = st.selectbox("Sujets Prédéfinis :", list(knowledge_base.keys()))
    
    # Logique de mise à jour du texte
    if "context" not in st.session_state:
        st.session_state.context = knowledge_base["🎓 Université UVSQ"]
    
    # Si l'utilisateur change le sujet dans le menu, on met à jour la zone de texte
    if selected_topic != "✍️ Personnalisé (Vide)" and knowledge_base[selected_topic] != st.session_state.context:
         # On vérifie si le texte actuel correspond déjà au sujet pour ne pas écraser les modifs manuelles
         # Petite astuce : on force la mise à jour si on change de menu
         st.session_state.context = knowledge_base[selected_topic]

    st.info("L'IA cherchera la réponse ICI 👇")
    
    # Zone de texte modifiable
    context_text = st.text_area(
        "Contenu du document", 
        value=st.session_state.context, 
        height=300, 
        label_visibility="collapsed"
    )
    # On sauvegarde ce que l'utilisateur tape (même si c'est manuel)
    st.session_state.context = context_text

# --- 5. ZONE PRINCIPALE (HISTORIQUE CHAT) ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Bonjour ! Sélectionnez un sujet à gauche (ou collez votre propre texte), puis posez-moi une question."}]

st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

# Affichage des bulles
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"""<div class="user-bubble"><strong>Vous</strong><br>{msg["content"]}</div>""", unsafe_allow_html=True)
    else:
        with st.chat_message("assistant", avatar="✨"):
             st.markdown(msg["content"])

st.markdown("</div>", unsafe_allow_html=True)

# --- 6. ZONE DE SAISIE ---
st.markdown("---") 

with st.form(key="chat_input_form", clear_on_submit=True):
    col_input, col_btn = st.columns([8, 1])
    
    with col_input:
        user_input = st.text_input(
            "Votre question...", 
            placeholder="Posez votre question à l'IA...", 
            label_visibility="collapsed"
        )
    
    with col_btn:
        submit_btn = st.form_submit_button("➤")

# --- 7. LOGIQUE DE RÉPONSE ---
if submit_btn and user_input:
    # 1. Ajout message utilisateur
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # 2. Génération
    with st.chat_message("assistant", avatar="✨"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Check context
        if not context_text or len(context_text) < 5:
            final_answer = "⚠️ J'ai besoin d'un contexte dans la barre latérale pour répondre."
            
        else:
            prompt = f"question: {user_input} context: {context_text}"
            
            try:
                # Tokenization sur le bon device (CPU ou GPU)
                inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(device)
                
                # Génération
                outputs = model.generate(inputs.input_ids, max_length=128)
                raw_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Gestion du "Unanswerable"
                if "unanswerable" in raw_answer.lower():
                    final_answer = "❌ Je ne trouve pas la réponse dans ce texte."
                else:
                    final_answer = raw_answer
                    
            except Exception as e:
                final_answer = f"Erreur : {e}"

        # 3. Animation de frappe
        for chunk in final_answer.split(" "):
            full_response += chunk + " "
            time.sleep(0.05)
            message_placeholder.markdown(full_response + "▌")
            
        message_placeholder.markdown(full_response)
    
    # 4. Sauvegarde
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    st.rerun()
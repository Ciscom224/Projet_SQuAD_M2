import streamlit as st
import time
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForQuestionAnswering

# --- 1. CONFIGURATION & CSS (DESIGN GEMINI) ---
st.set_page_config(page_title="Projet SQuAD ", page_icon="🧠", layout="wide")

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
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] label, [data-testid="stSidebar"] p, [data-testid="stSidebar"] span {
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
        border: 1px solid #4F46E5 !important;
    }
    div.stButton > button {
        border-radius: 50% !important; width: 55px; height: 55px;
        background: linear-gradient(135deg, #2563EB 0%, #4F46E5 100%);
        color: white; border: none; display: flex; align-items: center; justify-content: center;
        font-size: 24px; padding: 0 !important;
    }

    div.stButton > button:hover { 
        transform: scale(1.1); 
        box-shadow: 0 6px 8px rgba(0,0,0,0.3);
            
    }
</style>
""", unsafe_allow_html=True)

# --- 2. BASE DE CONNAISSANCE ---
knowledge_base = {
    "✍️ Personnalisé": "",
    "🎓 Master DATASCALE": """Le M2 DataScale forme des experts dotés d’une double compétence en ingénierie et analyse des données. Il couvre l’administration de grands volumes de données, l’analyse de données de capteurs, la protection de la vie privée et la prédiction de phénomènes complexes, en s’appuyant sur la fouille de données, le machine learning et l’IA.
La formation prépare à des métiers variés : Data Engineer, Data Scientist, IA Analyst, CDO, DBA ou Urbaniste SI.
Le programme combine un tronc commun, des options spécialisées et des modules de professionnalisation (projets et séminaires).
Responsables : Mustapha Lebbah et Zoubida Kedad-Cointot.""",
    "🐍 Langage Python": """Python est un langage de programmation interprété, multiparadigme et multiplateformes. Il favorise la programmation impérative structurée, fonctionnelle et orientée objet. Il a été créé par Guido van Rossum et publié pour la première fois en 1991.""",
    "🗼 Tour Eiffel": """La tour Eiffel est une tour de fer puddlé de 330 m de hauteur située à Paris. Construite par Gustave Eiffel et ses collaborateurs pour l'Exposition universelle de Paris de 1889, elle est devenue le symbole de la capitale française."""
}

# --- 3. CHARGEMENT HYBRIDE (OPTIMISÉ MAC M1/M2/M3) ---
@st.cache_resource
def load_model(model_info):
    path = model_info["path"]
    model_type = model_info["type"]
    
    # 1. DÉTECTION INTELLIGENTE DU MATÉRIEL
    if torch.cuda.is_available():
        device = "cuda" # Pour PC avec NVIDIA
    elif torch.backends.mps.is_available():
        device = "mps"  # <--- C'EST ICI POUR VOTRE MAC ! 🍎
    else:
        device = "cpu"  # Sinon, on utilise le processeur classique
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(path)
        
        # SÉLECTION DU BON ARCHITECTURE
        if model_type == "seq2seq":
            # Pour T5
            model = AutoModelForSeq2SeqLM.from_pretrained(path).to(device)
        else:
            # Pour BERT et ALBERT (Question Answering)
            model = AutoModelForQuestionAnswering.from_pretrained(path).to(device)
            
        return tokenizer, model, device, None
    except Exception as e:
        return None, None, "cpu", str(e)

# --- 4. SIDEBAR : CONFIGURATION ---
with st.sidebar:
    st.title("⚙️ Configuration")  # <--- TITRE RAJOUTÉ ICI
    
    model_options = {
        "🏆 Model T5 ": {
            "path": "ciscom224/fine-tuning-t5-small-model-for-squad", 
            "type": "seq2seq"
        },
        "🦁 Model BERT": {
            "path": "bert-large-uncased-whole-word-masking-finetuned-squad",
            "type": "extractive"
        },
        "⚡ Model ALBERT ": {
            "path": "models/albert/checkpoint-8000", 
            "type": "extractive"
        }
    }
    
    selected_name = st.selectbox("Choisir le modèle", list(model_options.keys()))
    current_info = model_options[selected_name]

    with st.spinner(f"Chargement de {selected_name}..."):
        tokenizer, model, device, err = load_model(current_info)
    
    if err:
        st.error(f"Erreur de chargement : {err}")
    else:
        st.success(f"Prêt ({device.upper()})")
        if current_info["type"] == "seq2seq":
            st.caption("📝 Mode : Génération (Reformulation)")
        else:
            st.caption("🔍 Mode : Extraction (Surlignage)")

    st.markdown("---")

    # GESTION DU CONTEXTE
    st.markdown("### Document Source")
    selected_topic = st.selectbox("Sujets Prédéfinis :", list(knowledge_base.keys()))
    
    if "context" not in st.session_state:
        st.session_state.context = knowledge_base["🎓 Master DATASCALE"]
        
    # Mise à jour si changement de sélection
    if selected_topic != "✍️ Personnalisé (Vide)" and knowledge_base[selected_topic] != st.session_state.context:
        st.session_state.context = knowledge_base[selected_topic]
        
    context_text = st.text_area("Contenu du document", value=st.session_state.context, height=300)
    st.session_state.context = context_text

# --- 5. TITRE PRINCIPAL & CHAT ---
st.title("🧠 Projet SQuAD") # <--- TITRE RAJOUTÉ ICI

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Bonjour ! Je suis prêt à analyser votre texte. Posez une question."}]

st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
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
        user_input = st.text_input("Votre question...", placeholder="Posez votre question...", label_visibility="collapsed")
    with col_btn:
        submit_btn = st.form_submit_button("➤")

# --- 7. LOGIQUE D'INFÉRENCE UNIFIÉE ---
if submit_btn and user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant", avatar="✨"):
        message_placeholder = st.empty()
        full_response = ""

        if not context_text or len(context_text) < 5:
            final_answer = "⚠️ Veuillez fournir un contexte plus long."
        else:
            try:
                # --- CAS A : T5 (Génératif) ---
                if current_info["type"] == "seq2seq":

                    prompt = f"question: {user_input} context: {context_text}"
                    try:
                        # Tokenization sur le bon device (CPU ou GPU)
                        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(device)
                        
                        # Génération
                        outputs = model.generate(
                            inputs.input_ids,
                            max_length=128,   # Assurez-vous que c'est assez grand
                            num_beams=4,      # Augmentez un peu (4 -> 5) pour qu'il explore plus
                            length_penalty=2.5, 
                            early_stopping=True,
                            no_repeat_ngram_size=2
                        )
                        raw_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
                        
                        # Gestion du "Unanswerable"
                        if "unanswerable" in raw_answer.lower():
                            final_answer = "❌ Déesolé!  Cette question est hors contexte."
                        else:
                            final_answer = raw_answer
                            
                    except Exception as e:
                        final_answer = f"Erreur : {e}"

                # --- CAS B : BERT / ALBERT (Extractif) ---
                else:
                    inputs = tokenizer(user_input, context_text, return_tensors="pt", max_length=512, truncation=True).to(device)
                    
                    with torch.no_grad():
                        outputs = model(**inputs)
                    
                    # Logique : On prend le meilleur début et la meilleure fin
                    start_idx = torch.argmax(outputs.start_logits)
                    end_idx = torch.argmax(outputs.end_logits)
                    
                    # Vérification SQuAD v2 (Si fin < début, c'est impossible -> pas de réponse)
                    if end_idx < start_idx:
                        final_answer = "🚫 Désolé!!! Pas de réponse trouvée dans ce contexte."
                    else:
                        tokens = inputs.input_ids[0][start_idx : end_idx + 1]
                        final_answer = tokenizer.decode(tokens, skip_special_tokens=True)
                        
                        # Nettoyage
                        final_answer = final_answer.replace("[CLS]", "").replace("[SEP]", "").strip()
                        if not final_answer: 
                             final_answer = "🚫 Pas de réponse trouvée."

            except Exception as e:
                final_answer = f"Erreur technique : {e}"

        # Animation d'écriture
        for chunk in final_answer.split(" "):
            full_response += chunk + " "
            time.sleep(0.05)
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
    st.rerun()
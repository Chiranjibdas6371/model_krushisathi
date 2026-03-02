import os
import uuid
import pickle
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from flask_cors import CORS
from dotenv import load_dotenv
load_dotenv()
from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

import google.generativeai as genai

import azure.cognitiveservices.speech as speechsdk
# ---------------- GEMINI SETUP ----------------
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
ai_model = genai.GenerativeModel("models/gemini-flash-lite-latest")


speech_key = os.getenv("AZURE_SPEECH_KEY")
speech_region = os.getenv("AZURE_SPEECH_REGION")

speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
speech_config.speech_recognition_language = "or-IN"
speech_config.speech_synthesis_voice_name = "or-IN-MadhuraNeural"

def validate_chemical_safety(advice):

    for chem in SAFE_CHEMICALS:
        if chem.lower() in advice.lower():
            return advice + f"\n\nSafe Dosage: {SAFE_CHEMICALS[chem]}"

    if "spray" in advice.lower():
        return advice + "\n\n⚠ Chemical not verified. Consult agriculture officer."

    return advice

def translate_to_odia(text):

    prompt = f"Translate to simple Odia for farmers:\n{text}"

    try:
        response = ai_model.generate_content(prompt)
        return response.text
    except:
        return text
    

def speech_to_text(audio_path):

    audio_config = speechsdk.AudioConfig(filename=audio_path)

    recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config,
        audio_config=audio_config
    )

    result = recognizer.recognize_once()

    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        return result.text
    return ""


SAFE_CHEMICALS = {
    "Neem oil": "2-3 ml per litre",
    "Imidacloprid": "0.3 ml per litre",
    "Chlorpyrifos": "2 ml per litre",
    "Mancozeb": "2 gm per litre",
    "Carbendazim": "1 gm per litre",
    "Copper oxychloride": "3 gm per litre"
}

def detect_problem_type(query):

    if not query:
        return "general"

    prompt = f"""
Classify this farmer query into one word:

disease
pest
soil
water
fertilizer
general

Query: {query}
"""

    try:
        result = ai_model.generate_content(prompt)

        if not result or not result.text:
            return "general"

        output = result.text.lower()

        if "pest" in output:
            return "pest"
        elif "disease" in output:
            return "disease"
        elif "soil" in output:
            return "soil"
        elif "water" in output:
            return "water"
        elif "fertilizer" in output:
            return "fertilizer"
        else:
            return "general"

    except:
        return "general"

def detect_local_agri_intent(text):

    if not text:
        return "general"

    text = text.lower()

    pest_words = ["ପୋକ","कीड़ा","insect"]
    disease_words = ["ରୋଗ","disease"]
    soil_words = ["ମାଟି","soil"]
    water_words = ["ପାଣି","water"]
    fertilizer_words = ["ସାର","fertilizer"]

    if any(w in text for w in pest_words):
        return "pest"

    if any(w in text for w in disease_words):
        return "disease"

    if any(w in text for w in soil_words):
        return "soil"

    if any(w in text for w in water_words):
        return "water"

    if any(w in text for w in fertilizer_words):
        return "fertilizer"

    return "general"

def gemini_advice(question, disease="", language="english"):

    if not question:
        return "Please describe the issue."

    lang_instruction = "Reply in simple English."
    if language == "odia":
        lang_instruction = "Reply in simple Odia language for farmers."

    prompt = f"""
Farmer asked: {question}
Detected issue: {disease}

Rules:
if farmer greets or says thanks → reply with polite greeting
if query is about soil or water → give direct advice without asking for image
if query is about fertilizer → give direct advice without asking for image
if query is about exit or any quit or stop words → reply with polite thank you for using service
 
If query is about pest or disease AND image is not provided →
Reply:
"Please upload crop image for accurate diagnosis."

If query is general →
Give direct solution.

If disease is known →
Give:
1. Organic solution
2. Chemical solution

Use simple language.
Never guess disease.
Reply only in selected language.
"""

    try:
        response = ai_model.generate_content(prompt)

        if not response or not response.text:
            return "AI could not generate advice."

        return response.text

    except Exception as e:
        print("Gemini Error:", e)
        return "AI advice unavailable."
    
def detect_crop_issue(filepath):

    img = preprocess_pest_image(filepath)
    preds = pest_model.predict(img)
    idx = np.argmax(preds)

    # Decode label using encoder
    if hasattr(pest_labels, 'classes_'):
        issue = pest_labels.classes_[idx]
    else:
        issue = pest_labels.inverse_transform([idx])[0]

    confidence = float(preds[0][idx])

    return issue, confidence

# ---------------- PEST MODEL ----------------
pest_model = tf.keras.models.load_model("plant_disease_cnn_model_final.h5", compile=False)

with open("labels_encoder.pkl", "rb") as f:
    pest_labels = pickle.load(f)



# ---------------- FERTILIZER MODEL ----------------
fert_model = joblib.load("fertilizer_dt_model.pkl")
soil_encoder = joblib.load("soil_encoder.pkl")
crop_encoder = joblib.load("crop_encoder.pkl")
fertilizer_encoder = joblib.load("fertilizer_encoder.pkl")

# ---------------- YIELD MODEL ----------------
crop_model = joblib.load("crop_pred_gradient_boosting_model.pkl")
crop_name_enc = joblib.load("le_crop.pkl")
crop_season_enc = joblib.load("le_season.pkl")
crop_state_enc = joblib.load("le_state.pkl")

with open("crop_model.pkl", "rb") as f:
    loaded_model = pickle.load(f)


app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
CORS(app)

def preprocess_pest_image(filepath):
    img = image.load_img(filepath, target_size=(100,100), color_mode="grayscale")
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def predict_pest(filepath):
    img = preprocess_pest_image(filepath)
    preds = pest_model.predict(img)
    idx = np.argmax(preds)

    if hasattr(pest_labels, 'classes_'):
        pest = pest_labels.classes_[idx]
    else:
        pest = pest_labels.inverse_transform([idx])[0]

    confidence = float(preds[0][idx])

    return {"pest": pest, "confidence": confidence}

@app.route("/predict", methods=["POST"])
def pest_predict():

    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400

    file = request.files["file"]
    filename = secure_filename(file.filename)
    unique_name = str(uuid.uuid4()) + "_" + filename
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)

    file.save(filepath)

    try:
        result = predict_pest(filepath)
        return jsonify(result)
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

class_label = {
    20: 'rice', 11: 'maize', 3: 'chickpea',
    9: 'kidneybeans', 18: 'pigeonpeas', 13: 'mothbeans',
    14: 'mungbean', 2: 'blackgram', 10: 'lentil',
    19: 'pomegranate', 1: 'banana', 12: 'mango',
    7: 'grapes', 21: 'watermelon', 15: 'muskmelon',
    0: 'apple', 16: 'orange', 17: 'papaya',
    4: 'coconut', 6: 'cotton', 8: 'jute', 5: 'coffee'
}


crop_nutrient_req = {
    "Arhar/Tur": {"N": 25, "P": 50, "K": 25},
    "Bajra": {"N": 50, "P": 30, "K": 20},
    "Barley": {"N": 60, "P": 30, "K": 20},
    "Coriander": {"N": 40, "P": 25, "K": 20},
    "Cotton (Lint)": {"N": 100, "P": 50, "K": 50},
    "Cowpea (Lobia)": {"N": 20, "P": 40, "K": 20},
    "Dry Chillies": {"N": 80, "P": 40, "K": 40},
    "Garlic": {"N": 100, "P": 50, "K": 50},
    "Ginger": {"N": 120, "P": 60, "K": 60},
    "Gram (Chickpea)": {"N": 20, "P": 40, "K": 20},
    "Groundnut": {"N": 25, "P": 50, "K": 25},
    "Jowar": {"N": 60, "P": 30, "K": 20},
    "Linseed (Flax)": {"N": 40, "P": 20, "K": 20},
    "Maize (Grain)": {"N": 120, "P": 60, "K": 40},
    "Maize (Fodder)": {"N": 100, "P": 50, "K": 30},
    "Masoor (Red Lentil)": {"N": 20, "P": 40, "K": 20},
    "Moong (Green Gram)": {"N": 20, "P": 40, "K": 20},
    "Onion": {"N": 120, "P": 60, "K": 40},
    "Peas & Beans (Pulses)": {"N": 25, "P": 50, "K": 25},
    "Potato": {"N": 150, "P": 60, "K": 100},
    "Ragi (Finger Millet)": {"N": 60, "P": 30, "K": 20},
    "Rapeseed & Mustard": {"N": 60, "P": 40, "K": 40},
    "Rice": {"N": 120, "P": 60, "K": 40},
    "Safflower": {"N": 40, "P": 40, "K": 40},
    "Sugarcane": {"N": 200, "P": 60, "K": 120},
    "Sunflower": {"N": 80, "P": 40, "K": 40},
    "Turmeric": {"N": 120, "P": 60, "K": 60},
    "Urad (Black Gram)": {"N": 20, "P": 40, "K": 20},
    "Urad Bean": {"N": 20, "P": 40, "K": 20},
    "Wheat": {"N": 120, "P": 60, "K": 40}
}

fertilizer_content = {
    "MOP": {"N": 0, "P": 0, "K": 60},
    "20-40-20": {"N": 20, "P": 40, "K": 20},
    "DAP": {"N": 18, "P": 46, "K": 0},
    "Urea": {"N": 46, "P": 0, "K": 0},
    "40-20-20": {"N": 40, "P": 20, "K": 20},
    "20-20-20": {"N": 20, "P": 20, "K": 20},
    "15-15-15": {"N": 15, "P": 15, "K": 15},
    "30-15-15": {"N": 30, "P": 15, "K": 15},
    "Compound NPK (0-20-20)": {"N": 0, "P": 20, "K": 20},
    "0-20-20": {"N": 0, "P": 20, "K": 20},
    "60-40-60": {"N": 60, "P": 40, "K": 60},
    "40-30-40": {"N": 40, "P": 30, "K": 40},
    "20-10-20": {"N": 20, "P": 10, "K": 20},
    "Organic Compost": {"N": 2, "P": 1, "K": 1},  
    "120-60-60": {"N": 120, "P": 60, "K": 60},
    "0-40-40": {"N": 0, "P": 40, "K": 40},
    "20-30-20": {"N": 20, "P": 30, "K": 20},
    "30-10-20": {"N": 30, "P": 10, "K": 20},
    "25-10-15": {"N": 25, "P": 10, "K": 15},
    "SSP": {"N": 0, "P": 16, "K": 0},
    "20-15-15": {"N": 20, "P": 15, "K": 15},
    "30-15-20": {"N": 30, "P": 15, "K": 20},
    "30-20-20": {"N": 30, "P": 20, "K": 20},
    "20-10-10": {"N": 20, "P": 10, "K": 10},
    "100-50-50": {"N": 100, "P": 50, "K": 50}
}

def calculate_fertilizer_quantity(fertilizer_type, crop_name, soil_n, soil_p, soil_k, field_size_ha):
    if crop_name not in crop_nutrient_req:
        return None, "Crop not found in database!"
    if fertilizer_type not in fertilizer_content:
        return None, "Fertilizer type not found!"

    N_deficit = max(0, crop_nutrient_req[crop_name]["N"] - soil_n)
    P_deficit = max(0, crop_nutrient_req[crop_name]["P"] - soil_p)
    K_deficit = max(0, crop_nutrient_req[crop_name]["K"] - soil_k)

    fert_N = fertilizer_content[fertilizer_type]["N"]
    fert_P = fertilizer_content[fertilizer_type]["P"]
    fert_K = fertilizer_content[fertilizer_type]["K"]

    qty_per_ha = 0
    nutrient_used = ""
    if fert_N > 0 and N_deficit > 0:
        qty_per_ha = N_deficit / (fert_N / 100)
        nutrient_used = "N"
    elif fert_P > 0 and P_deficit > 0:
        qty_per_ha = P_deficit / (fert_P / 100)
        nutrient_used = "P"
    elif fert_K > 0 and K_deficit > 0:
        qty_per_ha = K_deficit / (fert_K / 100)
        nutrient_used = "K"
    else:
        return 0, f"Soil already sufficient in nutrients for {fertilizer_type}!"

    total_qty = round(qty_per_ha * field_size_ha, 2)
    return total_qty, f"Fertilizer to supply {nutrient_used}"



@app.route("/")
def home():
    return " API Running"

@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        data = request.json 

        df_new = pd.DataFrame([{
            "temperature": data["temperature"],
            "humidity": data["humidity"],
            "soil_moisture": data["soil_moisture"],
            "soil_type": soil_encoder.transform([data["soil_type"]])[0],
            "crop_type": crop_encoder.transform([data["crop_type"]])[0],
            "nitrogen": data["nitrogen"],
            "phosphorus": data["phosphorus"],
            "potassium": data["potassium"]
        }])

        pred_encoded = fert_model.predict(df_new.values)
        pred_label = fertilizer_encoder.inverse_transform(pred_encoded)
        fertilizer_type = pred_label[0]

        qty, message = calculate_fertilizer_quantity(
            fertilizer_type=fertilizer_type,
            crop_name=data["crop_type"],  
            soil_n=data["nitrogen"],
            soil_p=data["phosphorus"],
            soil_k=data["potassium"],
            field_size_ha=data["field_size_ha"]
        )

        return jsonify({
            "fertilizer": fertilizer_type,
            "quantity_kg": qty,
            "message": message
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route("/yield_predict", methods=["POST"])
def predict_yield():
    try:
        data = request.json

        new_df = pd.DataFrame([{
            "Crop": data["Crop"],
            "Crop_Year": data["Crop_Year"],
            "Season": data["Season"],
            "State": data["State"],
            "Area": data["Area"],
            "Production": data["Production"],
            "Annual_Rainfall": data["Annual_Rainfall"],
            "Fertilizer": data["Fertilizer"],
            "Pesticide": data["Pesticide"]
        }])

      
        new_df["Crop"] = new_df["Crop"].str.strip().str.lower()
        new_df["Season"] = new_df["Season"].str.strip().str.lower()
        new_df["State"] = new_df["State"].str.strip().str.lower()

        new_df["Crop"] = crop_name_enc.transform(new_df["Crop"])
        new_df["Season"] = crop_season_enc.transform(new_df["Season"])
        new_df["State"] = crop_state_enc.transform(new_df["State"])

        prediction_log = crop_model.predict(new_df)
        prediction = np.expm1(prediction_log)
        

        return jsonify({"predicted_yield": round(prediction[0], 2)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/crop_recommendation", methods=["POST"])
def crop_recommendation():
    try:
        data=request.json
        user_data = [[
            data["N"], data["P"], data["K"],
            data["temperature"], data["humidity"],
            data["ph"], data["rainfall"]
        ]]
        pred_encoded = loaded_model.predict(user_data)[0]
        recommended_crop = class_label[pred_encoded]
        return jsonify({"recommended_crop": recommended_crop})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/chatbot", methods=["POST"])
def chatbot():

    try:

        query = None
        detected_issue = None
        language = "english"   # default

        # -------- JSON TEXT INPUT --------
        if request.is_json:
            data = request.json
            query = data.get("query")
            language = data.get("language", "english")

       
        else:

            if "language" in request.form:
                language = request.form.get("language")

        
            if "query" in request.form:
                query = request.form.get("query")

           
            if "audio" in request.files:

                file = request.files["audio"]
                filename = secure_filename(file.filename)
                path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

                file.save(path)
                voice_text = speech_to_text(path)
                os.remove(path)

                if voice_text:
                    query = voice_text

    
            if "image" in request.files:

                img_file = request.files["image"]
                img_name = secure_filename(img_file.filename)
                img_path = os.path.join(app.config["UPLOAD_FOLDER"], img_name)

                img_file.save(img_path)
                detected_issue, conf = detect_crop_issue(img_path)
                os.remove(img_path)

       
        if not query and not detected_issue:
            return jsonify({
                "next_action": "ask_query",
                "message": "Please describe the crop issue."
            })

       
        if detected_issue:
            problem_type = "disease"
        else:

    # Step 1: Try local intent detection (Odia/Hindi safe)
            problem_type = detect_local_agri_intent(query)

    # Step 2: If still unclear → use Gemini
            if problem_type == "general":
                problem_type = detect_problem_type(query)

      
        if problem_type in ["disease", "pest"] and detected_issue is None:

            return jsonify({
                "next_action": "request_image",
                "recognized_text": query,
                "problem_type": problem_type,
                "message": "Please upload crop image for accurate diagnosis."
            })

      
        final_advice = gemini_advice(
            question=query if query else "Crop issue detected",
            disease=detected_issue if detected_issue else "",
            language=language
        )

        safe_advice = validate_chemical_safety(final_advice)

        return jsonify({
            "next_action": "advice",
            "recognized_text": query,
            "detected_crop_issue": detected_issue,
            "problem_type": problem_type,
            "advice": safe_advice
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ------------------------
# Run
# ------------------------



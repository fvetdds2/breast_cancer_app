import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys
import types
from pathlib import Path
import base64
from PIL import Image

#define weighted_logloss

def weighted_logloss(y_true: np.ndarray, y_pred: np.ndarray):
    grad = np.zeros_like(y_pred)
    hess = np.zeros_like(y_pred)
    return grad, hess

# add in the original training module name:
mod = types.ModuleType("model_train")
mod.weighted_logloss = weighted_logloss
sys.modules["model_train"] = mod

# Streamlit page setup
st.set_page_config(page_title="EmpowerHER: Know Your Risks. Lead Your Life. ", layout="centered")
st.markdown(
    '''
    <style>
     .block-container {
      width: 100%;
      max-width: 900px;
      margin: auto;
      padding: 1rem;
    }
   
    @media (max-width: 600px) {
      .block-container {
        width: 100%;
        padding: 0.75rem;
      }
    }
    @media (max-width: 1024px) and (min-width: 601px) {
      .block-container {
        width: 90%;
        padding: 1rem;
      }
    }
    </style>
    ''',
    unsafe_allow_html=True
)

def img_to_base64(img_path):
    with open(img_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

icon_base64 = img_to_base64("figures/title_icon2.png")

# Header design
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Source+Sans+Pro:wght@400;600&display=swap');

    /* Base styles */
    body {{ font-family: 'Source Sans Pro', sans-serif; background-color: #FAFAFB; color: #2E3A45; }}
    h1, h2, h3 {{ font-family: 'Playfair Display', serif; color: #2E3A45; }}

    /* Navbar */
    nav.navbar {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 1rem 2rem;
      background: linear-gradient(135deg, #7C3AED, #EC4899);
      box-shadow: 0 4px 12px rgba(0,0,0,0.1);
      margin-bottom: 1.5rem;
      border-radius: 8px;
    }}
    .navbar-logo span {{ color: #FFFFFF !important; font-size: 2rem; font-weight: 700; letter-spacing: 0.1em; }}
    .navbar-logo img {{ width: 48px; height: 48px; margin-right: 0.75rem; border-radius: 8px; }}

    /* Buttons */
    .stButton>button {{
      background: linear-gradient(135deg, #7C3AED, #EC4899);
      color: #FFFFFF;
      border: none;
      border-radius: 8px;
      padding: 0.5rem 1rem;
      font-family: 'Source Sans Pro', sans-serif;
      font-weight: 600;
      cursor: pointer;
      transition: opacity 0.3s ease;
    }}
    .stButton>button:hover {{ opacity: 0.9; }}

    /* Cards for steps or outputs */
    .risk-card {{ border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); padding: 1rem; background: #FFFFFF; margin-bottom: 1rem; }}

    /* Accent divider */
    hr.divider {{ border: none; height: 4px; background: #FBBF24; border-radius: 2px; margin: 1.5rem 0; }}
    </style>

    <nav class="navbar">
      <div class="navbar-logo">
        <img src="data:image/png;base64,{icon_base64}" alt="EmpowerHER Logo">
        <span>EmpowerHER</span>
      </div>
    </nav>

    <hr class="divider"/>
""", unsafe_allow_html=True)
st.markdown(
    """
    <style>
        .incidence-rate {
        font-size: 2.5rem;          
        font-weight: 600;
        margin: 1rem 0;
        color: #2A9D8F !important;  
        text-align: center;          
      }
    </style>
    """,
    unsafe_allow_html=True
)

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["About EmpowerHER", "Breast Cancer Risk Prediction", "Mind & Move", "Local support groups"])

# Load pre-trained model and threshold
BASE_DIR = Path(__file__).resolve().parent
model = joblib.load(BASE_DIR / "models" / "bcsc_xgb_model.pkl")
threshold = joblib.load(BASE_DIR / "models" / "threshold.pkl")

# Tab 1: About EmpowerHER application
with tab1:
    
    st.info("Research & Education use only: This tool is not a medical diagnosis.")
    st.markdown(
    """
    <div style="text-align:left; margin: 20px 0;">
      <h1 style="
        display: inline-block;
        padding: 5px 10px;
        margin: 0;
        color: #FFFFFF !important; 
        font-family: 'Montserrat', sans-serif;
        font-size: 1.5rem;
        border-radius: 6px;
        background: linear-gradient(135deg, #B292E5, #F4A8D4);
      ">
        ✨ About EmpowerHER ✨
      </h1>
    
    """,
    unsafe_allow_html=True
)
    st.markdown("""
EmpowerHER is a Streamlit web app that uses an XGBoost model trained on the Breast Cancer Surveillance Consortium **BCSC** cohort https://www.bcsc-research.org/index.php/datasets/rf/ to deliver personalized breast cancer risk predictions—combining your demographic and clinical inputs into an easy-to-understand probability score and actionable guidance. Predictions may be less reliable for populations under-represented e.g. certain ethnic groups.""")
    st.markdown(
    """
    <div style="text-align:left; margin: 20px 0;">
      <h1 style="
        display: inline-block;
        padding: 5px 10px;
        margin: 0;
        color: #FFFFFF !important; 
        font-family: 'Montserrat', sans-serif;
        font-size: 1.5rem;
        border-radius: 5px;
        background: linear-gradient(135deg, #B292E5, #F4A8D4);
      ">
        How we evaluate model performance
      </h1>
   """,
    unsafe_allow_html=True
)
    st.markdown("""
When you hear things like"overall accuracy", “precision” or “70% recall,” it can feel like jargon. Here’s what those numbers really mean and why we chose them, what they tell us about the model, and why they matter for you.
- **Overall accuracy**: The model’s overall accuracy is 77.6%, meaning it makes the correct prediction in about 8 out of every 10 cases. 
- **Precision**: Of all cases which the model flagged as cancer case, the percentage that actually developed cancer. A 0.4 precision means 4 out of every 10 flagged cases were true positives.
- **Recall (Sensitivity)**: Of all true cancer cases, the percentage the model correctly identified. A 0.916 recall means the model detected 9 out of every 10 real cases.
- **Brier Score**: The average squared difference between predicted probabilities and actual outcomes. Lower scores (closer to 0) indicate that probability estimates are more accurate. The model has Brier score = 0.1012.

""")
# USA breast cancer statistics 2025
    st.subheader("Current US Breast Cancer Statistics in 2025 from American Cancer Society")
    new_cases = "317,000"
    deaths = "42,680"
    incidence = "130.8 new cases per 100,000 women"

    st.metric("Estimated New invasive cases (2025)", new_cases)
    st.metric("Estimated Breast cancer deaths (2025)", deaths)
    st.metric("Age-adjusted incidence rate (females) (2025)", incidence)

# work flowchart 
    st.subheader("Key Visual Summaries")
    visuals = {
        "Prediction Work Flow":      "figures/EmpowerHER-flow-chart.png",
        }

# Display images
    for caption, rel_path in visuals.items():
        img_path = BASE_DIR / rel_path
        try:
            img = Image.open(img_path)
            width = 700 if caption == "Prediction Work Flow" else 500
            st.image(img, caption=caption, width=width)
        except FileNotFoundError:
            st.error(f"Unable to load image: {img_path}")

# Tab 2: Risk fortor Insights 
with tab2:
    st.subheader("Breast Cancer Risk Prediction")
    st.info("**Research & Education Use Only:** Not for medical diagnosis.")
    with st.expander("Enter information", expanded=True):
        def sel(label, opts):
            return st.selectbox(label, list(opts.keys()), format_func=lambda k: opts[k])
        age_groups  = {1:"18–29",2:"30–34",3:"35–39",4:"40–44",5:"45–49",6:"50–54",
                       7:"55–59",8:"60–64",9:"65–69",10:"70–74",11:"75–79",12:"80–84",13:">85"}
        race_eth    = {1:"White",2:"Black",3:"Asian/Pacific",4:"Native",5:"Hispanic",6:"Other"}
        menarche    = {0:">14",1:"12–13",2:"<12"}
        birth_age   = {0:"<20",1:"20–24",2:"25–29",3:">30",4:"Nulliparous"}
        fam_hist    = {0:"No",1:"Yes"}
        biopsy      = {0:"No",1:"Yes"}
        density     = {1:"Almost fat",2:"Scattered",3:"Hetero-dense",4:"Extremely dense"}
        hormone_use = {0:"No",1:"Yes"}
        menopause   = {1:"Pre/peri",2:"Post",3:"Surgical"}
        bmi_group   = {1:"10–24.9",2:"25–29.9",3:"30–34.9",4:"35+"}
        inputs = {
            "age_group":         sel("Age group", age_groups),
            "race_eth":          sel("Race/Ethnicity", race_eth),
            "age_menarche":      sel("Age at 1st period", menarche),
            "age_first_birth":   sel("Age at first birth", birth_age),
            "family_history":    sel("Family history", fam_hist),
            "personal_biopsy":   sel("Personal biopsy history", biopsy),
            "density":           sel("BI-RADS density", density),
            "hormone_use":       sel("Hormone use", hormone_use),
            "menopausal_status": sel("Menopausal status", menopause),
            "bmi_group":         sel("BMI group", bmi_group),
        }

    raw_df = pd.DataFrame(inputs, index=[0])
    expected = model.get_booster().feature_names
    df_new   = raw_df.reindex(columns=expected, fill_value=0).astype(np.float32)
    prob = model.predict_proba(df_new)[0,1]
    st.markdown(f"<h3>Estimated Probability of Breast Cancer: {prob:.1%}</h3>", unsafe_allow_html=True)
        
#Tab 3: Mind & Move 
with tab3:
    st.header("Glow and Grow")
    st.write(
        "Every healthy choice you make today is a step toward "
        "a brighter, happier you. You’ve got this!"
    )

    # Daily Rituals
    st.subheader("Daily Rituals")
    for tip in [
        "🧘🏼‍♀️ Practice 10 min mindfulness",
        "🥗 Eat ≥5 servings fruits/veggies",
        "🚶‍♀️ Take a 30 min walk",
        "💧 Drink 8 glasses of water",
        "😴 Get 7–8 h sleep",
    ]:
        st.markdown(f"- {tip}")

    # Live‐updating Tracker
    st.subheader("Tracker")
    c1, c2, c3 = st.columns(3)
    with c1:
        med = st.number_input("Meditation (min)", 0, 60, 0, key="med")
        st.progress(med / 10, text=f"{med} / 10 min")
        st.metric("Meditation", f"{med} min", f"{10-med} to goal")
    with c2:
        ex = st.number_input("Exercise (min)", 0, 180, 0, key="ex")
        st.progress(ex / 30, text=f"{ex} / 30 min")
        st.metric("Exercise", f"{ex} min", f"{30-ex} to goal")
    with c3:
        water = st.number_input("Water (glasses)", 0, 20, 0, key="water")
        st.progress(water / 8, text=f"{water} / 8 glasses")
        st.metric("Hydration", f"{water} glasses", f"{8-water} to goal")

    # Compute overall + mood
    overall = np.mean([med/10, ex/30, water/8])
    if overall == 0:
        mood = "😔"
    elif overall < 0.4:
        mood = "😐"
    elif overall < 0.6:
        mood = "🙂"
    elif overall < 0.8:
        mood = "😃"
    else:
        mood = "🤩"

    
    st.metric(
        label="Overall Wellness Progress",
        value=f"{overall*100:.0f}%",
        delta=mood
    )

    if st.button("Save Entry"):
        entry = {
            "Meditation (min)": med,
            "Exercise (min)":   ex,
            "Water (glasses)":  water,
            "Timestamp":        pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Mood":             mood,
        }
        st.success("✅ Your daily wellness entry has been recorded!")
        st.json(entry)
    st.title("Additional Resources")
    videos = {
    "Mindfulness Meditation for Cancer Support": "https://www.youtube.com/watch?v=1ZYbU82GVz4&t=31s",
    "Gentle Move for All":                     "https://www.youtube.com/watch?v=Ev6yE55kYGw&t=169s",
    "Healthy Eating During Cancer Treatment":   "https://www.youtube.com/watch?v=VaVC3PAWqLk&t=1353s"
}

    st.markdown("## 🎥 YouTube Videos")
    cols = st.columns(len(videos))
    for col, (title, url) in zip(cols, videos.items()):
        col.video(url)
        col.caption(f"**{title}**")
#support groups
with tab4:
    st.markdown("#You are not alone, check out🫂Local Support Groups")
    support = [
    {"Name": "American Cancer Society Middle Tennessee",            "Phone": "(615) 342-0840", "Website": "https://www.cancer.org/support-programs-and-services/patient-lodging/hope-lodge/nashville.html?utm_source=chatgpt.com"},
    {"Name": "Tennessee Breast Cancer Coalition",            "Phone": "(615) 377-8777", "Website": "https://www.tbcc.org/"},
    {"Name": "Susan G. Komen Nashville",            "Phone": "(615) 673-6633", "Website": "https://komen.org/nashville"},
    {"Name": "Vanderbilt Breast Cancer Support Group","Phone": "(615) 322-3900", "Website": "https://www.vanderbilthealth.com/service-line/breast-center"},
    {"Name": "Alive Hospice Cancer Support",        "Phone": "(615) 327-1085", "Website": "https://alivehospice.org"},
    {"Name": "YMCA of Middle Tennessee – LIVESTRONG® at the Y", "Website": "https://www.ymcamidtn.org/programs/health-and-fitness/support-groups/after-breast-cancer"},
    {"Name": "Gilda's club Middle Tennessee", "Phone": "(615) 329‑1124", "Website": "https://gildasclubmiddletn.org/"},
    {"Name": "TriStar Health", "Phone": "(800) 242-5665", "Website": "https://www.tristarhealth.com/specialties/oncology/cancer-support?"},
    {"Name": "Tennessee Breast Cancer Coalition", "Phone": "(615) 377- 8777", "Website": "https://www.tbcc.org/resources?"},   
    {"Name": "Breast cancer Recovery in Action", "Phone": "(615) 472-9478", "Website": "https://www.bragroups.org/about-us"}, 
    {"Name": "Nashville General Hospital Foundation", "Phone": "(615) 341-4431 ", "Website": "https://www.nashgenfoundation.org/funds?"}, 
    {"Name": "Ascension", "Phone": "(615) 284-2273", "Website": "https://healthcare.ascension.org/specialty-care/cancer/why-ascension/tnnas-nashville-tn-cancer-wellness-survivorship?"}  
]
    support_df = pd.DataFrame(support)
    support_df["Website"] = support_df["Website"].apply(lambda url: f"[Visit]({url})")
    st.table(support_df)
#Contact us tab
    st.sidebar.header("About Me")
    st.sidebar.markdown("""
Hi there—I’m Dollada Srisai. My journey from neuroscientist to data scientist has been guided not just by numbers, but by the resilience I witnessed living alongside cancer survivors. Those firsthand experiences taught me that behind every data point is a story of courage and hope. With a PhD in Neuroscience, veterinary medicine training, and hands-on expertise in machine learning and clinical research, I’m dedicated to turning complex insights into compassionate support—helping people truly understand, navigate, and thrive through their cancer journeys.
📬 ddsrisai@gmail.com | 🔗 https://www.linkedin.com/in/dollada-srisai/ | 🐙 https://github.com/fvetdds 

""")

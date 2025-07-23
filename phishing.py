# # import streamlit as st
# # import pandas as pd
# # import numpy as np
# # from sklearn.ensemble import RandomForestClassifier
# # from sklearn.linear_model import LinearRegression
# # from sklearn.cluster import KMeans
# # from sklearn.preprocessing import StandardScaler
# # from sklearn.tree import DecisionTreeClassifier, plot_tree
# # import matplotlib.pyplot as plt
# # from urllib.parse import urlparse
# # import re
# # import whois
# # from datetime import datetime
# # import os

# # # --------- Helper Functions ---------

# # def has_ip(url):
# #     ip_pattern = r'http[s]?://(\d{1,3}\.){3}\d{1,3}'
# #     return 1 if re.match(ip_pattern, url) else 0

# # def count_subdomains(domain):
# #     parts = domain.split('.')
# #     return len(parts) - 2 if len(parts) > 2 else 0

# # def get_domain_age(domain):
# #     try:
# #         w = whois.whois(domain)
# #         creation_date = w.creation_date
# #         if isinstance(creation_date, list):
# #             creation_date = creation_date[0]
# #         if creation_date is None:
# #             return -1
# #         return max((datetime.now() - creation_date).days, 0)
# #     except:
# #         return -1

# # def has_https(url):
# #     return 1 if urlparse(url).scheme == 'https' else 0

# # def has_popup_window(url):
# #     return 0  # Assume 0 - can't detect from URL

# # @st.cache_resource
# # def load_models():
# #     if not os.path.exists("cleaned_file.csv"):
# #         st.error("Dataset 'cleaned_file.csv' not found.")
# #         st.stop()
# #     df = pd.read_csv("cleaned_file.csv")

# #     features = ['url_length', 'subdomains_count', 'https', 'popup_window', 'domain_age', 'has_ip_address']
# #     X = df[features]
# #     y = df['label']

# #     scaler = StandardScaler()
# #     X_scaled = scaler.fit_transform(X)

# #     rf = RandomForestClassifier(n_estimators=100, random_state=42)
# #     rf.fit(X_scaled, y)

# #     reg = LinearRegression()
# #     reg.fit(X[['url_length', 'subdomains_count']], df['domain_age'])

# #     kmeans = KMeans(n_clusters=2, random_state=42)
# #     kmeans.fit(X_scaled)

# #     dt = DecisionTreeClassifier(max_depth=4, random_state=42)
# #     dt.fit(X_scaled, y)

# #     return rf, reg, kmeans, scaler, dt

# # rf_model, reg_model, kmeans_model, scaler, dt_model = load_models()

# # # -------- Tooltip HTML --------

# # tooltip_html = """
# # <style>
# # .tooltip {
# #   position: relative;
# #   display: inline-block;
# #   cursor: pointer;
# #   color: red;
# #   text-decoration: underline;
# #   font-weight: bold;
# # }
# # .tooltip .tooltiptext {
# #   visibility: hidden;
# #   width: 280px;
# #   background-color: #aa0000;
# #   color: #fff;
# #   text-align: center;
# #   border-radius: 8px;
# #   padding: 10px;
# #   position: absolute;
# #   z-index: 1;
# #   bottom: 125%;
# #   left: 50%;
# #   margin-left: -140px;
# #   opacity: 0;
# #   transition: opacity 0.3s;
# #   box-shadow: 0 4px 8px rgba(170,0,0,0.7);
# # }
# # .tooltip .tooltiptext::after {
# #   content: "";
# #   position: absolute;
# #   top: 100%;
# #   left: 50%;
# #   margin-left: -5px;
# #   border-width: 5px;
# #   border-style: solid;
# #   border-color: #aa0000 transparent transparent transparent;
# # }
# # .tooltip:hover .tooltiptext {
# #   visibility: visible;
# #   opacity: 1;
# # }
# # .tooltip .tooltiptext img {
# #   max-width: 100%;
# #   height: auto;
# #   border-radius: 6px;
# #   margin-top: 8px;
# # }
# # </style>

# # <div class="tooltip">🚫 Phishing or Suspicious Website!
# #   <div class="tooltiptext">
# #     <p><strong>This site looks suspicious or dangerous.</strong></p>
# #     <p>It may use an IP address, be too new, lack HTTPS, or match known phishing patterns.</p>
# #     <img src="https://cdn-icons-png.flaticon.com/512/564/564619.png" alt="Warning" />
# #   </div>
# # </div>
# # """

# # # --------- Sidebar Navigation ---------

# # st.sidebar.title("🔎 Phishing Detection Suite")
# # menu = ["Home", "Predict Phishing", "Feature Importance", "Predict Domain Age", "Decision Rules", "About"]
# # choice = st.sidebar.radio("Navigation", menu)

# # # --------- Pages ---------

# # def page_home():
# #     st.markdown("<h1 style='text-align:center;'>🔐 Welcome to the Phishing Detection Suite</h1>", unsafe_allow_html=True)
# #     st.markdown("""
# #     ---
# #     ✅ Detect phishing websites in real-time  
# #     📊 View important URL features  
# #     ⏳ Predict domain age  
# #     🌳 Visualize how decisions are made
    
# #     👉 Try the **Predict Phishing** tab!
# #     """)
# #     st.image("https://images.unsplash.com/photo-1504805572947-34fad45aed93?auto=format&fit=crop&w=800&q=80", use_column_width=True)

# # def page_predict_phishing():
# #     st.header("🔍 Predict if Website is Phishing or Legitimate")

# #     url_input = st.text_input("Enter full website URL (e.g., https://example.com):")

# #     if st.button("Predict"):
# #         if not url_input:
# #             st.error("Please enter a URL.")
# #             return
# #         if not url_input.startswith(("http://", "https://")):
# #             url_input = "http://" + url_input

# #         domain = urlparse(url_input).netloc.lower()

# #         # Extract features
# #         url_length = len(url_input)
# #         subdomains_count = count_subdomains(domain)
# #         has_ip_address = has_ip(url_input)
# #         https_flag = has_https(url_input)
# #         popup_window = has_popup_window(url_input)
# #         domain_age = get_domain_age(domain)

# #         # Heuristic flag: common suspicious domains
# #         known_suspicious = ['bit.ly', 'tinyurl.com', 't.co', 'ow.ly']
# #         is_known_suspicious = any(x in domain for x in known_suspicious)

# #         # Predict with ML model
# #         features = np.array([[url_length, subdomains_count, https_flag, popup_window,
# #                               domain_age if domain_age >= 0 else 0, has_ip_address]])
# #         features_scaled = scaler.transform(features)
# #         ml_pred = rf_model.predict(features_scaled)[0]
# #         cluster = kmeans_model.predict(features_scaled)[0]
# #         domain_age_pred = reg_model.predict([[url_length, subdomains_count]])[0]

# #         # Final decision (ML or heuristic)
# #         is_suspicious = (
# #             ml_pred == 1 or
# #             has_ip_address == 1 or
# #             https_flag == 0 or
# #             is_known_suspicious or
# #             domain_age == -1
# #         )
# #         label = "🚫 Phishing or Suspicious!" if is_suspicious else "✅ Legitimate"

# #         # Display
# #         st.subheader("🔎 Result")
# #         st.write(f"**URL:** {url_input}")
# #         st.write(f"**Prediction:** {label}")
# #         st.write(f"**Estimated Domain Age:** {int(domain_age_pred)} days")
# #         st.write(f"**Behavioral Cluster:** Group {cluster}")

# #         if is_suspicious:
# #             st.markdown(tooltip_html, unsafe_allow_html=True)
# #         else:
# #             st.success("✅ This site looks safe!")

# # def page_feature_importance():
# #     st.header("📊 Feature Importance (Random Forest)")

# #     importances = rf_model.feature_importances_
# #     feature_names = ['URL Length', 'Subdomains Count', 'HTTPS', 'Popup Window', 'Domain Age', 'Has IP Address']
# #     imp_df = pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values(by="Importance", ascending=False)

# #     st.bar_chart(imp_df.set_index("Feature"))

# # def page_predict_domain_age():
# #     st.header("⏳ Predict Domain Age")

# #     url_input = st.text_input("Enter URL (e.g., https://example.com):")

# #     if st.button("Predict Domain Age"):
# #         if not url_input:
# #             st.error("Please enter a URL.")
# #             return
# #         if not url_input.startswith(("http://", "https://")):
# #             url_input = "http://" + url_input
# #         domain = urlparse(url_input).netloc

# #         url_length = len(url_input)
# #         subdomains_count = count_subdomains(domain)

# #         pred_age = reg_model.predict([[url_length, subdomains_count]])[0]
# #         st.write(f"Estimated Domain Age for {url_input}: **{int(pred_age)} days**")

# # def page_decision_rules():
# #     st.header("🌳 Decision Tree Rules")

# #     fig, ax = plt.subplots(figsize=(15, 8))
# #     plot_tree(dt_model, feature_names=['url_length', 'subdomains_count', 'https', 'popup_window', 'domain_age', 'has_ip_address'],
# #               class_names=['Legitimate', 'Phishing'], filled=True, ax=ax)
# #     st.pyplot(fig)

# # def page_about():
# #     st.header("ℹ️ About This App")
# #     st.markdown("""
# #     This app was built for academic and demo purposes. It uses:

# #     - 🧠 Machine Learning (Random Forest, Decision Tree)
# #     - 🌐 URL feature extraction
# #     - 📉 Domain age estimation
# #     - 🎯 Heuristic rules for real-world safety

# #     Developed with ❤️ by Anupama Chaudhary
# #     """)

# # # --------- Main Navigation ---------

# # if choice == "Home":
# #     page_home()
# # elif choice == "Predict Phishing":
# #     page_predict_phishing()
# # elif choice == "Feature Importance":
# #     page_feature_importance()
# # elif choice == "Predict Domain Age":
# #     page_predict_domain_age()
# # elif choice == "Decision Rules":
# #     page_decision_rules()
# # elif choice == "About":
# #     page_about()







# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LinearRegression
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import StandardScaler
# from sklearn.tree import DecisionTreeClassifier, plot_tree
# import matplotlib.pyplot as plt
# from urllib.parse import urlparse
# import re
# import whois
# from datetime import datetime
# import os

# # --------- Helper Functions ---------

# def has_ip(url):
#     ip_pattern = r'http[s]?://(\d{1,3}\.){3}\d{1,3}'
#     return 1 if re.match(ip_pattern, url) else 0

# def count_subdomains(domain):
#     parts = domain.split('.')
#     return max(len(parts) - 2, 0)

# def get_domain_age(domain):
#     try:
#         w = whois.whois(domain)
#         creation_date = w.creation_date
#         if isinstance(creation_date, list):
#             creation_date = creation_date[0]
#         if creation_date is None:
#             return 0
#         return max((datetime.now() - creation_date).days, 0)
#     except Exception:
#         return 0

# def has_https(url):
#     return 1 if urlparse(url).scheme == 'https' else 0

# def has_popup_window(url):
#     return 0

# @st.cache_resource
# def load_models():
#     if not os.path.exists("cleaned_file.csv"):
#         st.error("Dataset 'cleaned_file.csv' not found. Please add it.")
#         st.stop()
#     df = pd.read_csv("cleaned_file.csv")

#     features = ['url_length', 'subdomains_count', 'https', 'popup_window', 'domain_age', 'has_ip_address']
#     X = df[features]
#     y = df['label']

#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)

#     rf = RandomForestClassifier(n_estimators=100, random_state=42)
#     rf.fit(X_scaled, y)

#     reg = LinearRegression()
#     reg.fit(X[['url_length', 'subdomains_count']], df['domain_age'])

#     kmeans = KMeans(n_clusters=2, random_state=42)
#     kmeans.fit(X_scaled)

#     dt = DecisionTreeClassifier(max_depth=4, random_state=42)
#     dt.fit(X_scaled, y)

#     return rf, reg, kmeans, scaler, dt

# rf_model, reg_model, kmeans_model, scaler, dt_model = load_models()

# st.sidebar.title("🔎 Phishing Detection Suite")
# menu = ["Home", "Predict Phishing", "Feature Importance", "Predict Domain Age", "Decision Rules", "About"]
# choice = st.sidebar.radio("Navigation", menu)

# def page_home():
#     st.markdown("<h1 style='text-align:center;'>🔐 Welcome to the Phishing Detection Suite</h1>", unsafe_allow_html=True)
#     st.markdown("""
#     ---
#     #### This tool helps you:
#     - Detect phishing websites in real-time.
#     - Understand which URL features matter most.
#     - Predict domain age to assess trustworthiness.
#     - Visualize decision rules used in phishing classification.
#     """)
#     st.image("https://images.unsplash.com/photo-1504805572947-34fad45aed93?auto=format&fit=crop&w=800&q=80", use_column_width=True)

# def page_predict_phishing():
#     st.header("🔍 Predict if Website is Phishing or Legitimate")
#     url_input = st.text_input("Enter full website URL (e.g., https://example.com):")

#     if st.button("Predict"):
#         if not url_input:
#             st.error("Please enter a URL to analyze.")
#             return
#         if not url_input.startswith(("http://", "https://")):
#             url_input = "http://" + url_input

#         domain = urlparse(url_input).netloc

#         url_length = len(url_input)
#         has_ip_address = has_ip(url_input)
#         https_flag = has_https(url_input)
#         popup_window = has_popup_window(url_input)
#         subdomains_count = count_subdomains(domain)
#         domain_age = get_domain_age(domain)

#         features = np.array([[url_length, subdomains_count, https_flag, popup_window, domain_age, has_ip_address]])
#         features_scaled = scaler.transform(features)

#         pred = rf_model.predict(features_scaled)[0]
#         label = "Phishing 🚫" if pred == 1 else "Legitimate ✅"
#         cluster = kmeans_model.predict(features_scaled)[0]
#         domain_age_pred = reg_model.predict([[url_length, subdomains_count]])[0]

#         st.markdown("### 🔎 Extracted Features")
#         st.write(f"- URL Length: `{url_length}`")
#         st.write(f"- Subdomains Count: `{subdomains_count}`")
#         st.write(f"- HTTPS Enabled: `{https_flag}`")
#         st.write(f"- IP Address in URL: `{has_ip_address}`")
#         st.write(f"- Domain Age: `{domain_age}` days")

#         st.markdown("### 📋 Prediction Result")
#         st.write(f"**URL:** {url_input}")
#         st.write(f"**Prediction:** {label}")
#         st.write(f"**Estimated Domain Age (regression):** `{int(domain_age_pred)} days`")
#         st.write(f"**Behavioral Cluster:** Group `{cluster}`")

#         if pred == 1:
#             st.error("🚫 This website appears to be phishing. Avoid entering sensitive information.")
#         elif has_ip_address == 1:
#             st.warning("⚠️ This website is legitimate, but it uses an IP address instead of a domain.")
#         elif https_flag == 0:
#             st.warning("⚠️ This website is legitimate, but it does not use HTTPS.")
#         elif domain_age == 0:
#             st.warning("⚠️ This website is legitimate, but WHOIS lookup failed or domain is very new.")
#         else:
#             st.success("✅ This website appears safe based on all features.")

# def page_feature_importance():
#     st.header("📊 Feature Importance (Random Forest)")
#     importances = rf_model.feature_importances_
#     feature_names = ['URL Length', 'Subdomains Count', 'HTTPS', 'Popup Window', 'Domain Age', 'Has IP Address']
#     imp_df = pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values(by="Importance", ascending=False)
#     st.bar_chart(imp_df.set_index("Feature"))

# def page_predict_domain_age():
#     st.header("⏳ Predict Domain Age")
#     url_input = st.text_input("Enter URL (e.g., https://example.com):")

#     if st.button("Predict Domain Age"):
#         if not url_input:
#             st.error("Please enter a URL.")
#             return
#         if not url_input.startswith(("http://", "https://")):
#             url_input = "http://" + url_input
#         domain = urlparse(url_input).netloc

#         url_length = len(url_input)
#         subdomains_count = count_subdomains(domain)

#         pred_age = reg_model.predict([[url_length, subdomains_count]])[0]
#         st.write(f"Estimated Domain Age for {url_input}: **{int(pred_age)} days**")

# def page_decision_rules():
#     st.header("🌳 Decision Tree Rules")
#     fig, ax = plt.subplots(figsize=(15, 8))
#     plot_tree(dt_model, feature_names=['url_length', 'subdomains_count', 'https', 'popup_window', 'domain_age', 'has_ip_address'],
#               class_names=['Legitimate', 'Phishing'], filled=True, ax=ax)
#     st.pyplot(fig)

# def page_about():
#     st.header("ℹ️ About This App")
#     st.markdown("""
#     This Phishing Detection Suite is a Streamlit app using machine learning to:

#     - Predict phishing websites based on URL features.
#     - Visualize which features are most important.
#     - Predict domain age to assess trustworthiness.
#     - Show decision rules used in classification.

#     Built with ❤️ for educational use.
#     """)

# if choice == "Home":
#     page_home()
# elif choice == "Predict Phishing":
#     page_predict_phishing()
# elif choice == "Feature Importance":
#     page_feature_importance()
# elif choice == "Predict Domain Age":
#     page_predict_domain_age()
# elif choice == "Decision Rules":
#     page_decision_rules()
# elif choice == "About":
#     page_about()







import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from urllib.parse import urlparse
import re
import whois
from datetime import datetime
import os

# --------- Helper Functions ---------

def has_ip(url):
    ip_pattern = r'http[s]?://(\d{1,3}\.){3}\d{1,3}'
    return 1 if re.match(ip_pattern, url) else 0

def count_subdomains(domain):
    parts = domain.split('.')
    return max(len(parts) - 2, 0)

def get_domain_age(domain):
    try:
        w = whois.whois(domain)
        creation_date = w.creation_date
        if isinstance(creation_date, list):
            creation_date = creation_date[0]
        if creation_date is None:
            return 0
        return max((datetime.now() - creation_date).days, 0)
    except Exception:
        return 0

def has_https(url):
    return 1 if urlparse(url).scheme == 'https' else 0

def has_popup_window(url):
    # Placeholder: no popup detection implemented
    return 0

@st.cache_resource
def load_models():
    if not os.path.exists("cleaned_file.csv"):
        st.error("Dataset 'cleaned_file.csv' not found. Please add it to the app folder.")
        st.stop()
    df = pd.read_csv("cleaned_file.csv")

    features = ['url_length', 'subdomains_count', 'https', 'popup_window', 'domain_age', 'has_ip_address']
    X = df[features]
    y = df['label']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_scaled, y)

    reg = LinearRegression()
    reg.fit(X[['url_length', 'subdomains_count']], df['domain_age'])

    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(X_scaled)

    dt = DecisionTreeClassifier(max_depth=4, random_state=42)
    dt.fit(X_scaled, y)

    return rf, reg, kmeans, scaler, dt

rf_model, reg_model, kmeans_model, scaler, dt_model = load_models()

st.sidebar.title("🔎 Phishing Detection Suite")
menu = ["Home", "Predict Phishing", "Feature Importance", "Predict Domain Age", "Decision Rules", "About"]
choice = st.sidebar.radio("Navigation", menu)

def page_home():
    st.markdown("<h1 style='text-align:center;'>🔐 Welcome to the Phishing Detection Suite</h1>", unsafe_allow_html=True)
    st.markdown("""
    ---
    #### This tool helps you:
    - Detect phishing websites in real-time.
    - Understand which URL features matter most.
    - Predict domain age to assess trustworthiness.
    - Visualize decision rules used in phishing classification.
    """)
    st.image("https://images.unsplash.com/photo-1504805572947-34fad45aed93?auto=format&fit=crop&w=800&q=80", use_column_width=True)

def page_predict_phishing():
    st.header("🔍 Predict if Website is Phishing or Legitimate")
    url_input = st.text_input("Enter full website URL (e.g., https://example.com):")

    if st.button("Predict"):
        if not url_input:
            st.error("Please enter a URL to analyze.")
            return
        if not url_input.startswith(("http://", "https://")):
            url_input = "http://" + url_input

        domain = urlparse(url_input).netloc

        url_length = len(url_input)
        has_ip_address = has_ip(url_input)
        https_flag = has_https(url_input)
        popup_window = has_popup_window(url_input)
        subdomains_count = count_subdomains(domain)
        domain_age = get_domain_age(domain)

        features = np.array([[url_length, subdomains_count, https_flag, popup_window, domain_age, has_ip_address]])
        features_scaled = scaler.transform(features)

        pred_proba = rf_model.predict_proba(features_scaled)[0][1]

        # Thresholds and combined heuristic flags
        DOMAIN_AGE_SUSPICIOUS_THRESHOLD = 30  # days
        PHISHING_PROB_THRESHOLD = 0.7

        is_domain_age_suspicious = (domain_age == 0 or domain_age < DOMAIN_AGE_SUSPICIOUS_THRESHOLD)
        is_no_https = (https_flag == 0)
        is_ip_in_url = (has_ip_address == 1)

        # Smarter combined logic for labeling
        if pred_proba >= PHISHING_PROB_THRESHOLD:
            label = "Phishing 🚫"
        elif is_no_https and (is_domain_age_suspicious or is_ip_in_url):
            label = "Suspicious ⚠️ (No HTTPS + Other risk factors)"
        elif is_ip_in_url and (is_domain_age_suspicious or is_no_https):
            label = "Suspicious ⚠️ (IP in URL + Other risk factors)"
        elif is_domain_age_suspicious and (is_no_https or is_ip_in_url):
            label = "Suspicious ⚠️ (New domain + Other risk factors)"
        else:
            label = "Legitimate ✅"

        cluster = kmeans_model.predict(features_scaled)[0]
        domain_age_pred = reg_model.predict([[url_length, subdomains_count]])[0]

        st.markdown("### 🔎 Extracted Features")
        st.write(f"- URL Length: `{url_length}`")
        st.write(f"- Subdomains Count: `{subdomains_count}`")
        st.write(f"- HTTPS Enabled: `{https_flag}`")
        st.write(f"- IP Address in URL: `{has_ip_address}`")
        st.write(f"- Domain Age: `{domain_age}` days")

        st.markdown("### 📋 Prediction Result")
        st.write(f"**URL:** {url_input}")
        st.write(f"**Prediction:** {label}")
        st.write(f"**Phishing Probability:** {pred_proba:.2%}")
        st.write(f"**Estimated Domain Age (regression):** `{int(domain_age_pred)} days`")
        st.write(f"**Behavioral Cluster:** Group `{cluster}`")

        if label == "Phishing 🚫":
            st.error("🚫 This website appears to be phishing. Avoid entering sensitive information.")
        elif "Suspicious" in label:
            st.warning(f"⚠️ This website is suspicious. Reason: {label.split('(')[1].strip(')')}")
        else:
            st.success("✅ This website appears safe based on all features.")

def page_feature_importance():
    st.header("📊 Feature Importance (Random Forest)")
    importances = rf_model.feature_importances_
    feature_names = ['URL Length', 'Subdomains Count', 'HTTPS', 'Popup Window', 'Domain Age', 'Has IP Address']
    imp_df = pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values(by="Importance", ascending=False)
    st.bar_chart(imp_df.set_index("Feature"))

def page_predict_domain_age():
    st.header("⏳ Predict Domain Age")
    url_input = st.text_input("Enter URL (e.g., https://example.com):")

    if st.button("Predict Domain Age"):
        if not url_input:
            st.error("Please enter a URL.")
            return
        if not url_input.startswith(("http://", "https://")):
            url_input = "http://" + url_input
        domain = urlparse(url_input).netloc

        url_length = len(url_input)
        subdomains_count = count_subdomains(domain)

        pred_age = reg_model.predict([[url_length, subdomains_count]])[0]
        st.write(f"Estimated Domain Age for {url_input}: **{int(pred_age)} days**")

def page_decision_rules():
    st.header("🌳 Decision Tree Rules")
    fig, ax = plt.subplots(figsize=(15, 8))
    plot_tree(dt_model, feature_names=['url_length', 'subdomains_count', 'https', 'popup_window', 'domain_age', 'has_ip_address'],
              class_names=['Legitimate', 'Phishing'], filled=True, ax=ax)
    st.pyplot(fig)

def page_about():
    st.header("ℹ️ About This App")
    st.markdown("""
    This Phishing Detection Suite is a Streamlit app using machine learning to:

    - Predict phishing websites based on URL features.
    - Visualize which features are most important.
    - Predict domain age to assess trustworthiness.
    - Show decision rules used in classification.

    Built with ❤️ for educational use.
    """)

if choice == "Home":
    page_home()
elif choice == "Predict Phishing":
    page_predict_phishing()
elif choice == "Feature Importance":
    page_feature_importance()
elif choice == "Predict Domain Age":
    page_predict_domain_age()
elif choice == "Decision Rules":
    page_decision_rules()
elif choice == "About":
    page_about()

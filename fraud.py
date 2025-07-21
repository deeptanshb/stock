import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import joblib
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score
from tensorflow.keras.models import load_model
from email.mime.text import MIMEText
import smtplib

st.title("🚀 Financial Fraud Detection Dashboard")

@st.cache_data
def load_data():
    df = pd.read_csv("/home/deeptanshu/.cache/kagglehub/datasets/ealaxi/paysim1/versions/2/PS_20174392719_1491204439457_log.csv")
    return df.head(1000)

df = load_data()
st.subheader("📈 Sample Transaction Data")
st.dataframe(df.head(10))

@st.cache_resource
def load_models():
    models = {
        "Logistic Regression": joblib.load("log_model.pkl"),
        "Random Forest": joblib.load("rf_model.pkl"),
        "Isolation Forest": joblib.load("iso_model.pkl"),
        "Autoencoder": load_model("autoencoder_model.keras", compile=False)
    }
    return models

models = load_models()
scaler = joblib.load("scaler.pkl")


df['amount_normalized'] = (df['amount'] - df['amount'].mean()) / df['amount'].std()


if 'type' in df.columns:
    df = pd.get_dummies(df, columns=['type'], drop_first=True)


df['errorBalanceOrig'] = df['newbalanceOrig'] + df['amount'] - df['oldbalanceOrg']
df['errorBalanceDest'] = df['oldbalanceDest'] + df['amount'] - df['newbalanceDest']

df = df.drop(['nameOrig', 'nameDest', 'type', 'timestamp'], axis=1, errors='ignore')
X = df.drop(['isFraud', 'isFlaggedFraud'], axis=1, errors='ignore')
y = df['isFraud']
X_scaled = scaler.transform(X)


st.subheader("🧮 Model Evaluation")
model_scores = {}
model_preds = {}


pred_log = models['Logistic Regression'].predict(X_scaled)
auc_log = roc_auc_score(y, pred_log)
model_scores['Logistic Regression'] = auc_log
model_preds['Logistic Regression'] = pred_log


pred_rf = models['Random Forest'].predict(X_scaled)
auc_rf = roc_auc_score(y, pred_rf)
model_scores['Random Forest'] = auc_rf
model_preds['Random Forest'] = pred_rf


pred_iso = models['Isolation Forest'].predict(X_scaled)
pred_iso = np.where(pred_iso == -1, 1, 0)
auc_iso = roc_auc_score(y, pred_iso)
model_scores['Isolation Forest'] = auc_iso
model_preds['Isolation Forest'] = pred_iso


recon = models['Autoencoder'].predict(X_scaled)
recon_error = np.mean(np.square(X_scaled - recon), axis=1)
threshold = np.percentile(recon_error, 95)
pred_auto = np.where(recon_error > threshold, 1, 0)
auc_auto = roc_auc_score(y, pred_auto)
model_scores['Autoencoder'] = auc_auto
model_preds['Autoencoder'] = pred_auto


st.subheader("🏛 Model AUC Comparison")
score_df = pd.DataFrame(model_scores.items(), columns=["Model", "ROC-AUC"])
fig = px.bar(score_df, x="Model", y="ROC-AUC", color="ROC-AUC", title="Model Comparison")
st.plotly_chart(fig)


def plot_conf_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    z = cm.tolist()
    x = ['Not Fraud', 'Fraud']
    y = ['Not Fraud', 'Fraud']
    fig = ff.create_annotated_heatmap(
        z, x=x, y=y, colorscale='Blues', showscale=True,
        annotation_text=[[str(cell) for cell in row] for row in z],
        hoverinfo='z'
    )
    fig.update_layout(title_text=f'Confusion Matrix - {model_name}')
    st.plotly_chart(fig)

for name, pred in model_preds.items():
    plot_conf_matrix(y, pred, name)

st.subheader("📊 Precision, Recall, F1-Score")
metrics_df = pd.DataFrame([
    {
        'Model': name,
        'Precision': precision_score(y, pred),
        'Recall': recall_score(y, pred),
        'F1-Score': f1_score(y, pred)
    }
    for name, pred in model_preds.items()
])

fig_metrics = px.bar(
    metrics_df.melt(id_vars='Model'),
    x='Model', y='value', color='variable',
    barmode='group', title='Precision, Recall & F1-Score per Model'
)
st.plotly_chart(fig_metrics)


best_model_name = score_df.loc[score_df['ROC-AUC'].idxmax(), 'Model']
st.success(f"Best Model: {best_model_name}")


st.subheader("🧾 Enter Transaction Details")


amount = st.number_input("Amount", min_value=0.0)
oldbalanceOrg = st.number_input("Old Balance (Sender)", min_value=0.0)
newbalanceOrig = st.number_input("New Balance (Sender)", min_value=0.0)
oldbalanceDest = st.number_input("Old Balance (Receiver)", min_value=0.0)
newbalanceDest = st.number_input("New Balance (Receiver)", min_value=0.0)


type_transfer = st.radio("Is this a TRANSFER transaction?", [0, 1], index=0)


if st.button("🚀 Run Fraud Detection"):
    try:
        
        transaction = pd.DataFrame([{
            'step':1,
            'amount': amount,
            'type_TRANSFER': type_transfer,
            'oldbalanceOrg': oldbalanceOrg,
            'newbalanceOrig': newbalanceOrig,
            'oldbalanceDest': oldbalanceDest,
            'newbalanceDest': newbalanceDest
        }])

        
        transaction['amount_normalized'] = (transaction['amount'] - df['amount'].mean()) / df['amount'].std()
        transaction['errorBalanceOrig'] = transaction['newbalanceOrig'] + transaction['amount'] - transaction['oldbalanceOrg']
        transaction['errorBalanceDest'] = transaction['oldbalanceDest'] + transaction['amount'] - transaction['newbalanceDest']

       
        for col in df.columns:
            if col.startswith("type_") and col not in transaction.columns:
                transaction[col] = 0

        
        transaction = transaction[df.drop(['isFraud', 'isFlaggedFraud'], axis=1, errors='ignore').columns]

    
        sample_scaled = scaler.transform(transaction)

        if best_model_name == "Autoencoder":
            recon = models['Autoencoder'].predict(sample_scaled)
            err = np.mean(np.square(sample_scaled - recon))
            pred = 1 if err > threshold else 0
        elif best_model_name == "Isolation Forest":
            pred = models['Isolation Forest'].predict(sample_scaled)
            pred = 1 if pred[0] == -1 else 0
            err = None
        else:
            pred = models[best_model_name].predict(sample_scaled)[0]
            err = None

    
        if pred == 1:
            st.error("🚨 Fraudulent Transaction Detected!")
        else:
            st.success("✅ Transaction is Normal.")


        if best_model_name == "Autoencoder" and err is not None:
            st.info(f"Reconstruction Error: {err:.4f} | Threshold: {threshold:.4f}")
            fig = px.bar(x=['Reconstruction Error'], y=[err], labels={'x': 'Metric', 'y': 'Error'}, title='Autoencoder Anomaly Score')
            st.plotly_chart(fig)

    except Exception as e:
        st.error(f"❌ Error in prediction: {e}")



def send_alert(transaction):
    sender_email = "deepb2601@gmail.com"
    receiver_email = "deepu26jan2004@gmail.com"
    password = "efww jpkb rwry valb"
    subject = "🚨 Fraud Alert Detected"
    body = f"""
    Fraudulent transaction detected!

    Details:
    {transaction.to_string(index=True)}
    """
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = receiver_email
    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, msg.as_string())
        server.quit()
        st.success("✅ Email alert sent successfully!")
    except Exception as e:
        st.error(f"❌ Failed to send alert: {e}")


    if pred == 1:
        send_alert(transaction.iloc[0])

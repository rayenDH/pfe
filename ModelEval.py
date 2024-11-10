import streamlit as st
from PIL import Image 
import pandas as pd
import os 
import pickle as pk 
import matplotlib
import scikitplot as skplt
import time
#from ydata_profiling import ProfileReport
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn import metrics
st.set_option('deprecation.showPyplotGlobalUse', False)
model=pk.load(open('modelxg.pkl','rb'))
image=Image.open("oore.png")
st.markdown(
   f"""
   <style>
   .stapp{{
   background-image: url('oore.png');
   }}
   </style>
   """,
   unsafe_allow_html=True)

col2, col3= st.columns(2)
with st.sidebar:
    st.image(image,width=160)
    st.title(' Model Evaluation')

    file=st.file_uploader("upload the CSV file")
    if file:
        df= pd.read_csv(file,index_col=None)
        df.to_csv("sourcedata.csv",index=None)
        df.dropna(inplace=True)
        df= df.drop(columns="subscriber_id.1")
        df= df.drop(columns="subscriber_id.2")
        df= df.drop(columns="decision_utilisateur")
        df= df.drop(columns="international_mou_out_2021_12")
        df= df.drop(columns="nb_sms_out_international_2021_12")
        df= df.drop(columns="international_mou_out_2021_11")
        df= df.drop(columns="nb_sms_out_international_2021_11")
        df= df.drop(columns="international_mou_out_2021_10")
        df= df.drop(columns="nb_sms_out_international_2021_10")
        df= df.drop(columns="international_mou_out_2021_09")
        df= df.drop(columns="nb_sms_out_international_2021_09")
        df= df.drop(columns="international_mou_out_2021_08")
        df= df.drop(columns="nb_sms_out_international_2021_08")
        df= df.drop(columns="international_mou_out_2021_07")
        df= df.drop(columns="nb_sms_out_international_2021_07")
        df= df.drop(columns="nb_sms_out_international_2021_06")
        df['revenu_upsell'].replace(to_replace='achat inf seuil', value=1, inplace=True)
        df['revenu_upsell'].replace(to_replace='Non achat',  value=0, inplace=True)
        df['revenu_upsell'].replace(to_replace='Achat',  value=2, inplace=True)
        y = df['interesse'].values
        X = df.drop(columns = ['interesse'])
        
        #booster = model.get_booster()
        #booster.set_feature_types({'feature_name': 'feature_type'})
        # Create Train & Test Data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=0)
        from sklearn.preprocessing import MinMaxScaler
        features = X.columns.values
        scaler = MinMaxScaler(feature_range = (0,1))
        scaler.fit(X)
        X = pd.DataFrame(scaler.transform(X))
        X.columns = features
       
    with st.expander("Click to Choose Metrics"):
        options=st.multiselect('choose the metrics',['Accuracy','Precision','Recall','F1'])
    with st.expander("Click to Choose Plots"):
        opt=st.multiselect('what metric to plot',['ROC Curve plot','Confusion Matrix','Precision-Recall curve'])
with col2:
    
    
    if 'Accuracy' in  options :
        from sklearn.metrics import classification_report
        ypred = model.predict(X_val)
        report = classification_report(y_val, ypred, output_dict=True)
        macro_precision =  report['macro avg']['precision'] 
        macro_recall = report['macro avg']['recall']    
        macro_f1 = report['macro avg']['f1-score']
        st.write('<p style="font-size:26px;color:#FF3352;">Accuracy:</p>', unsafe_allow_html=True)
        st.write(report['accuracy'])
    if 'Precision' in options:
        from sklearn.metrics import classification_report
        ypred = model.predict(X_val)
        report = classification_report(y_val, ypred, output_dict=True)
        st.write('<p style="font-size:26px;color:#FF3352;">Precision:</p>', unsafe_allow_html=True)
        st.write(report['macro avg']['precision'])
        
    if 'Recall' in options:
        from sklearn.metrics import classification_report
        ypred = model.predict(X_val)
        report = classification_report(y_val, ypred, output_dict=True)
        st.write('<p style="font-size:26px;color:#FF3352;">Recall:</p>', unsafe_allow_html=True)
        st.write(report['macro avg']['recall'])
        
    if 'F1' in options:
        from sklearn.metrics import classification_report
        ypred = model.predict(X_val)
        report = classification_report(y_val, ypred, output_dict=True)
        st.write('<p style="font-size:26px;color:#FF3352;">F1 Score:</p>', unsafe_allow_html=True)
        st.write(report['macro avg']['f1-score'])

with col3:
    if "ROC Curve plot" in opt:
        with st.expander("ROC Curve"):
            y_probas = model.predict_proba(X_test)[::,1]
            ns_probs = [0 for _ in range(len(y_test))]
            lr_probs = model.predict_proba(X_test)
            # keep probabilities for the positive outcome only
            lr_probs = lr_probs[:, 1]
            # calculate scoresplt
            ns_auc = roc_auc_score(y_test, ns_probs)
            lr_auc = roc_auc_score(y_test, lr_probs)
            # summarize scores
            
            # calculate roc curves
            ns_fpr, ns_tpr, _ = metrics.roc_curve(y_test, ns_probs)
            lr_fpr, lr_tpr, _ = metrics.roc_curve(y_test, lr_probs)
            # plot the roc curve for the model
            plt.plot(ns_fpr, ns_tpr, linestyle='--', label='random guess')
            plt.plot(lr_fpr, lr_tpr, marker='.', label='XGBOOST')
            # axis labels
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            # show the legend
            # show the plot
            with st.spinner("Loading..."):
                time.sleep(5)
            
            st.subheader("ROC Curve")
            st.pyplot() 
            st.write('Random guess: ROC AUC=%.3f' % (ns_auc))
            st.write('XGBOOST: ROC AUC=%.3f' % (lr_auc))    
    if "Confusion Matrix" in opt:
        with st.expander("Confusion Matrix"):
            from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay  
            predictions=model.predict(X_test)     
            cm = confusion_matrix(y_test, predictions, labels=model.classes_)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=model.classes_)
            disp.plot()
            with st.spinner("Loading..."):
                time.sleep(5)
            st.subheader("Confusion Matrix")
            st.pyplot()
    if "Precision-Recall curve" in opt:
        with st.expander("ROC Curve"):
            from sklearn.metrics import (precision_recall_curve,PrecisionRecallDisplay) 
            predictions = model.predict(X_test)
            precision, recall, _ = precision_recall_curve(y_test, predictions)
            disp = PrecisionRecallDisplay(precision=precision, recall=recall)
            disp.plot()
            with st.spinner("Loading..."):
                time.sleep(5)
            st.subheader("Precision Recall Curve")
            st.pyplot()

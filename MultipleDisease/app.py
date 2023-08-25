import joblib;
import numpy as np;
import streamlit as st;
from streamlit_option_menu import option_menu;


heart_model = joblib.load('heart_model.joblib');
diabetes_model = joblib.load('diabetes_model.joblib');
parkinson_model = joblib.load('parkinson_model.joblib');


heart_scaler = joblib.load("heart_scaler.joblib");
diabetes_scaler = joblib.load("diabetes_scaler.joblib");
parkinson_scaler = joblib.load("parkinsonScaler.joblib");



# diabetes_input = [6,148,72,35,0,33.6,0.627,50]
# heart_input = [62,0,0,160,164,0,0,145,0,6.2,0,3,3]
# parkinson_input = [119.99200,157.30200,74.99700,0.00784,0.00007,0.00370,0.00554,0.01109,0.04374,0.42600,0.02182,0.03130,0.02971,0.06545,0.02211,21.03300,0.414783,0.815285,-4.813031,0.266482,2.301442,0.284654]



def predictHeartDisease(x):
    x = np.array(x)
    x = x.reshape(1,-1)
    x = heart_scaler.transform(x)
    prediction = heart_model.predict(x)
    if (prediction[0] == 0):
     return 'The person does not have heart disease';

    else:
     return 'The person have heart disease';

def predictDiabetes(x):
    x = np.array(x);
    x = x.reshape(1,-1)
    x = diabetes_scaler.transform(x)
    prediction = diabetes_model.predict(x)
    if (prediction[0] == 0):
     return 'The person is not diabetic';
    else:
     return 'The person is diabetic';

def predictParkinson(x):
    x = np.array(x);
    x = x.reshape(1,-1)
    x = parkinson_scaler.transform(x)
    prediction = parkinson_model.predict(x)
    if (prediction[0] == 0):
     return 'The person is Healthy';
    else:
     return 'The person is Parkinson Postive';



def main():

    prediction = ''

    with st.sidebar:
    
      selected = option_menu('Multiple Disease Prediction System',
                          ['Diabetes Prediction',
                           'Heart Disease Prediction',
                           'Parkinsons Prediction'],
                          icons=['activity','heart','person'],
                          default_index=0)
        


    if (selected == 'Diabetes Prediction'):
        st.title('Diabetes Prediction using ML')
        
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            Pregnancies = st.text_input('Number of Pregnancies')
            
        with col2:
            Glucose = st.text_input('Glucose Level')
        
        with col3:
            BloodPressure = st.text_input('Blood Pressure value')
        
        with col1:
            SkinThickness = st.text_input('Skin Thickness value')
        
        with col2:
            Insulin = st.text_input('Insulin Level')
        
        with col3:
            BMI = st.text_input('BMI value')
        
        with col1:
            DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
        
        with col2:
            Age = st.text_input('Age of the Person')
        
        diabetes_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]

        if st.button('Diabetes Test Result'):
            prediction = predictDiabetes(diabetes_input)

        if len(prediction) == 0:
            st.warning('Fill the values to see the result');  
            return   
        if('not' in prediction or 'Health' in prediction):    
            st.success(prediction)
        else :
            st.error(prediction)
    
        # For Heart Disease Prediciton

    if (selected == 'Heart Disease Prediction'):
        st.title('Heart Disease Prediction using ML')
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.text_input('Age')
            
        with col2:
            sex = st.text_input('Sex')
            
        with col3:
            cp = st.text_input('Chest Pain types')
            
        with col1:
            trestbps = st.text_input('Resting Blood Pressure')
            
        with col2:
            chol = st.text_input('Serum Cholestoral in mg/dl')
            
        with col3:
            fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')
            
        with col1:
            restecg = st.text_input('Resting Electrocardiographic results')
            
        with col2:
            thalach = st.text_input('Maximum Heart Rate achieved')
            
        with col3:
            exang = st.text_input('Exercise Induced Angina')
            
        with col1:
            oldpeak = st.text_input('ST depression induced by exercise')
            
        with col2:
            slope = st.text_input('Slope of the peak exercise ST segment')
            
        with col3:
            ca = st.text_input('Major vessels colored by flourosopy')
            
        with col1:
            thal = st.text_input('thal')
            
        heart_input = [age, sex, cp, trestbps, chol, fbs, restecg,thalach,exang,oldpeak,slope,ca,thal]
        
        if st.button('Heart Disease Test Result'):
            prediction = predictHeartDisease(heart_input)                          

        if len(prediction) == 0:
            st.warning('Fill the values to see the result');  
            return   
        if('not' in prediction or 'Health' in prediction):    
            st.success(prediction)
        else :
            st.error(prediction)

      #For Parkinson

    if (selected == "Parkinsons Prediction"):
            st.title("Parkinson's Disease Prediction using ML")
    
            col1, col2, col3, col4, col5 = st.columns(5)  
            
            with col1:
                mdvp_fo = st.text_input('MDVP:Fo(Hz)')
                
            with col2:
                mdvp_fhi = st.text_input('MDVP:Fhi(Hz)')
                
            with col3:
                mdvp_flo = st.text_input('MDVP:Flo(Hz)')
                
            with col4:
                mdvp_jitter_percent = st.text_input('MDVP:Jitter(%)')
                
            with col5:
                mdvp_jitter_abs = st.text_input('MDVP:Jitter(Abs)')
                
            with col1:
               mdvp_rap = st.text_input('MDVP:RAP')
                
            with col2:
                mdvp_ppq = st.text_input('MDVP:PPQ')
                
            with col3:
                jitter_ddp = st.text_input('Jitter:DDP')

        
            with col4:
                mdvp_shimmer = st.text_input('MDVP:Shimmer')
                
            with col5:
                mdvp_shimmer_db = st.text_input('MDVP:Shimmer(dB)')
                
            with col1:
                shimmer_apq3 = st.text_input('Shimmer:APQ3')
                
            with col2:
                shimmer_apq5 = st.text_input('Shimmer:APQ5')
                
            with col3:
                mdvp_apq = st.text_input('MDVP:APQ')
                
            with col4:
                shimmer_dda = st.text_input('Shimmer:DDA')
                
            with col5:
                nhr = st.text_input('NHR')
                
            with col1:
                hnr = st.text_input('HNR')
                
            with col2:
                rpde = st.text_input('RPDE')
                
            with col3:
                dfa  = st.text_input('DFA')
                
            with col4:
                spread1 = st.text_input('spread1')
                
            with col5:
                spread2 = st.text_input('spread2')
                
            with col1:
                d2 = st.text_input('D2')
                
            with col2:
                ppe = st.text_input('PPE')
        

            parkinson_input = [
                mdvp_fo, mdvp_fhi, mdvp_flo, mdvp_jitter_percent, mdvp_jitter_abs,
                mdvp_rap, mdvp_ppq, jitter_ddp, mdvp_shimmer, mdvp_shimmer_db,
                shimmer_apq3, shimmer_apq5, mdvp_apq, shimmer_dda, nhr,
                hnr, rpde, dfa, spread1, spread2, d2, ppe
            ]
              

            if st.button('Predict Parkinson Disease'):
                prediction = predictParkinson(parkinson_input)                         
                
            if len(prediction) == 0:
                st.warning('Fill the values to see the result');  
                return   
            if('not' in prediction or 'Health' in prediction):    
                st.success(prediction)
            else :
                st.error(prediction)
     


if __name__ == "__main__":
    main()










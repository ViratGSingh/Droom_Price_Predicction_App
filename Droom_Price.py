import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import xgboost
#st.write("")
# Draw a title and some text to the app:
'''
# Droom Price Prediction App


'''
# Add a selectbox to the sidebar:
# Add a slider to the sidebar:
st.sidebar.subheader("User Input Parameters")
st.subheader("Car Details")
def User_inputs():
        year= st.sidebar.slider('Year of purchase', 2000, 2020)
        trust_score=st.sidebar.slider( 'Trust score on droom', 0.0, 10.0)
        kms_driven=st.sidebar.slider('Kilometres driven ',500,150000)

        reg_state= st.sidebar.selectbox('Registration State',('Karnataka', 'Not Karnataka'))
        if(reg_state=="Karnataka"):
            reg_state_code=0
        else:
            reg_state_code=1


        fuel_type= st.sidebar.selectbox('Fuel',('Petrol','Diesel','Petrol+Cng','Electric','Petrol+Lpg'))
        if(fuel_type=="Diesel"):
            fuel_type_code=0
        elif(fuel_type=="Petrol"):
            fuel_type_code=2
        elif(fuel_type=="Petrol+Cng"):
            fuel_type_code=3  
        elif(fuel_type=="Petrol+Lpg"):
            fuel_type_code=4           
        else:
            fuel_type_code=1


        transmission= st.sidebar.selectbox('Transmission',('Automated Manual', 'Automatic', 'Manual'))
        if(transmission=="Automatic"):
            transmission_code=0
        elif(transmission=="Manual"):
            transmission_code=2    
        elif(transmission=="Automated Manual") :
            transmission_code=1   
        
        brands=('Maruti Suzuki', 'Renault', 'Nissan', 'Datsun', 'Hyundai','Mercedes-Benz', 'Volkswagen', 'Toyota', 'Tata', 'Ford',
        'Land Rover', 'Volvo', 'BMW', 'Mahindra', 'Honda', 'Skoda', 'Audi',
        'Jaguar', 'Mini', 'Jeep', 'Lexus', 'Porsche', 'Mitsubishi',
        'Chevrolet', 'Fiat', 'Mahindra Ssangyong', 'Rolls Royce',
        'Maserati', 'Mahindra Renault', 'DC')
        brand= st.sidebar.selectbox('Brand',brands)
        brand_dict={"Maruti Suzuki":16, 
                     "Renault" :23,
                     "Nissan" :21,
                     "Datsun" : 4,
                     "Hyundai" : 8,
                     "Mercedes-Benz" : 18,
                     "Volkswagen" : 28,
                     "Toyota" : 27,
                     "Tata" : 26,
                     "Ford" : 6,
                     "Land Rover" : 11,
                     "Volvo" : 29,
                     "BMW" : 1,
                     "Mahindra" : 13,
                     "Honda" : 7,
                     "Skoda" : 25,
                     "Audi" : 0,
                     "Jaguar" : 9,
                     "Mini" : 19,
                     "Jeep" : 10,
                     "Lexus" : 12,
                     "Porsche" : 22,
                     "Mitsubishi" : 20,
                     "Chevrolet" : 2,
                     "Fiat" : 5,
                     "Mahindra Ssangyong" : 15,
                     "Rolls Royce" : 24,
                     "Maserati" : 17,
                     "Mahindra Renault" : 14,
                     "DC" : 3
                    }
        for i in brands:
            if brand==i:
                brand_code=brand_dict[i]
            else:
                continue   
        
        data= {"Trust_Score":trust_score,
                 "Kms_Driven":kms_driven,
                 "Year":year,
                 "RegState_Codes":reg_state_code,
                 "Brand_Codes":brand_code,
                 "Fuel_Codes":fuel_type_code,
                 "Trans_Codes":transmission_code
                }
        features =pd.DataFrame(data,index=[0])
        st.write("1. Year of purchase: ",year)
        st.write("2. Trust score of seller on Droom(take 7 if not on droom): ",trust_score)
        st.write("3. Total number of kilometres driven: ",kms_driven,"Kms")
        st.write("4. Registration state: ",reg_state)
        st.write("5. Fuel type: ",fuel_type)
        st.write("6. Transmission: ",transmission)
        st.write("7. Brand: ",brand)  
        return features 
          
#st.subheader('Features')                       
values=User_inputs()


df=pd.read_csv("droom_price_str.csv")

X = df[["Trust_Score","Kms_Driven","Year","RegState_Codes","Brand_Codes","Fuel_Codes","Trans_Codes"]]
Y = df["Price"]
standardScaler = StandardScaler()
standardScaler.fit(X)
X= standardScaler.transform(X)

values=standardScaler.transform(values)

xgb=xgboost.XGBRegressor(base_score=0.25, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, gamma=0,
             importance_type='gain', learning_rate=0.1, max_delta_step=0,
             max_depth=10, min_child_weight=3, missing=None, n_estimators=100,
             n_jobs=1, nthread=None, objective='reg:linear', random_state=0,
             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
             silent=None, subsample=1, verbosity=1)
xgb.fit(X,Y)

prediction = xgb.predict(values)
prediction=np.exp(prediction)

st.subheader('Prediction')
st.write("Price: Rs.",prediction[0])
#st.write(prediction)
    

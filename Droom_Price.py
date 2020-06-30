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
name_dict={"Maruti Suzuki":("A-star","Alto","Alto 800","Alto K10","Baleno","Celerio","Celerio X","Ciaz","Dzire","Eeco","Ertiga","Esteem","Grand Vitara","Gypsy","Ignis","Kizashi","Omni","Ritz","S-cross","Stingray","Swift","Swift Dzire","Sx4","Versa","Vitara Brezza","Wagon R","Wagon R 1.0","Wagon R Duo","Wagon R Stingray","Zen","Zen Estilo","800","1000" ),
           "Renault":("Captur","Duster","Fluence","Koleos","Kwid","Kwid Ev","Lodgy","Pulse","Scala"),
           "Nissan":("350z","Evalia","Gt-r","Kicks","Micra","Micra Active","Sunny","Teana","Terrano","X-trail"),
           "Datsun":("Go","Go Plus","Redi-go"),
           "Hyundai":("Accent","Creta","Elantra","Elite I20","Eon","Fluidic Elantra","Getz","Getz Prime","Grand I10","Grand I10 Nios","Grand I10 Prime","I10","I20","I20 Active","Neo Fluidic Elantra","Santa Fe","Santro","Santro Xing","Sonata","Sonata Embera","Sonata Transform","Terracan","Tucson","Venue","Verna","Xcent","Xcent Prime" ),
           "Mercedes-Benz":("A Class","A-class","Amg","B-class","C-class","C-class Cabriolet","Cla","Cla Class","Clk","Cls","E-class","G-class","Gl","Gla","Gla-class","Glc","Gle","Gle Coupe","Gls","M-class","Mb 100","Mb-class","R Class","S-class","Sl Class","Slc","Slk","Slk-class","Sls","Viano","W212" ),
           "Volkswagen":("Ameo","Beetle","Caravelle","Cross Polo","Jetta","Passat","Phaeton","Polo","Tiguan","Touareg","Vento" ),
           "Toyota":("Alphard","Camry","Corolla","Corolla Altis","Corolla Levin","Corona","Crown","Etios","Etios Cross","Etios Liva","Fortuner","Glanza","Innova","Innova Crysta","Land Cruiser","Land Cruiser Prado","Mr-s","Platinum Etios","Prius","Qualis","Starlet","Tercel","Yaris" ),
           "Tata":("Ace","Aria","Bolt","Grande Dicor","Harrier","Hexa","Indica","Indica Ev2","Indica V2","Indica V2 Turbo","Indica V2 Xeta","Indica Vista","Indigo","Indigo Cs","Indigo Ecs","Indigo Marina","Indigo Xl","Indigocs","Magic","Manza","Movus","Nano","Nano Genx","Nexon","Nexon Ev","Safari","Safari Dicor","Safari Storme","Sierra","Sumo","Sumo Gold","Sumo Grande","Tiago","Tiago Jtp","Tiago Nrg","Tigor","Tigor Ev","Venture","Vista","Vista Tech","Winger","Xenon Xt","Zest","207" ),
           "Ford":("Aspire","Classic","Ecosport","Endeavour","Escort","Fiesta","Fiesta Classic","Figo","Figo Aspire","Freestyle","Fusion","Ikon","Jeep","Mondeo","Mustang","Super Model" ),
           "Land Rover":("Discovery","Discovery 4","Discovery Sport","Freelander","Freelander 2","Range Rover","Range Rover Evoque","Range Rover Lwb","Range Rover Sport","Range Rover Velar","Range Rover Vogue" ),
           "Volvo":("S60","S80","S90","V40","V40 Cross Country","V90 Cross Country","Xc 90","Xc40","Xc60","Xc90" ),
           "BMW":("1 Series","3 Series","3 Series Gt","5 Series","5 Series Gt","6 Series","6 Series Gt","7 Series","I8","M Series","M2","M3","X1","X3","X4","X5","X6","Z4"),
           "Mahindra":("Alturas G4","Armada","Bolero","Bolero Camper","Bolero Pick Up","E2o","E2o Plus","Genio","Jeep","Kuv 100","Kuv100","Kuv100 Nxt","Logan","Marazzo","Marshal","Maxx","Maxximo","Nuvosport","Quanto","Reva","S204","Scorpio","Scorpio Getaway","Supro","Thar","Tuv300","Verito","Verito Vibe","Verito Vibe Cs","Voyager","Xuv300","Xuv500","Xylo" ),
           "Honda":("Accord","Amaze","Br-v","Brio","City","City Zx","Civic","Civic Hybrid","Cr-v","Jazz","Mobilio","Stepwgn","Wr-v" ),
           "Skoda":("Fabia","Kodiaq","Laura","Octavia","Octavia Combi","Rapid","Superb","Yeti" ),
           "Audi":("A3","A3 Cabriolet","A4","A5","A5 Cabriolet","A6","A7","A8 L","Q3","Q5","Q7","R8","Rs 7 Sportback","Rs5","Rs6","S4","S5 Sportback","Tt" ),
           "Jaguar":("F-pace","F-type","Xe","Xf","Xj","Xj L","Xkr" ),
           "Mini":("Cooper","Cooper Convertible","Cooper Countryman","Cooper S","Countryman" ),
           "Jeep":("Compass","Grand Cherokee","Wrangler" ),
           "Lexus":("Es","Ls","Lx","Nx" ),
           "Porsche":("Boxster","Cayenne","Cayman","Macan","Panamera","718","911" ),
           "Mitsubishi":("Cedia","Diamante","Lancer","Lancer Evolution","Montero","Outlander","Pajero","Pajero Sport" ),
           "Chevrolet":("Aveo","Aveo U Va","Beat","Biscayne","Captiva","Cruze","Enjoy","Optra","Optra Magnum","Optra Srv","Sail","Sail Hatchback","Sail U-va","Spark","Tavera","Trailblazer","U-va" ),
           "Fiat":("Avventura","Grand Punto","Grande Punto","Linea","Linea Classic","Palio","Palio Adventure","Palio D","Palio Nv","Palio Stile","Petra","Petra D","Premier Padmini","Punto","Punto Evo","Punto Pure","Siena","Siena Weekend","Super Select","Uno" ),
           "Mahindra Ssangyong":("Rexton"),
           "Rolls Royce":("Ghost"),
           "Maserati":("Ghibli","Gran Turismo","Quattroporte" ),
           "Mahindra Renault":("Logan"),
           "DC":("Avanti")
           }
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
        model=st.sidebar.selectbox('Model',name_dict[brand])  
        data= {"Trust_Score":trust_score,
                 "Kms_Driven":kms_driven,
                 "Year":year,
                 "RegState_Codes":reg_state_code,
                 "Brand_Codes":brand_code,
                 "Fuel_Codes":fuel_type_code,
                 "Trans_Codes":transmission_code,
                 "Model_Name":model
                }
        features =pd.DataFrame(data,index=[0])
        st.write("1. Year of purchase: ",year)
        st.write("2. Trust score of seller on Droom(take 7 if not on droom): ",trust_score)
        st.write("3. Total number of kilometres driven: ",kms_driven,"Kms")
        st.write("4. Registration state: ",reg_state)
        st.write("5. Fuel type: ",fuel_type)
        st.write("6. Transmission: ",transmission)
        st.write("7. Brand: ",brand)  
        st.write("8. Model: ",model)  
        return features 
          
#st.subheader('Features')                       
values=User_inputs()


df=pd.read_csv("droom_price.csv")

X = df[["Trust_Score","Kms_Driven","Year","RegState_Codes","Brand_Codes","Fuel_Codes","Trans_Codes","Model_Name"]]
Y = df["Price"]
X=pd.concat([X,values])
X=X.reset_index(drop=True)
X["Model_Name"]=X["Model_Name"].astype('category')
X["Model_Name"]=X["Model_Name"].cat.codes
values=X[(len(X)-1):len(X)]
X=X.drop(X.index[(len(X)-1)])
X=X.reset_index(drop=True)
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
    

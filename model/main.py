import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,classification_report
import pickle


def create_model(data):
    scaler=MinMaxScaler()
    X=data.drop(['label'],axis=1)
    y=data['label']

    #scale
    X=scaler.fit_transform(X)
    
    #split the data
    X_train, X_test,y_train,y_test=train_test_split(
        X,y,test_size=0.2,random_state=42
    )

    #train
    model= KNeighborsClassifier(n_neighbors = 5, weights = 'distance',metric = 'manhattan')
    model.fit(X_train,y_train)

    y_pred=model.predict(X_test)

    print("Accuracu of our model: ",accuracy_score(y_test,y_pred))
    print("Classification report: \n",classification_report(y_test,y_pred))


    return model, scaler

def main():
    data=pd.read_csv('data/Crop_recommendation.csv')
    model,scaler=create_model(data)
    
    with open('model/model.pkl','wb') as f:
        pickle.dump(model,f)
    
    with open('model/scaler.pkl','wb') as f:
        pickle.dump(scaler,f)
    
if __name__=='__main__':
    main()




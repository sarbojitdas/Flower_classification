##iris dataset

#load packages
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris


#load dataset
iris=load_iris()

#print(iris)

#load iris database

db=iris.data

#print(db) 

#load the feature 
feature=iris.feature_names
#print(feature)



# divide the dataset into data(independent) and target(depenedent)
x=iris.data

y=iris.target
#plotting of graph
plt.plot(x[:, 0][y == 0] * x[:, 1][y == 0], x[:, 2][y == 0] * x[:, 3][y == 0], 'r.', label="Satosa")
plt.plot(x[:, 0][y == 1] * x[:, 1][y == 1], x[:, 2][y == 1] * x[:, 3][y == 1], 'g.', label="Virginica")
plt.plot(x[:, 0][y == 2] * x[:, 1][y == 2], x[:, 2][y == 2] * x[:, 3][y == 2], 'b.', label="Versicolour")
plt.title(' Plot showing the data for different flower species')

plt.xlabel('feature')
plt.ylabel('target')

plt.legend()

plt.show()



# Split the dataset

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.3)

#scale the data

from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
x_train=sc.fit_transform(x_train)

x_test=sc.fit_transform(x_test)

def models(x_train,y_train):
    #Logistic Regression
    from sklearn.linear_model import LogisticRegression
    log=LogisticRegression(random_state=0)
    log.fit(x_train,y_train)



    #Random Forest

    from sklearn.ensemble import RandomForestClassifier
    forest=RandomForestClassifier(n_estimators=10,criterion='entropy', random_state=0)
    forest.fit(x_train,y_train)

    #kmean

    from sklearn.neighbors import KNeighborsClassifier
    classifier = KNeighborsClassifier(n_neighbors = 10, metric = 'minkowski', p = 2)
    classifier.fit(x_train, y_train)

    #print model accuracy

    print('[0]Logistic Regression Training Accuracy:',log.score(x_train,y_train))

   # print('[1]Desicion Tree Classifier Training Accuracy:',tree.score(x_train,y_train))

    print('[1]Random Forest Training Accuracy:',forest.score(x_train,y_train))

    print('[2] K Nearest Neighbor Training Accuracy:',classifier.score(x_train,y_train))


    

    return log,forest,classifier

model=models(x_train,y_train)

#test model accuracy on test data on confusion matrix
from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test, model[0].predict(x_test))
for i in range(len(model)):
        TP=cm[0][0]
        TN=cm[1][1]
        FP=cm[0][1]
        FN=cm[1][0]

        print(cm)

        print('Testing Accuracy:',(TP+TN)/(TP+TN+FN+FP))

#another way to calculate the score
from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score
for i in range(len(model)):
    print("Model",i)
    print(classification_report(y_test, model[0].predict(x_test)))

    print(accuracy_score(y_test, model[0].predict(x_test)))

    print()


##print(x_train.shape)
##
##print(y_train.shape)

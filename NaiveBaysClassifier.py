import pandas as pd
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
dataset = pd.read_csv("Diabetes.csv")
print("Dataset Size: ", len(dataset))
split = int(len(dataset) * 0.7)
train, test = dataset.iloc[:split], dataset.iloc[split:]
p = train["Pregnancies"].values
g = train["Glucose"].values
bp = train["BloodPressure"].values
st = train["SkinThickness"].values
ins = train["Insulin"].values
bmi = train["BMI"].values
dpf = train["DiabetesPedigreeFunction"].values
a = train["Age"].values
d = train["Diabetes"].values
trainfeatures = zip(p, g, bp, st, ins, bmi, dpf, a)
traininput = list(trainfeatures)
# print(traininput)
model = GaussianNB()
model.fit(traininput, d)
p = test["Pregnancies"].values
g = test["Glucose"].values
bp = test["BloodPressure"].values
st = test["SkinThickness"].values
ins = test["Insulin"].values
bmi = test["BMI"].values
dpf = test["DiabetesPedigreeFunction"].values
a = test["Age"].values
d = test["Diabetes"].values
testfeatures = zip(p, g, bp, st, ins, bmi, dpf, a)
testinput = list(testfeatures)
predicted = model.predict(testinput)

print("Confusion Matrix:")
print(metrics.confusion_matrix(d, predicted))
print("\nClassification Measures:")
print("Accuracy:", metrics.accuracy_score(d, predicted))
print("Recall:", metrics.recall_score(d, predicted))
print("Precision:", metrics.precision_score(d, predicted))
print("F1-score:", metrics.f1_score(d, predicted))
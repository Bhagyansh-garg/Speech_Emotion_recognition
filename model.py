import os, glob
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import joblib  
from audio import extract_feature
import matplotlib.pyplot as plt
from collections import Counter

# Define emotions
emotions={
  '01':'neutral',
  '02':'calm',
  '03':'happy',
  '04':'sad',
  '05':'angry',
  '06':'fearful',
  '07':'disgust',
  '08':'surprised'
}


# Load dataset
def load_data(test_size=0.2):
    x,y=[],[]
    for file in glob.glob("speech-emotion-recognition-ravdess-data/Actor_*/*.wav"):
        file_name=os.path.basename(file)
        emotion=emotions[file_name.split("-")[2]]
        feature=extract_feature(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)
    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)


# Split dataset
x_train, x_test, y_train, y_test = load_data()


#Get the shape of the training and testing datasets
print((x_train.shape[0], x_test.shape[0]))


#Get the number of features extracted
print(f'Features extracted: {x_train.shape[1]}')


# Standardize features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# Train MLP Classifier
params = {
    'hidden_layer_sizes': [(100,), (300, 100)],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate': ['constant', 'adaptive'],
}
grid = GridSearchCV(MLPClassifier(max_iter=500), param_grid=params, cv=3)
grid.fit(x_train, y_train)


# Save Model and Scaler
joblib.dump(grid, "emotion_model.pkl")
joblib.dump(scaler, "scaler.pkl")


# Model Accuracy
y_pred=grid.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")


emotion_counts = Counter(y_train)

# Bar Chart
plt.figure(figsize=(6, 4))
plt.bar(emotion_counts.keys(), emotion_counts.values(), color="#007acc")
plt.xlabel("Emotion", fontsize=12, fontweight="bold")
plt.ylabel("Count", fontsize=12, fontweight="bold")
plt.title("Class Distribution", fontsize=14, fontweight="bold")
plt.xticks(rotation=45)
plt.savefig("bar_chart.png", bbox_inches='tight')
plt.close()

# Pie Chart
plt.figure(figsize=(6, 6))
plt.pie(emotion_counts.values(), labels=emotion_counts.keys(), autopct='%1.1f%%', colors=["#FF9999", "#66B3FF", "#99FF99", "#FFD700", "#FF4500", "#8A2BE2", "#8B4513", "#32CD32"])
plt.title("Class Distribution (Pie Chart)", fontsize=14, fontweight="bold")
plt.savefig("pie_chart.png", bbox_inches='tight')
plt.close()
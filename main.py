import pandas as pd
import time
from transformers import pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, matthews_corrcoef, cohen_kappa_score,log_loss
from datasets import load_dataset
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

models = [
    "Sakil/IMDB_URDUSENTIMENT_MODEL",
    "mnoukhov/gpt2-imdb-sentiment-classifier",
    "AdapterHub/bert-base-uncased-pf-imdb",
    "wrmurray/roberta-base-finetuned-imdb",
    "kurianbenoy/distilbert-base-uncased-finetuned-sst-2-english-finetuned-imdb",

]

dataset = load_dataset('imdb',streaming=True)

data = [item  for item in dataset['test'] if len(item['text']) <512]

df = pd.DataFrame(data)

NUM_ROWS = 2000
newDf = pd.concat([df[df.label == 0].head(NUM_ROWS //2),df[df.label ==1].head(NUM_ROWS//2)]).sample(frac = 1).reset_index(drop = True)
# newDf.head()
texts = newDf.text.tolist()
labels = newDf.label.tolist()
sns.countplot(x = labels)
FORMAT_LABELS ={
    'Label_0':0,
    "Label_1":1,
    "POSITIVE":1,
    "NEGATIVE":0,
    'neg':0,
    'pos':1
}

def evaluate_model(model_name):

    print('model initialized')
    pipe = pipeline("text-classification", model=model_name)
    start_time = time.time()
    res = pipe(texts)
    end_time = time.time()

    predicted_labels = list(map(lambda x: FORMAT_LABELS.get(x['label'],0),res))
    probs = [ item['score'] for item in res ]
    print('Calculate evaluation metrics')
    # Calculate evaluation metrics
    accuracy = accuracy_score(labels, predicted_labels)
    precision = precision_score(labels, predicted_labels, average="weighted")
    recall = recall_score(labels, predicted_labels, average="weighted")
    f1 = f1_score(labels, predicted_labels, average="weighted")
    roc_auc = roc_auc_score(labels, predicted_labels)
    avg_precision = average_precision_score(labels, predicted_labels)
    mcc = matthews_corrcoef(labels, predicted_labels)
    kappa = cohen_kappa_score(labels, predicted_labels)
    logloss = log_loss(labels,probs)

    # Calculate training time
    training_time = end_time - start_time

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc,
        "Average Precision": avg_precision,
        "Matthews Correlation Coefficient": mcc,
        "Cohen's Kappa": kappa,
        "Time (s)": training_time,
        'Log Loss': logloss,
    }
results = []

for model_name in models:
        try:
            result = evaluate_model(model_name)
            results.append(result)
            print("Done",model_name)
        except:
            print(model_name)

# making a dataframe
df = pd.DataFrame(results)
df.head()
df.to_csv("Inputdata.csv")

def topsis(df: pd.DataFrame, wts: np.ndarray, impact: np.ndarray) -> pd.DataFrame:
    mat = np.array(df.iloc[:, 1:])
    rows, cols = mat.shape


    for i in range(cols):
        temp = 0
        for j in range(rows):
            temp += mat[j][i] ** 2
        temp = temp**0.5
        wts[i] /= temp

    weightedNormalized = mat * wts

    idealBestWorst = []

    for i in range(cols):
        maxi = weightedNormalized[:, i].max()
        mini = weightedNormalized[:, i].min()
        idealBestWorst.append((maxi, mini) if impact[i] == 1 else (mini, maxi))
    topsisScore = []
    for i in range(rows):
        temp_p, temp_n = 0, 0
        for j in range(cols):
            temp_p += (weightedNormalized[i][j] - idealBestWorst[j][0]) ** 2
            temp_n += (weightedNormalized[i][j] - idealBestWorst[j][1]) ** 2
        temp_p, temp_n = temp_p**0.5, temp_n**0.5

        topsisScore.append(temp_n / (temp_p + temp_n))

    df["score"] = np.array(topsisScore).T
    df["rank"] = df["score"].rank(method="max", ascending=False)
    df["rank"] = df.astype({"rank": int})["rank"]
    return df

df1 = pd.read_csv("Inputdata.csv",index_col=0)
wts =np.array([10]*10)
impacts = np.array([1,1,1,1,1,1,1,1,-1,-1])

newDF = topsis(df1,wts,impacts)
newDF.to_csv('Output.csv')
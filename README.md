
## Topsis For Pretrained Model Text Classification

Welcome to the Topsis-Pretrained-Text-Classification repository! This project harnesses the power of TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution) for fine-tuning and evaluating pretrained models for text classification tasks. If you're looking to enhance the performance of your text classification models with minimal effort, this repository is your go-to resource.

## Acknowledgements

 - [Hugging Face](https://huggingface.co/)
 - [Pytorch](https://pytorch.org/)
 - [TensorFlow](https://www.tensorflow.org/)


## Parameter Evaluation


#### Parameter Evaluation Notebook: [ParameterEvaluation.ipynb](https://colab.research.google.com/drive/1ej88-dWaA2YeuOnPqTAYMl9FHsnmbnBo?usp=sharing)

Certainly! The provided code snippet evaluates various performance metrics for pretrained models. Here's a description for each of the metrics:

####  Accuracy:

Measures the overall correctness of the model by comparing the correctly predicted instances to the total instances.
#### Precision:

Reflects the accuracy of positive predictions, indicating the proportion of correctly predicted positive instances among all instances predicted as positive.
#### Recall:

Represents the ability of the model to capture all relevant instances by measuring the proportion of correctly predicted positive instances among all actual positive instances.
#### F1 Score:

Harmonic mean of precision and recall, providing a balance between precision and recall. It is useful when there is an uneven class distribution.
#### ROC-AUC (Receiver Operating Characteristic - Area Under the Curve):

Evaluates the model's ability to discriminate between positive and negative instances. A higher ROC-AUC indicates better discrimination.
#### Average Precision:

Computes the average precision for different recall values, providing a summary measure of the precision-recall curve.
#### Matthews Correlation Coefficient:

Takes into account true positives, true negatives, false positives, and false negatives to measure the correlation between predicted and actual classifications.
#### Cohen's Kappa:

Evaluates the agreement between predicted and actual classifications while considering the possibility of random agreement.
#### Time (s):

Represents the time taken for training the pretrained models, providing insights into the computational efficiency of the models.
#### Log Loss:

Measures the performance of a classification model by quantifying the accuracy of probability estimates. Lower log loss values indicate better model performance.


These metrics collectively offer a comprehensive evaluation of the pretrained models, covering aspects of accuracy, precision, recall, balance, discrimination ability, and computational efficiency. The interpretation of these metrics depends on the specific context and goals of the machine learning task at hand.


## Topsis Analysis

#### Topsis Analysis Notebook: [TopsisAnalysis.ipynb](https://colab.research.google.com/drive/1-nsEHwaZxnd1K86VWRLM9hlviUPlfoGx?usp=sharing)

### Input Data
`inputdata.csv`
| Model                                                 | Accuracy | Precision | Recall | F1 Score      | ROC-AUC | Average Precision | Matthews Correlation Coefficient | Cohen's Kappa | Time (s)    | Log Loss     |
| ------------------------------------------------------ | -------- | ----------| ------ | ------------- | ------- | ------------------ | -------------------------------- | ------------- | ----------- | ------------ |
| Sakil/IMDB_URDUSENTIMENT_MODEL                        | 0.5      | 0.25      | 0.5    | 0.333333333   | 0.5     | 0.5                | 0                                | 0             | 232.2006447 | 0.693560283  |
| mnoukhov/gpt2-imdb-sentiment-classifier               | 0.5      | 0.25      | 0.5    | 0.333333333   | 0.5     | 0.5                | 0                                | 0             | 443.7905507 | 2.84753836   |
| wrmurray/roberta-base-finetuned-imdb                  | 0.5      | 0.25      | 0.5    | 0.333333333   | 0.5     | 0.5                | 0                                | 0             | 426.102942  | 3.108170235  |
| kurianbenoy/distilbert-base-uncased-finetuned-sst-2-english-finetuned-imdb | 0.925 | 0.926152316 | 0.925 | 0.924949266   | 0.925   | 0.884196768       | 0.851151536                     | 0.85          | 222.5569022 | 2.817650585  |

### Output Data
`output.csv`
| Model                                                 | Accuracy | Precision | Recall | F1 Score      | ROC-AUC | Average Precision | Matthews Correlation Coefficient | Cohen's Kappa | Time (s)    | Log Loss     | Score        | Rank |
| ------------------------------------------------------ | -------- | --------- | ------ | ------------- | ------- | ------------------ | -------------------------------- | ------------- | ----------- | ------------ | ------------ | ---- |
| Sakil/IMDB_URDUSENTIMENT_MODEL                        | 0.5      | 0.25      | 0.5    | 0.333333333   | 0.5     | 0.5                | 0                                | 0             | 232.2006447 | 0.693560283  | 0.126805798  | 2    |
| mnoukhov/gpt2-imdb-sentiment-classifier               | 0.5      | 0.25      | 0.5    | 0.333333333   | 0.5     | 0.5                | 0                                | 0             | 443.7905507 | 2.84753836   | 0.015307196  | 3    |
| wrmurray/roberta-base-finetuned-imdb                  | 0.5      | 0.25      | 0.5    | 0.333333333   | 0.5     | 0.5                | 0                                | 0             | 426.102942  | 3.108170235  | 0            | 4    |
| kurianbenoy/distilbert-base-uncased-finetuned-sst-2-english-finetuned-imdb | 0.925 | 0.926152316 | 0.925 | 0.924949266   | 0.925   | 0.884196768       | 0.851151536                     | 0.85          | 222.5569022 | 2.817650585  | 0.886738215  | 1    |



## Analysis
### Parameters Analysis
![Parameters Analysis](https://raw.githubusercontent.com/pulkit8690/Topsis-For-Pretrained-Model-Text-Classification-/main/parameter.png)

### Topsis Score vs Model
![Topsis Score vs Model](https://github.com/pulkit8690/Topsis-For-Pretrained-Model-Text-Classification-/blob/main/comparison.png)


# Machine Learning Application: Predicting Wine Quality

Machine learning is an application of AI that enables systems to learn and improve from experience without being explicitly programmed. Machine learning focuses on developing computer programs that can access data and use it to learn for themselves.

Furthermore, it has proven valuable because it can solve problems at a speed and scale that cannot be duplicated by the human mind alone. With massive amounts of computational ability behind a single task or multiple specific tasks, machines can be trained to identify patterns in and relationships between input data and automate routine processes. 

For this project, I used Kaggle’s 'Red Wine Quality' dataset to build various classification models to predict whether a particular red wine is “good" or "bad" quality. Each wine in this dataset is given a “quality” score between 0 and 8. The quality of a wine is determined by 11 input variables:

    1) Fixed acidity
    2) Volatile acidity
    3) Citric acid
    4) Residual sugar
    5) Chlorides
    6) Free sulfur dioxide
    7) Total sulfur dioxide
    8) Density
    9) pH
    10) Sulfates
    11) Alcohol

Classification algorithms in machine learning use input training data to predict the likelihood that subsequent data will fall into one of the predetermined categories.

Therefore, the main aim of this project is to experiment with different classification methods and determine which one yields the highest accuracy rate.

 

## Acknowledgements

 - [Datasets for Machine Learning](https://pub.towardsai.net/best-datasets-for-machine-learning-data-science-computer-vision-nlp-ai-c9541058cf4f)
 - [How Machine Learning Works](https://www.ibm.com/cloud/learn/machine-learning)
 - [Classification Models](https://www.educative.io/blog/scikit-learn-cheat-sheet-classification-regression-methods)

## New Digital Age

“Unprecedented and simultaneous advances in artificial intelligence (AI), robotics,
the internet of things, autonomous vehicles, 3D printing, nanotechnology, biotechnology, materials science, energy storage, quantum computing and others are
redefining industries, blurring traditional boundaries, and creating new opportunities. We have dubbed this the Fourth Industrial Revolution, and it is fundamentally
changing the way we live, work and relate to one another.”
—Professor Klaus Schwab, 2016

It is common practice to prototype concepts in a language such as
Python, then port the solution over to Java or C++ for deployment. Another
recent development is the availability of cloud-based Machine-Learning-asa-Service platforms that include:

    • Amazon Machine Learning
    • DataRobot
    • Google Prediction
    • IBM Watson, and
    • Microsoft Azure Machine Learning.

Here is a list of languages commonly used in machine
learning:

    • Python
    • MATLAB/Octave
    • R
    • Java/Scala
    • C/C++
    • Julia, and
    • Go.

One of the core technologies driving the 4th industrial revolution is
machine learning.
## Authors

- [@MichaelTobiko](https://www.github.com/miketobz)


## API Reference

#### MLP Classification Trainer:

```http
  from sklearn.neural_network import MLPClassifier
```
class ggml.classification.MLPClassificationTrainer(arch, env_builder=<ggml.common.LearningEnvironmentBuilder object>, loss='mse', learning_rate=0.1, max_iter=1000, batch_size=100, loc_iter=10, seed=None)¶
Bases: ggml.classification.ClassificationTrainer

__init__(arch, env_builder=<ggml.common.LearningEnvironmentBuilder object>, loss='mse', learning_rate=0.1, max_iter=1000, batch_size=100, loc_iter=10, seed=None)
Constructs a new instance of MLP classification trainer.

env_builder : Environment builder. arch : Architecture. loss : Loss function (‘mse’, ‘log’, ‘l2’, ‘l1’ or ‘hinge’, default value is ‘mse’). update_strategy : Update strategy. max_iter : Max number of iterations. batch_size : Batch size. loc_iter : Number of local iterations. seed : Seed.


#### SVM classification trainer:

```http
  from sklearn.svm import SVC
  from sklearn import svm
```
class ggml.classification.SVMClassificationTrainer(env_builder=<ggml.common.LearningEnvironmentBuilder object>, l=0.4, max_iter=200, max_local_iter=100, seed=1234)
Bases: ggml.classification.ClassificationTrainer

__init__(env_builder=<ggml.common.LearningEnvironmentBuilder object>, l=0.4, max_iter=200, max_local_iter=100, seed=1234)
Constructs a new instance of SVM classification trainer.

env_builder : Environment builder. l : Lambda. max_iter : Max number of iterations. max_loc_iter : Max number of local iterations. seed : Seed.


#### RandomForest classification trainer:

```http
  from sklearn.ensemble import RandomForestClassifier
```
class ggml.classification.RandomForestClassificationTrainer(features, env_builder=<ggml.common.LearningEnvironmentBuilder object>, trees=1, sub_sample_size=1.0, max_depth=5, min_impurity_delta=0.0, seed=None)
Bases: ggml.classification.ClassificationTrainer

__init__(features, env_builder=<ggml.common.LearningEnvironmentBuilder object>, trees=1, sub_sample_size=1.0, max_depth=5, min_impurity_delta=0.0, seed=None)¶
Constructs a new instance of RandomForest classification trainer.

features : Number of features. env_builder : Environment builder. trees : Number of trees. sub_sample_size : Sub sample size. max_depth : Max depth. min_impurity_delta : Min impurity delta. seed : Seed.


## 🔗 Links

[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/michael-tobiko-1563a693)
[![twitter](https://img.shields.io/badge/twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white)](https://twitter.com/MichaelTobiko)


## Installation

Install my-project with npm

```bash
  npm install my-project
  cd my-project
```
    
![Logo](https://www.einfochips.com/blog/wp-content/uploads/2018/11/how-to-develop-machine-learning-applications-for-business-featured.jpg)


## Demo

![Machine Learning Demo](https://miro.medium.com/max/1400/1*SuKim2w9IPbRxdtz23UNcw.gif)

## Deployment

To deploy this project run

```bash
  npm run deploy
```

```bash
  pip install -r requirements.txt
```
## Screenshots

![ML Screenshot](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQUAI8CYSFw1EAyc7Pfst8eOpgnxB-w3BFIxQ&usqp=CAU)

![ML Screenshot](https://serokell.io/files/cr/crlo72ua.22_(2)_(1).jpg)


## Features

- Live previews
- Fullscreen mode
- Cross platform


## Dataset

Download the dataset used for wine quality predictions:

[Download .CSV file](https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/)


## Lessons Learned

What did you learn while building this project? 

    1)  The Apache OpenOffice spreadsheet application, named Calc, is very effective in handling CSV files.
        Use it to preview the dataset ('winequality-red.csv) for more clarity.

    2)  The Random Forest Algorithm is best suited for handling data classification problems.


# Hi, I'm Michael Tobiko 👋


## 🚀 About Me
I'm a Data Analyst & Innovator.


## 🛠 Skills
Python, C++, MySQL, R


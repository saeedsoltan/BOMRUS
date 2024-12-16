import pandas as pd
import numpy as np
from imblearn.ensemble import RUSBoostClassifier
from sklearn.model_selection import cross_val_score
from skopt import BayesSearchCV
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('fulldata.csv', skiprows=0)
df_train=df[df['test-train']=='Train']
df_test=df[df['test-train']=='Test']
df_train_np = df_train.to_numpy()
df_test_np = df_test.to_numpy()
X_train = df_train_np[:,0:19]
y_train = df_train_np[:,-5]
X_test = df_test_np[:,0:19]
y_test= df_test_np[:,-3]


def objective_function(params):
    clf = RUSBoostClassifier(**params)
    return -np.mean(cross_val_score(clf, X_train, y_train, cv=10, n_jobs=-1, scoring='accuracy'))

search_space = {
    'base_estimator__max_depth': (3,50),
    'n_estimators': (10,100),
    'learning_rate': (0.01, 1),
    'algorithm': ['SAMME', 'SAMME.R'],
    'sampling_strategy': ['not majority', 'all', 'auto', 'majority', 'not minority'],
    'random_state': [42]
}

opt = BayesSearchCV(
    RUSBoostClassifier(base_estimator=DecisionTreeClassifier()),
    search_space,
    n_iter=50,
    cv=10,
    n_jobs=-1,
    scoring='accuracy'
)


opt.fit(X_train, y_train)
best_params = opt.best_params_
print("Best hyperparameters:", best_params)
best_model = opt.best_estimator_
test_score = best_model.score(X_test, y_test)
print("Test score with best hyperparameters:", test_score)
y_pred = best_model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
cm_display = ConfusionMatrixDisplay(cm).plot()




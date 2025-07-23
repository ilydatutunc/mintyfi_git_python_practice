Train class distribution:
label
ham 0.873761
spam 0.126239
Name: proportion, dtype: float64
Validation class distribution:
label
ham 0.873308
spam 0.126692
Name: proportion, dtype: float64
train: (4135, 7414), valid: (1034, 7414)

Model: Logistic Regression
Accuracy: 0.9564796905222437
Precision: 0.9777777777777777
Recall: 0.6717557251908397
F1 Score: 0.7963800904977375
Confusion Matrix:
[[901 2]
[ 43 88]]

Model: Multinomial NB
Accuracy: 0.9671179883945842
Precision: 1.0
Recall: 0.7404580152671756
F1 Score: 0.8508771929824561
Confusion Matrix:
[[903 0]
[ 34 97]]

Model: Random Forest
Accuracy: 0.9787234042553191
Precision: 0.990990990990991
Recall: 0.8396946564885496
F1 Score: 0.9090909090909091
Confusion Matrix:
[[902 1]
[ 21 110]]

# kaggle-competition
The goal of this competition is the prediction of the price of diamonds based on their characteristics (weight, color, quality of cut, etc.), putting into practice all the machine learning techniques you know.
![Alt-Text](/INPUT/diamond.png)


Data: https://www.kaggle.com/c/diamonds-datamad0320/overview

Files in data:

* diamonds_train.csv: training set
* diamond_test.csv: test set
* sample_submission.csv: sample submission

Features:

* id: only for test & sample submission files, id for prediction sample identification
* price: price in USD
* carat: weight of the diamond
* cut: quality of the cut (Fair, Good, Very Good, Premium, Ideal)
* color: diamond colour, from J (worst) to D (best)
* clarity: a measurement of how clear the diamond is (I1 (worst), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (best))
* x: length in mm
* y: width in mm
* z: depth in mm
* depth: total depth percentage = z / mean(x, y) = 2 * z / (x + y) (43--79)
* table: width of top of diamond relative to widest point (43--95)

Estimators:

* LinearRegression-->Very bad results (deleted)
* DecisionTreeRegressor--> rsme: 648488.756
      			    r2score: 0.9586950351109729
* KNeighborsRegressor--> rsme: 1548948.478
      			  r2score: 0.9016002419155986
* GradientBoostingRegressor-->rsme: 365376.08
      			       r2score: 0.9766642204138182
* RandomForestRegressor-->
* HistGradientBoostingRegressor--> rsme: 299148.833
      				    r2score: 0.9812237906981642


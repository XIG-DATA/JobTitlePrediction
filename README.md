# README #

14th solution for Job Title competition of DataCastle (http://www.pkbigdata.com/common/competition/147.html)
### Task ###

* Prediction major given a person's job experience 
* Predict a person's second last company size
* Predict a person's salary in its second last company 
* Predict a person's job title in its second last company 
* The final result is weighted averaged of the four tasks

### Feature  Engineering###

* TFIDF of position, department descriptions 
* some hard-coded features 


### Models ###

* Xgboost, DNN, CNN
* The final model is ensemble of the 3 models
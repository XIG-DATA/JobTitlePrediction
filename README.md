# README #

14th solution for Job Title competition of DataCastle (http://www.pkbigdata.com/common/competition/147.html)
### Task ###

* Predict a person's major given a person's job experience 
* Predict a person's second last company size
* Predict a person's salary in its second last company 
* Predict a person's job title in its second last company 
* The final result is weighted averaged of the four tasks

### Feature  Engineering###

* TFIDF of position, department descriptions 
*  Cross-entropy features of A with respect to target C
* some hard-coded features 


### Models ###

* Xgboost, DNN, CNN with 1d convolution
* The final model is ensemble of the 3 models
#Linear

2 0


library(vip)
vip(lm_fit)

#Poisson
summary(m1 <- glm(SeonsorID_ ~ bikeshop_countsNUMPOINTS + edupois_reprojected_joinNUMPOINTS + 
                    shoppoints_count_NUMPOINTS + trafficpoints_count_NUMPOINTS + trafficsignals_count_NUMPOINTS + 
                    NDVImean20 + LSTmean + slope + UC_bikeshops_HubDist + UC_edu_HubDist + UC_shops_HubDist + 
                    UC_streetlights_HubDist + UC_trafficpoints_HubDist, family="poisson", data=train.set.sample))

poi_fit <- poi_plan %>% 
  fit(SeonsorID_ ~ bikeshop_countsNUMPOINTS + edupois_reprojected_joinNUMPOINTS + 
        shoppoints_count_NUMPOINTS + trafficpoints_count_NUMPOINTS + trafficsignals_count_NUMPOINTS + 
        NDVImean20 + LSTmean + slope + UC_bikeshops_HubDist + UC_edu_HubDist + UC_shops_HubDist +  
        UC_streetlights_HubDist + UC_trafficpoints_HubDist, data = train.set.sample)

# View lm_fit properties
poi_fit
summary(poi_fit$fit)
vip(poi_fit)

###RF
set.seed(234)
rf_fit <- rf_plan %>%
  fit(SeonsorID_ ~ bikeshop_countsNUMPOINTS + edupois_reprojected_joinNUMPOINTS + 
        shoppoints_count_NUMPOINTS + trafficpoints_count_NUMPOINTS + trafficsignals_count_NUMPOINTS + 
        NDVImean20 + LSTmean + slope + UC_bikeshops_HubDist + UC_edu_HubDist + UC_shops_HubDist +  
        UC_streetlights_HubDist + UC_trafficpoints_HubDist, data = citybound)
rf_fit
summary(rf_fit$fit)
vip(rf_fit)

#load libraries

#install.packages("easypackages")

easypackages::packages ("sf", "sp", "tmap", "mapview", "car", "RColorBrewer", 
                        "tidyverse", "osmdata", "nngeo", "FNN", "rpart", "rpart.plot", 
                        "randomForest", "sessioninfo", "caret", "rattle", "ipred", "tidymodels", 
                        "ranger", "recipes", "workflows", "themis","xgboost", "modelStudio", "DALEX", 
                        "DALEXtra", "vip", "pdp")

#Fit the decision tree 
#Note: We are using, method = "poisson". It can also be "a"nova", "poisson", "class" or "exp". Depending on the data type. In this case pedal is a count data so we selected poisson. 
DT0 <- rpart (SeonsorID_ ~ bikeshop_countsNUMPOINTS + edupois_reprojected_joinNUMPOINTS + 
                shoppoints_count_NUMPOINTS + trafficpoints_count_NUMPOINTS + trafficsignals_count_NUMPOINTS + 
                NDVImean20 + LSTmean + slope + UC_bikeshops_HubDist + UC_edu_HubDist + UC_shops_HubDist +  
                UC_streetlights_HubDist + UC_trafficpoints_HubDist, data = train.set.sample,  method  = "anova") 


summary (DT0)

rpart.plot(DT0)

#plot the complexity parameter
plotcp(DT0)

DT1 <- rpart (SeonsorID_ ~ bikeshop_countsNUMPOINTS + edupois_reprojected_joinNUMPOINTS + 
                shoppoints_count_NUMPOINTS + trafficpoints_count_NUMPOINTS + trafficsignals_count_NUMPOINTS + 
                NDVImean20 + LSTmean + slope + UC_bikeshops_HubDist + UC_edu_HubDist + UC_shops_HubDist +  
                UC_streetlights_HubDist + UC_trafficpoints_HubDist, data = train.set.sample,  method  = "poisson", 
              control = list(cp = 0)) 

#plot the cp values
plotcp(DT1)

DT1_pruned <- rpart::prune (DT1, cp = 0.01) # here I am selecting a value of 0.1, you can play with other values and see how the model changes with different values of cp.


#plot the model
rpart.plot(DT1_pruned)

#plot the cp values
plotcp(DT1_pruned)

#plot the model
rpart.plot(DT1)

#explain the model
#let us plot the feature importance for the top ten features from the model
vip (DT1_pruned, num_features = 19, aesthetics = list(fill = "green3"), include_type = T) 

#create model explainer
explainer_DT <- DALEX::explain(
  model = DT1_pruned,
  data = train.set,
  y = as.integer (train.set$pedal),
  label = "Decision Tree Purned",
  verbose = FALSE
)

#now make an interactive dashboard
modelStudio::modelStudio(explainer_DT)
##XGB
set.seed(234)
xgb_fit <- xgb_plan %>%
  fit(SeonsorID_ ~ bikeshop_countsNUMPOINTS + edupois_reprojected_joinNUMPOINTS + 
        shoppoints_count_NUMPOINTS + trafficpoints_count_NUMPOINTS + trafficsignals_count_NUMPOINTS + 
        NDVImean20 + LSTmean + slope + UC_bikeshops_HubDist + UC_edu_HubDist + UC_shops_HubDist +  
        UC_streetlights_HubDist + UC_trafficpoints_HubDist, data = citybound)
xgb_fit
summary(xgb_fit$fit)
vip(xgb_fit)

set.seed(123)
#fit a random forest model, here I am selecting some hyper parameter such as mtry = 6 you can test different numbers but we shall explore these in details in the following session.
#here the rule of thumb is to use p/3 variables for regression trees, here p is the number of predictors in the model, and we had 19 predictors
xgb_final <-xgboost(SeonsorID_ ~ bikeshop_countsNUMPOINTS + edupois_reprojected_joinNUMPOINTS + 
                      shoppoints_count_NUMPOINTS + trafficpoints_count_NUMPOINTS + trafficsignals_count_NUMPOINTS + 
                      NDVImean20 + LSTmean + slope + UC_bikeshops_HubDist + UC_edu_HubDist + UC_shops_HubDist +  
                      UC_streetlights_HubDist + UC_trafficpoints_HubDist, data= train.set, max.depth = 2, eta = 1, nthread = 2, nrounds = 2, objective = "binary:logistic", verbose = 1) 

print(xgb_final)

#define final model
library(xgboost)
library(caret) 			# for general data preparation and model fitting
library(rpart.plot)
library(tidyverse)

# createDataPartition() function from the caret package to split the original dataset into a training and testing set and split data into training (80%) and testing set (20%)
parts = createDataPartition(citybound$SeonsorID_, p = .8, list = F)
train = data[parts, ]
test = data[-parts, ]

#define predictor and response variables in training set
train_x = data.matrix(train[, -1])
train_y = train[,1]

#define predictor and response variables in testing set
test_x = data.matrix(test[, -1])
test_y = test[, 1]

#define final training and testing sets
xgb_train = xgb.DMatrix(data = train_x, label = train_y)
xgb_test = xgb.DMatrix(data = test_x, label = test_y)



library(caret) 
library(xgboost) 
library(tidyverse)

xgb_fit <- train(SeonsorID_ ~ bikeshop_countsNUMPOINTS + edupois_reprojected_joinNUMPOINTS + 
                   shoppoints_count_NUMPOINTS + trafficpoints_count_NUMPOINTS + trafficsignals_count_NUMPOINTS + 
                   NDVImean20 + LSTmean + slope + UC_bikeshops_HubDist + UC_edu_HubDist + UC_shops_HubDist +  
                   UC_streetlights_HubDist + UC_trafficpoints_HubDist, 
                 data = train.set.sample, 
                 method = "xgbLinear")

caret_imp <- varImp(xgb_fit)
caret_imp

plot(caret_imp)

xgb.ggplot.importance(xgb_imp)

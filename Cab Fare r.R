## CODE TESTED AND RUN ALMOST 5 TIMES BEFORE SUBMITTING. PLEASE WAIT FOR OUTPUT WHILE RUNNING 
## RANDOM FOREST FUNCTION AS R TAKES TIME TO IMPUTE AND SHOW RESULT. 



#Clear the envoirment
rm(list=ls())

#Help for exploring missing data dependencies with minimal deviation.
library(naniar)
#Calculates correlation of variables and displays the results graphically.
library(corrgram)
#Offers a powerful graphics language for creating elegant and complex plots.
library(ggplot2)
#Contains functions to streamline the model training process for complex regressions.
library(caret)
#This package includes functions and data accompanying the book "Data Mining with R.
library(DMwR)
#Recursive partitioning for classification, regression and survival trees.
library(rpart)
#For random forest
library(randomForest)

setwd("C:/Users/HP/Desktop/Pro1")

#Read or load data
train = read.csv('train_cab.csv', header = T)
test=read.csv('test.csv', header = T)

#Invalid value treated.
train$pickup_datetime[train$pickup_datetime %in% c('43')] <- NA

dim(train)

#Summary
summary(train)

#Checking type
sapply(train,class)

#We know that there is no duplicate data in our data-set.
#Converting data type of fare amount to float
train$fare_amount= as.numeric(as.character(train$fare_amount))

#Getting the missing values
missing_values = data.frame(apply(train,2,function(x){sum(is.na(x))}))
missing_values$Columns = row.names(missing_values)
row.names(missing_values) = NULL
names(missing_values)[1] =  "Missing_percentage"
missing_values$Missing_percentage = (missing_values$Missing_percentage/nrow(train)) * 100
missing_values= missing_values[order(-missing_values$Missing_percentage),]
missing_values = missing_values[,c(2,1)]
View(missing_values)


#Dropping NA values
train=na.omit(train)

#Checking shape
dim(train)

#Box plot 
#dropoff_longitude
boxplot(train$dropoff_longitude, data = train, ylab = "dropoff_longitude")

#dropoff_latitude
boxplot(train$dropoff_latitude, data = train, ylab = "dropoff_latitude")

#pickup_longitude
boxplot(train$pickup_longitude, data = train, ylab = "pickup_longitude")

#pickup_latitude
boxplot(train$pickup_latitude, data = train, ylab = "pickup_latitude")

#fare_amount
boxplot(train$fare_amount, data = train, ylab = "fare_amount")

#passenger_count
boxplot(train$passenger_count, data=train, ylab = "passenger_count")

#1.> passenger_count can't be greater than 6 and less than 1.
#2.> fare_amount can't be less than 1 and at the same time very ultra high.
#3.> fare_amount fractional entries (2) are also removed.
#4.> cordinates beyond nyc range will be dropped. 

#Removing outliers in fare_amount 
train[which(train$fare_amount < 1),] 
#Total 5 observations are found, thus dropping them (15981)
train = train[-which(train$fare_amount < 1),]

train[which(train$fare_amount > 434),] 
#Total 3 observations are found, thus dropping them (15978)
train = train[-which(train$fare_amount > 434),]
dim(train)
train[which(train$passenger_count > 6),] 
#Total 19 observations are found, thus dropping them (15959)
train = train[-which(train$passenger_count > 6),]

train[which(train$passenger_count == 0 ),] 
#Total 57 observations are found, thus dropping them (15902)
train = train[-which(train$passenger_count == 0),]

train[which(train$passenger_count == 0.12 ),] 
train[which(train$passenger_count == 1.30 ),] 
train = train[-which(train$passenger_count == 0.12),] #(15900)
train = train[-which(train$passenger_count == 1.30),]

dim(train)

#Creating boundary dataframe
boundary <- data.frame('min_lng'=-74.263242,'min_lat'=40.573143,'max_lng'=-72.986532, 'max_lat'=41.709555)
boundary$min_lat

#Now according to the boundary ranges we will detect and drop outliers (348)
train=train[-which((train$dropoff_longitude <= boundary$min_lng) | 
               (train$dropoff_longitude >= boundary$max_lng) |
               (train$dropoff_latitude <= boundary$min_lat) |
               (train$dropoff_latitude >= boundary$max_lat) |
               (train$pickup_longitude <= boundary$min_lng) |
               (train$pickup_longitude >= boundary$max_lng) |
               (train$pickup_latitude <= boundary$min_lat) |
               (train$pickup_latitude >= boundary$max_lat)),]
dim(train)

#Extracting a new feature that is travel distance.
train$date = as.Date(as.character(train$pickup_datetime)) # Converting column to datetime format
train$day_of_week = as.factor(format(train$date,"%u"))# Monday = 1, Tuesday = 2 and so on
train$month = as.factor(format(train$date,"%m"))
train$year = as.factor(format(train$date,"%Y"))
pickup_time = strptime(train$pickup_datetime,"%Y-%m-%d %H:%M:%S")
train$hour = as.factor(format(pickup_time,"%H"))

#Now we don't have missing values in our data-set.

test$date = as.Date(as.character(test$pickup_datetime)) # Converting column to datetime format
test$day_of_week = as.factor(format(test$date,"%u"))# Monday = 1, Tuesday = 2 and so on
test$month = as.factor(format(test$date,"%m"))
test$year = as.factor(format(test$date,"%Y"))
pickup_time = strptime(test$pickup_datetime,"%Y-%m-%d %H:%M:%S")
test$hour = as.factor(format(pickup_time,"%H"))

#Feature engineering

#Let's extract the distance in kms this time  as we have extracted distance in miles in python.

#Function to convert degrees to Radians.
deg_to_rad = function(deg){
  (deg * pi) / 180
}

#Function to calculate Haversine distance.
extract_d = function(lat1, long1, lat2, long2){
phi1 = deg_to_rad(lat1)
phi2 = deg_to_rad(lat2)
delphi = deg_to_rad(lat2 - lat1)
dellamda = deg_to_rad(long2 - long1)
a = sin(delphi/2) * sin(delphi/2) + cos(phi1) * cos(phi2) * 
sin(dellamda/2) * sin(dellamda/2)
c = 2 * atan2(sqrt(a),sqrt(1-a))
km = 6371 * c
return(km)
}

train$distance = extract_d(train$pickup_latitude, train$pickup_longitude, train$dropoff_latitude, train$dropoff_longitude)

test$distance = extract_d(test$pickup_latitude, test$pickup_longitude, test$dropoff_latitude, test$dropoff_longitude)

#Feature Selection
train = subset(train,select = -c(pickup_longitude,pickup_latitude,dropoff_longitude,dropoff_latitude,pickup_datetime,date))
test = subset(test,select = -c(pickup_longitude,pickup_latitude,dropoff_longitude,dropoff_latitude,pickup_datetime,date))

#Correlation analysis 
#Selecting relevant numeric variables
numeric_index = sapply(train,is.numeric)
numeric_data = train[,numeric_index]
cnames = colnames(numeric_data)
corrgram(train[,numeric_index],upper.panel=panel.pie, main = "Correlation Plotting")

#Plot between far amount and passenger count
ggplot(train, aes_string(x=train$passenger_count, y=train$fare_amount))+
  geom_bar(stat="summary",fill =  "DarkSlateBlue") + theme_bw() +
  xlab('passenger_count')+ylab("fare")+  
  theme(text=element_text(size=15))

#count plot for passengers
ggplot(train, aes_string(x = train$passenger_count)) +
  geom_bar(stat="count",fill =  "DarkSlateBlue") + theme_bw() +
  xlab("no of passengers") + ylab('Count')

#histogram of different variables

#fare amount
ggplot(train, aes_string(x = train$fare_amount)) + 
  geom_histogram(fill="cornsilk", colour = "black") + geom_density() +
  theme_bw() + xlab("fare_amount") + ylab("Frequency")



#distance
ggplot(train, aes_string(x = train$distance)) + 
  geom_histogram(fill="cornsilk", colour = "black") + geom_density() +
  theme_bw() + xlab("distance") + ylab("Frequency")

#Scatter plot between distance and fare
ggplot(train, aes(x = fare_amount, y =distance))+geom_point()

#Creating backup
backup_1 = train
fare_amount = train$fare_amount

#Feature scaling usigng log 
train$fare_amount = log1p(train$fare_amount)
train$distance = log1p(train$distance)
test$distance = log1p(test$distance)

#Splitting
#To produce the same result for different instances.
set.seed(17)
split_index = createDataPartition(train$fare_amount,p=0.80,list = FALSE)
train_data = train[split_index,]
test_data = train[-split_index,]


#Linear Regression
model_LR = lm(fare_amount ~., data=train_data)
predictions_LR = predict(model_LR,test_data[,2:7])
error_metrices_LR = regr.eval(expm1(test_data[,1]),expm1(predictions_LR))
print(paste('Error percentage of the model is: ', error_metrices_LR['mape']*100))
print(paste('The model is ', (1-error_metrices_LR['mape'])*100, ' percentage accurate.'))
result_LR = data.frame(expm1(test_data[,1]))
result_LR$Predicted = expm1(predictions_LR)
names(result_LR)[1] = 'Actual'
View(result_LR)

test_predictions_LR = predict(model_LR, test)
testResult_LR = data.frame(test)
testResult_LR$fare_amount = expm1(test_predictions_LR)
View(testResult_LR, title = 'Linear Regression Outcome')


#Decison Tree
model_DT = rpart(fare_amount ~ ., data = train_data, method = "anova")
predictions_DT = predict(model_DT, test_data[,2:7])
error_metrices_DT = regr.eval(expm1(test_data[,1]),expm1(predictions_DT))
print(paste('Error Percentage of the model is: ', error_metrices_DT['mape']*100))
print(paste('The model is ', (1-error_metrices_DT['mape'])*100, ' percentage accurate.'))
result_DT = data.frame(expm1(test_data[,1]))
result_DT$Predicted = expm1(predictions_DT)
names(result_DT)[1] = 'Actual'
View(result_DT)

test_predictions_DT = predict(model_DT, test)
testResult_DT = data.frame(test)
testResult_DT$fare_amount = expm1(test_predictions_DT)
View(testResult_DT, title = 'Decision Tree Outcome')


#Random Forest
model_RF = randomForest(fare_amount ~.,data=train_data)
predictions_RF = predict(model_RF,test_data[,2:7])
error_metrices_RF = regr.eval(expm1(test_data[,1]),expm1(predictions_RF))
print(paste('Error Percentage of the model is: ', error_metrices_RF['mape']*100))
print(paste('The model is ', (1-error_metrices_RF['mape'])*100, ' percentage accurate.'))
result_RF = data.frame(expm1(test_data[,1]))
result_RF$Predicted = expm1(predictions_RF)
names(result_RF)[1] = 'Actual'
View(result_RF)

test_predictions_RF = predict(model_RF, test)
testResult_RF = data.frame(test)
testResult_RF$fare_amount = expm1(test_predictions_RF)
View(testResult_RF, title = 'Random Forest Outcome')


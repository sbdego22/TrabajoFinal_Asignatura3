## 2.2 Modelo para detectar si un usuario controla las calorías ingeridas

# El objetivo de esta parte del trabajo final de "Técnicas de análisis y 
# explotación de datos" consiste en realizar un modelo predictivo de clasificación 
# binaria para detectar si un usuario controla o no las calorías ingeridas.
# El análisis se apoyará sobre las mismas variables utilizadas previamente.

#Carga de librerias necesarias
install.packages (c("dplyr", "caret", "ggplot2", "e1071", "DMwR", "ROCR", "pROC", "rpart", "randomForest"))
install.packages("DMwR2")
library (dplyr)
library (caret)
library (ggplot2)
library (e1071)
library (DMwR)
library (ROCR)
library (pROC)
library (rpart)
library (randomForest)
library (DMwR2)

#Carga y lectura de los datos
dataset <- read.csv ("/Users/RMF/Desktop/Máster en Big Data/3. Técnicas de análisis y exploración de datos/3.15 Trabajo final/obesity_data_set_preprocessed.csv", sep =";")

str(dataset)
head(dataset)


#Cambio de formato para convertir la coma en punto y transformar la columna a numérico
dataset$Height <- as.numeric(gsub(",", ".", dataset$Height))
dataset$BMI <- as.numeric(gsub(",", ".", dataset$BMI))

str(dataset)
summary(dataset)

# Convertimos la variable objetivo --Calory_Monitoring-- a factor para asegurarnos que R entiende que se trata de un problema de clasificación
dataset$Calory_Monitoring <- as.factor(dataset$Calory_Monitoring)

## Dividimos los datos para su entrenamiento (80%) y prueba (20%). La variable objetivo será Calory_Monitoring.
set.seed(42)  # Semilla para reproducibilidad

train_index <- createDataPartition(dataset$Calory_Monitoring, p = 0.8, list = FALSE)

train_data <- dataset[train_index, ]  # Datos para entrenamiento
test_data <- dataset[-train_index, ]  # Datos para prueba

dim(train_data)
dim(test_data)

## Utilizamos un modelo de regresión logística para predecir si la persona controla las calorías. 
#Utilizamos todas las variables del dataset como predictoras para cruzarlas con nuestra variable objetivo

model <- glm(
  Calory_Monitoring ~ .,       # Variable objetivo y todas las predictoras
  family = binomial(link = 'logit'), 
  data = train_data
)
summary (model)
str(dataset)
print(model)

#-------- Resultado del sumario

#Call:
  glm(formula = Calory_Monitoring ~ ., family = binomial(link = "logit"), 
      data = train_data)

#Coefficients: (1 not defined because of singularities)
#                                 Estimate Std. Error z value Pr(>|z|)    
#(Intercept)                      9.401547  10.746245   0.875  0.38165    
#Gender                          -0.569800   0.401061  -1.421  0.15540    
#Age                             -0.098177   0.037581  -2.612  0.00899 ** 
#Height                          -4.133746   6.133513  -0.674  0.50034    
#Weight                           0.009812   0.074567   0.132  0.89531    
#BMI                             -0.213259   0.251981  -0.846  0.39737    
#Weight_Class_Mendoza_De_La_Hoz  -1.114348   0.496493  -2.244  0.02480 *  
#Weight_Class_WHO                 1.904672   0.611329   3.116  0.00184 ** 
#Overweight_Family               -0.326230   0.327938  -0.995  0.31984    
#High_Caloric_Consumption        -1.394100   0.305883  -4.558 5.17e-06 ***
#Vegetable_Consumption            0.873339   0.321004   2.721  0.00652 ** 
#Main_Meals                      -0.151927   0.153424  -0.990  0.32205    
#Between_Meals                    0.169100   0.226572   0.746  0.45546    
#Smoker                           0.844711   0.757474   1.115  0.26478    
#Water_Consumption                0.397284   0.210588   1.887  0.05922 .  
#Physical_Activity                0.292590   0.153898   1.901  0.05728 .  
#Tech_Devices_Use                -0.123392   0.209579  -0.589  0.55602    
#Alcohol_Consumption              0.119663   0.252924   0.473  0.63613    
#Transport_Mean_Automobile       -0.139356   0.696788  -0.200  0.84148    
#Transport_Mean_Motorbike         1.601208   1.059856   1.511  0.13084    
#Transport_Mean_Bike              1.390710   1.326045   1.049  0.29429    
#Transport_Mean_Public_Transport -0.460891   0.566714  -0.813  0.41606    
#Transport_Mean_Foot                    NA         NA      NA       NA    
---
#  Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

#(Dispersion parameter for binomial family taken to be 1)

#Null deviance: 601.45  on 1688  degrees of freedom
#Residual deviance: 441.28  on 1667  degrees of freedom
#AIC: 485.28

#Number of Fisher Scoring iterations: 7

#De estos resultados se deduce a priori que las variables #High_Caloric_Consumption,
#Vegetable_Consumption, Weight_Class_WHO y Age tienen un impacto significatvo a la hora
#de predecir si se controla la ingesta de calorías. Todos ellos tiene p-valores muy bajos,
# (<0.01)
  
#Ajustamos el modelo incluyendo solo las variables significativas para evotar problemas. 
  model <- glm(
    Calory_Monitoring ~ Age + Weight_Class_Mendoza_De_La_Hoz + 
      Weight_Class_WHO + High_Caloric_Consumption + Vegetable_Consumption, 
    family = binomial(link = 'logit'), 
    data = train_data
  )
  

# Hacemos las predicciones en el conjunto de prueba para comprobar si nuestro modelo
# está generalizando bien los patrones aprendidos. 
predictions <- predict(model, newdata = dataset[-train_index, ], type = "response")
  
# Convertimos probabilidades en clases binarias (umbral = 0.5)
predicted_classes <- ifelse(predictions > 0.5, 1, 0)

head(predictions)
head(predicted_classes)

#Análisis del rendimiento de nuestro modelo mediante el cálculo de métricas
#como la matriz de confusión, precisión, recall y FI-Score para ver como se comporta
#en todo el conjunto (o datos) de prueba

#Cálculo de la precisión o accuracy
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
cat("Precisión del modelo:", round(accuracy, 4), "\n")

#Interpretación del resultado: la precisión arroja un valor de 0.955, lo que significa
#que significa que el modelo clasifica correctamente el 95% de las observaciones
#en el conjunto de prueba, lo cual es un buen desempeño a falta de evaluar otras métricas.


#Matriz de confusión
confusion_matrix <- table(Predicted = predicted_classes, Actual = dataset$Calory_Monitoring[-train_index])
print(confusion_matrix)

#Interpretación del resultado: el modelo es capaz de predecir con un elevado grado de precisión (403 de un total de 422 observaciones en el conjunto de prueba) los 'true negatives',
#en este caso, aquellos que no controlan las calorías (0), pero arroja 19 falsos negativos y no es capaz de predecir nunca la clase 1, 
#aquellos que controlan las calorías. Podría deberse a un umbral de clasificación demasiado alto
#o a un sesgo del modelo hacia la clase 0. 

#Matriz de confusión con el umbral rebajado a 0.4
predicted_classes <- ifelse(predictions > 0.4, 1, 0)
confusion_matrix <- table(Predicted = predicted_classes, Actual = dataset$Calory_Monitoring[-train_index])
print(confusion_matrix)

#¿Existe un desbalance de clases con la subrepresentación de la clase 1?
table(dataset$Calory_Monitoring)

#Recall
TP <- confusion_matrix["1", "1"]  # True Positives
FN <- confusion_matrix["0", "1"]  # False Negatives
recall <- TP / (TP + FN)
cat("Recall:", round(recall, 4), "\n")

#El recall arroja un valor de 0, lo que significa que el modelo no es capaz de capturar 
# correctamente la clase 1, aquellos que controlan las calorías. 

#RandomForest. Dado que nuestro modelo de regresión logística sigue dando problemas, 
#probaremos con un modelo más robusto frente al balance de clases. 
library(randomForest)
rf_model <- randomForest(Calory_Monitoring ~ ., data = train_data, ntree = 100)
rf_predictions <- predict(rf_model, test_data)
rf_confusion <- table(Predicted = rf_predictions, Actual = dataset$Calory_Monitoring[-train_index])
print(rf_confusion)

#Matriz de confusión con RandomForest

           Actual
#Predicted  0   1
#       0  403  12
#       1   0   7
           
#Cálculo del Accuracy con el nuevo modelo RForest
accuracy <- (TP + TN) / sum(confusion_matrix)
cat("Precisión (Accuracy):", round(accuracy, 4), "\n")

#Cálculo del Recall
recall <- TP / (TP + FN)
cat("Recall:", round(recall, 4), "\n")

#Cálculo de la precisión 
precision <- TP / (TP + FP)
cat("Precision:", round(precision, 4), "\n")

#Cálculo del F1-Score
f1_score <- 2 * (precision * recall) / (precision + recall)
cat("F1-Score:", round(f1_score, 4), "\n")

#CONCLUSIONES. De los resultados obtenidos al calcular las métricas clave para analizar este modelo predictivo
#se deduce que es modelo es mejorable. Sus fortalezas estriban en que es capaz 
#de clasificar con una alta precisión el total de las observaciones (Accuracy: 97.16%) 
# y nunca se equivoca a la hora de predecir quién controla las calorías (Precision:100%). Por el contrario,
#tiene también serias deficiencias. Su bajo Recall (36.8%) indica que solo identifica correctamente
#cuatro de cada diez casos reales de la clase 1, o lo que es lo mismo, de aquellas personas que sí controlan las calorías.
#De todo ello ser deriva también un sesgo hacia la clase mayoritaria o clase 0, esa 
#mayoría que no controla la ingesta de calorías. 

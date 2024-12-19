data <- read.csv("/Users/merche/Trabajo grupal/ObesityDataSet_raw_and_data_sinthetic.csv")

# Convertimos FAVC ("yes"/"no") a valores numéricos (1/0) 
data$FAVC <- ifelse(data$FAVC == "yes", 1, 0)

#Vamos a crear una columna nueva que será el BMI (Body Mass Index). Es importante porque relaciona peso y altura y es comunmente usada para analizar patrones con la salud. A parte de eso, simpifica el clustering al combinar altura y peso en una variable
data$BMI <- data$Weight / (data$Height^2)

#Seleccionamos las variables para el clustering: FAVC (consumo de comida calórica), FCVC (consumo de vegetales) y BMI. Creamos un dataset con estas columnas para simplificar el análisis
data_clustering <- data[, c("FAVC", "FCVC", "BMI")]

#Verificamos el set dentro del set creado:
head(data_clustering)

#Determinaremos el número de clusters con el Método del codo
set.seed(101)

#Ejecutamos k-means con un número arbitrario de clusters (3)
clustering_result<-kmeans(data_clustering, center=3, nstart=20)
#Lo visualizamos
clustering_result

#Evaluamos el clustering creando una tabla para analizar cómo se relacionan los clusters con una variable conocida
table(clustering_result$cluster,data$NObeyesdad)

#Evaluamos si 3 es un buen número de clusters o si necesitamos ajustar 
tot.withinss <- vector(mode = "numeric", length = 10)
for (i in 1:10) {
  kmeans_test <- kmeans(data_clustering, centers = i, nstart = 20)
  tot.withinss[i] <- kmeans_test$tot.withinss
}

# Graficamos el Método del Codo
plot(1:10, tot.withinss, type = "b", pch = 19, 
     main = "ELBOW GRAPH")
#Observamos que 3 es un buen número pues después del 3 la curva se vuelve más plana por lo que aumentar el número de clusters no aportará una mejora significativa.

#Visualizamos los clusters
library(cluster)
clusplot(data_clustering, clustering_result$cluster, color=TRUE, shade=TRUE, labels=0, lines=0, main = "CLUSTERING PLOT")
#Observamos que hay claramente 3 grupos: rojo, rosa y azul. Se obseerva una superposición entre los clusters, principalmente el rojo y el rosa lo que nos quiere decir que los grupos no están perfectamente diferenciados y que puede haber patrones de comportamiento similares entre individuos de estos clusters.

#Visualizamos el Coeficiente de Silueta 
library(factoextra)
fviz_silhouette(silhouette(clustering_result$cluster, dist(data_clustering)), main = "SILHOUETTE")

#La mayoría de los puntos tienen valores mayores a 0.5, lo cual sugiere que el custering es aceptable. El cluster verde tiene los valores de silueta más altos indicando que los puntos están mejor definidos. El rojo presenta mayor variabilidad y el azul muestra una forma regular con valores medios.

# Calculamos la matriz de distancias usando el método "canberra"
dist_matrix <- dist(data_clustering, method = "canberra")

# Aplicamos el clustering jerárquico con el método "complete"
hclust_result <- hclust(dist_matrix, method = "complete")

# Visualizamos el dendrograma
plot(hclust_result, hang = -1, cex = 0.6, main = "DENDROGRAM: Canberra Distance")

#Procedemos a cortar el dendrograma:
plot(hclust_result, hang = -1, cex = 0.6, main = "DENDROGRAM: Canberra Distance")
abline(h = 1.5, col = "red", lty = 2) 

#Observamos claramente 3 clusters gracias al corte.Las ramas largas indican clusters bien separados y las cortas en la parte inferior muestran grupos muy pequeños y similares.

#Conclusiones generales
#Silhouette Plot:
  #Proporciona información sobre la cohesión y separación de los clusters generados por K-means. Identifica qué clusters son más compactos y cuáles tienen mayor variabilidad.
#Dendrograma (Canberra Distance):
  #Proviene del clustering jerárquico y muestra la estructura de los datos en forma de árbol. Nos permite identificar 3 clusters principales al cortar a la altura de 1.5.
#Clustering Plot:
  #Este gráfico proyecta los clusters de K-means en dos componentes principales:
  #Muestra cómo se distribuyen los puntos.
  #Expone la superposición entre los clusters rojo y azul, lo que sugiere similitudes en las variables utilizadas (BMI, FAVC y FCVC).

#Cómo se Relacionan los Tres Gráficos
#Silhouette Plot:
  #Nos dice que el cluster verde es el más cohesivo (mejor agrupado).
  #Los clusters rojo y azul tienen mayor variabilidad, lo que indica posibles solapamientos.
#Dendrograma:
  #Muestra la estructura jerárquica de los datos.
  #Al cortar a 1.5, se observan claramente 3 clusters principales, lo cual confirma el número de clusters sugerido por el Método del Codo en K-means.
#Clustering Plot:
  #Visualmente, muestra los clusters rojo, verde y azul proyectados en dos dimensiones.
  #La superposición entre rojo y azul refuerza que estas dos agrupaciones tienen características similares.

#Conclusión General Integrada
#Los tres gráficos se complementan y nos permiten llegar a una conclusión más sólida:
  #Cluster Verde:Representa a las personas con mejores hábitos (alto consumo de vegetales, menor BMI, menor consumo calórico).
   #Confirmado por la alta cohesión en el Silhouette Plot y la visualización clara en el Clustering Plot.
  #Clusters Rojo y Azul:Representan a las personas con mayor riesgo de obesidad o sobrepeso:
   #FAVC frecuente (alta ingesta calórica).
   #FCVC bajo a moderado (bajo consumo de vegetales).
   #BMI elevado.
   #Estos grupos tienen superposición, como se ve en el Clustering Plot, lo que indica similitudes en sus comportamientos.
  #Consistencia del Número de Clusters:Tanto el Silhouette Plot como el Dendrograma confirman que 3 clusters es una segmentación adecuada.


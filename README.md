# Street Fighter Detector

## 🎯 Objetivo del trabajo

Desarrollar un sistema de reconocimiento automático de personajes del videojuego *Street Fighter* a partir de imágenes estáticas o en tiempo real, utilizando técnicas de procesamiento de imágenes y aprendizaje profundo. El proyecto compara dos enfoques: uno clásico basado en extracción manual de características y clasificación supervisada, y otro basado en redes neuronales convolucionales (CNN), con el fin de evaluar su precisión, eficiencia y capacidad de generalización.

## 👥 Integrantes del grupo

- Rafael Tomas Chui Sanchez – U201925837  
- Rodrigo Alejandro Meza Polo – U202224016  
- Axel Yamir Pariona Rojas – U202222148  
- Liam Mikael Quino Neff – U20221e167  

## 🗂️ Descripción del dataset

El conjunto de datos utilizado está organizado en una carpeta denominada `street_fighter_dataset`, que contiene 15 subcarpetas, cada una correspondiente a un personaje distinto del videojuego *Street Fighter*. Cada subcarpeta incluye al menos 100 imágenes en formato `.jpg`, con un total aproximado de 1,758 imágenes. Las imágenes presentan variabilidad en resolución, poses, ángulos y fondos, lo que permite entrenar modelos robustos. El dataset fue recolectado manualmente desde fuentes públicas como wikis, sitios oficiales y material generado por la comunidad. Se realizó una limpieza manual para eliminar duplicados, imágenes corruptas o irrelevantes.

## ✅ Conclusiones

El modelo basado en redes neuronales convolucionales (CNN) superó ampliamente a los enfoques clásicos (SVM y KNN), alcanzando una precisión del 80.11% frente al 26% de los modelos tradicionales. Esta diferencia se debe a la capacidad de las CNN para aprender representaciones jerárquicas directamente desde los datos, lo que les permite adaptarse mejor a la variabilidad visual del dataset. Además, el sistema fue validado en una interfaz gráfica con imágenes en tiempo real, manteniendo niveles de confianza superiores al 90%. Como trabajo futuro, se propone ampliar el dataset, optimizar la arquitectura del modelo y desarrollar una versión móvil de la aplicación.

## 📄 Licencia

Este proyecto se distribuye bajo la licencia MIT. Puede usar, modificar y distribuir el código con fines académicos o personales, siempre que se otorgue el crédito correspondiente a los autores.


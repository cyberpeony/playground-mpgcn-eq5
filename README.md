# MP-GCN Playground Pipeline — Equipo 5

Este repositorio contiene el pipeline completo para clasificar actividades humanas en un parque infantil utilizando un modelo **MP-GCN (Multi-Plane Graph Convolutional Network)**.  
El sistema procesa exclusivamente datos estructurados (poses, bounding boxes y relaciones humano–objeto) para predecir una de tres clases de actividad.

---

## 0. Documentación
Nota: Toda la documentación formal del proyecto se encuentra en la carpeta docs/:
- docs/Reporte_Final_MPGCN_Playground.pdf — Reporte técnico completo con metodología, arquitectura, resultados y conclusiones.
- docs/Presentacion_Final_E5.pdf — Presentación utilizada en la evaluación final del proyecto.

---

## 1. Objetivo del Proyecto

Clasificar cada segmento de video panorámico en una de las siguientes categorías:

1. **Transit** — Personas caminando o desplazándose sin interacción relevante.  
2. **Social_People** — Interacciones sociales entre personas.  
3. **Play_Object_Normal** — Actividades de juego que involucran objetos.

El modelo trabaja sin pixeles, empleando un grafo humano–objeto por frame y un MP-GCN multi-stream para capturar información temporal.

---

## 2. Arquitectura: Multi-Stream MP-GCN

El sistema usa **4 streams**:

- **J** — Coordenadas 2D de joints  
- **B** — Bounding boxes  
- **JM** — Motion de joints  
- **BM** — Motion de bounding boxes  

Cada muestra tiene dimensiones: [C=2, T=48, V=32 nodos, M=4 personas]


Los nodos del grafo incluyen:

- 17 joints humanos  
- Hasta 15 objetos  
- Aristas intra-humano, intra-objeto y humano–objeto  

Las matrices de adyacencia se construyen automáticamente con: A0, A_intra, A_inter → [3, 32, 32]


---

## 3. Dataset y Splits

Los splits se generan con: python build_train_val_csv.py


Distribución final:

| Clase               | Total | Train | Val | Test |
|--------------------|-------|-------|-----|------|
| Transit            | 71    | 50    | 11  | 10   |
| Social_People      | 71    | 50    | 11  | 10   |
| Play_Object_Normal | 70    | 49    | 10  | 11   |

Total test: **31 muestras**

Los archivos `.npy` con el panorama vectorizado se almacenan en `data/panoramic_npy/` (**no versionados en Git**).

---

## 4. Entrenamiento

Entrenamiento del modelo: python video_processing/train_mpgcn.py


Parámetros principales:

- Épocas: 40  
- Batch size: 4  
- Optimizador: Adam (lr=1e-3, weight_decay=1e-4)  
- Seed: 7  
- `use_att=False` (atención desactivada)

### Mejor resultado en validación
Validation Accuracy = 71.9 %


Checkpoint guardado en: checkpoints/mpgcn_playground_best.pth


---

## 5. Evaluación en Test

Ejecutar: python video_processing/eval_mpgcn.py


Resultado global: Test Accuracy = 48.39 %


---

## 6. Métricas Detalladas

Generar métricas: python video_processing/analyze_test_results.py


### Accuracy por clase

| Clase               | Accuracy |
|--------------------|----------|
| Transit            | 40.00%   |
| Social_People      | 30.00%   |
| Play_Object_Normal | 72.73%   |

### Classification Report
precision recall f1-score support
Transit 0.4444 0.4000 0.4211 10
Social_People 0.3333 0.3000 0.3158 10
Play_Object_Normal 0.6154 0.7273 0.6667 11
accuracy 0.4839 31
macro avg 0.4644 0.4758 0.4678 31
weighted avg 0.4693 0.4839 0.4743 31


### Matriz de confusión
[[4 5 1]
[3 3 4]
[2 1 8]]


Imagen generada: `confusion_matrix_test.png`

---

## 7. Interpretación de Resultados

- El modelo es más fuerte en **Play_Object_Normal**, donde existen objetos que actúan como nodos adicionales y facilitan la discriminación.
- Las clases **Transit** y **Social_People** se confunden entre sí, lo cual es coherente: comparten patrones cinemáticos muy similares y carecen de objetos distintivos.
- La diferencia entre validación (71.9%) y test (48.39%) es un indicio de overfitting esperado por el tamaño reducido del dataset.
- Con más datos sociales y de tránsito, el rendimiento debería aumentar significativamente.

---

## 8. Cómo Ejecutar Todo el Pipeline

### 1. Construir splits
python build_train_val_csv.py


### 2. Entrenar MP-GCN
python video_processing/train_mpgcn.py


### 3. Evaluar en test
python video_processing/eval_mpgcn.py


### 4. Generar métricas y gráficas
python video_processing/analyze_test_results.py


---

## 9. `.gitignore` Recomendado (incluido)

Ignora:

- `data/panoramic_npy/*.npy`
- `data/raw_videos/*.mp4`
- `checkpoints/`
- `*.pt`, `*.pth`
- Entornos virtuales
- Archivos temporales

---

## 10. Trabajo a mejorar 

- Activar atención en MP-GCN para modelar relaciones espaciales finas  
- Añadir data augmentation para trayectorias y cajas  
- Incorporar distancias interpersonales como features adicionales  
- Expandir el dataset para reducir la variancia del modelo  

---

## 11. Correcciones realizadas según retroalimentación de los profesores
Durante el desarrollo del proyecto recibimos retroalimentación en distintas actividades intermedias correspondientes a los módulos del bloque. A continuación se describen las correcciones aplicadas al pipeline y al repositorio en base a dichos comentarios.

### Correcciones del módulo: Big Data

Retroalimentación:
- El dataset original tenía desequilibrio fuerte entre clases, con muchas escenas de Transit y Social, pero muy pocas de otras categorías.
- Algunas clases eran tan pequeñas que no permitían un entrenamiento útil.
- Se pidió revisar la viabilidad del dataset y plantear una estrategia de reducción o reagrupación.

Correcciones aplicadas:
- Se redujo el problema a 3 clases balanceables: Transit, Social_People, Play_Object_Normal.
- Se descartaron clases con muy pocas muestras (<20).
- Se reconstruyó el dataset final solo con escenas limpias y utilizables, resultando en ~213 videos válidos.
- Se generaron ventanas uniformes de 48 frames para homogenizar las muestras.
---

### Correcciones del módulo: Arquitecturas de Deep Learning

Retroalimentación recibida:
- Faltaba justificar la elección de MP-GCN.
- Nodos y aristas mal definidos en la versión inicial.
- Faltaba explicación de los streams J, B, JM, BM.

Correcciones aplicadas:
- Se formalizó la definición del grafo panorámico con 32 nodos: 17 joints + ~15 objetos.
- Se corrigió y documentó la creación de:
- A₀ (intra-persona)
- A_intra (persona–objeto)
- A_inter (persona–persona)
- Se añadieron ecuaciones claras en el reporte y las slides.
- Se explicó correctamente la forma tensorial: [C, T, V, M].
---

### Correcciones del módulo: Estadística Avanzada

Retroalimentación recibida:
- Faltaban métricas completas y análisis del modelo.
- No se había generado matriz de confusión.

Correcciones aplicadas:
- Se generó classification report, incluyendo precision, recall y F1.
- Se añadió una matriz de confusión interpretada por clase.
- Se explicó la brecha entre validación (71%) y test (48%), asociándola al tamaño reducido del test set y al riesgo de overfitting.
---

### Correcciones del módulo: Cómputo en la Nube

Retroalimentación recibida:
- Faltaba reproducibilidad en el pipeline.
- .gitignore incompleto y mezcla de datos + código.

Correcciones aplicadas:
- Scripts ejecutables para cada fase:
build_train_val_csv.py, train_mpgcn.py, eval_mpgcn.py, analyze_test_results.py.
- .gitignore actualizado para excluir videos, .npy y checkpoints.
- Reorganización del repositorio en carpetas limpias.

---

### Correcciones del módulo: Reto (Metodología y Gestión)

Retroalimentación recibida:
- Pipeline poco claro para el socio formador.
- Presentación y reporte sin consistencia con la implementación.

Correcciones aplicadas:
- Reescritura completa de la metodología:
etiquetado → esqueletos → objetos → grafo → tensores → splits → MP-GCN
- Alineación entre presentación, pipeline del repo y reporte en LaTeX.
- Inclusión de la contribución individual en conclusiones del reporte.

---

### Correcciones del módulo: Técnicas y Arquitecturas de Deep Learning (Refinamiento)

Retroalimentación recibida:
- Se necesitaba refinar el modelo después de la primera iteración.
- El profesor pidió probar otras configuraciones o variantes, no solo el benchmark.
- Validar si activar módulos opcionales (como atención) mejoraba el rendimiento.

Correcciones aplicadas:
- Se reentrenó el modelo ajustando hiperparámetros: LR, batch size, weight decay.
- Se probó entrenamiento con y sin módulo de atención (use_att=True/False).
- Se evaluó impacto de normalización corregida y nueva construcción de grafos.
- Se compararon distintas semillas para confirmar estabilidad del resultado.
- Se documentaron los efectos:
- Mejor rendimiento en validación (hasta 71.9%)
- Reducción de pérdida durante entrenamiento
- Mejor comportamiento en Play_Object_Normal
- Se agregó una sección de “Trabajo a futuro” con arquitecturas alternativas sugeridas:
- ST-GCN
- AGCN
- MP-GCN con atención activada
- Multistream con features adicionales

---

## 12. Autores

Este proyecto fue desarrollado por:
- **Carlos Alberto Mentado Reyes** — A01276065  
- **Fernanda Díaz Gutiérrez** — A01639572  
- **José Eduardo Puentes Martínez** — A01733177  
- **Raymundo Iván Díaz Alejandre** — A01735644  

Tecnológico de Monterrey  
Curso: IA Avanzada











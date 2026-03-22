---
class: uni/nota
fecha: 2026-02-09
hora: 12:54
asignatura: Proyecto Integrador
tags:
  - ai/models
---
![[SAM3_arquitectura.png|Arquitectura de SAM3|700x338]]

SAM3 es un modelo [^1]fundacional de segmentación que unifica **detección, segmentación y tracking** en imágenes y vídeo, y que puede segmentar todas las instancias de un concepto a partir de prompts de concepto (frases cortas tipo *"yellow school bus"*, ejemplos de imagen, o ambos) mediante **Promptable Concept Segmentation (PCS)**.

[^1]: **“fundacional” (foundation model)** significa que es un modelo “base”:
	
	- **Entrenado con datos muy amplios y variados, a gran escala** (normalmente con _self-supervision_),
	    
	- que luego **se puede adaptar** (fine-tuning, prompting, etc.) a **muchas tareas distintas** en vez de estar hecho solo para una tarea concreta.

En SAM3 hay varios tipos de prompts que están contemplados:
- Prompt de texto -> "yellow school bus"
- Prompt de imagen -> *\*foto de un bus escolar amarillo\**
- Prompt geométrico -> región de la imagen donde hay un bus escolar amarillo

En SAM3 los prompts son usado para que a partir de una imagen de input
#### Arquitectura

La explicación de la arquitectura la voy a ir diseccionando por partes siguiendo el orden de procesamiento que sigue la red.
##### Perception Encoder

![[SAM3_arquitectura_perception_encoder.png]]
Como ya hemos dicho, SAM3 soporta prompts de imagen, texto y geometría
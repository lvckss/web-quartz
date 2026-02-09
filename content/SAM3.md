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

#### Arquitectura

. . .
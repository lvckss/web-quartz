---
class: uni/nota
fecha: 2026-02-08
hora: 11:33
asignatura: Proyecto Integrador
tags:
  - ai
  - pit
title: Entendiendo CU6
---
**Documentación aportada por Plexus Tech hasta ahora:**
- [presentación](https://cv.usc.es/pluginfile.php/3107602/mod_resource/content/1/Caso_Uso_IA_Plexus.pdf)
- [CU6](https://lvcks.s-ul.eu/8Q0Jhs9t)

>Me gustaría poder redactar sobre el tema con una mayor noción de estructura y orden pero por falta de tiempo y necesidad de yo mismo entender como abordar este proyecto, todo esto va a estar escrito como la *mierda*.

---

Plexus Tech es básicamente una empresa que ofrece soluciones digitales a **grandes** clientes (`$$`). Con respecto a su competencia, es probablemente una de las empresas del país que ofrece soluciones tecnológicas usando las herramientas/modelos/técnicas más *cutting-edge* o innovadoras lo cual vemos manifestado en nuestro proyecto:

`Evaluación automatizada de daños en vehículos`

Esto es un problema con un amplio interés alrededor de todo el mundo y al que se le ha empezado a dar mucha atención desde hace unos años (2022-2023).

Lo que se quiere es: dada una imagen de un vehículo, tener un modelo que sea capaz de **segmentar** (aislar en la imagen), regiones en las que se detecten abolladuras, fisuras, rasguños, grietas... para después mandar esos resultados a un modelo de lenguaje con capacidad de visión que pueda añadir datos como descripción del daño, ubicación del daño, etc.

![[image.png|Labels de imágenes en CarDD donde se ve la segmentación que se realiza junto a la clase de daño clasificada|700x424]]

Por suerte, el surtido de datos que nos aportan es bastante variado. Nos ofrecen un subconjunto con datos procedentes del dataset **CarDD** (imagen de arriba).
- 4000 imágenes
- 6 tipos de daño: **dent** (abolladura), **scratch** (arañazo), **crack** (grieta), **glass shatter** (cristal roto), **lamp broken** (faro roto), **tire flat** (rueda pinchada / desinflada).

*Ellos han ofrecido este subconjunto de datos por simplificar la tarea, pero podemos pillar más datos de diversas fuentes como RoboFlow, VehiDE...*

Lo que nos pide Plexus es (y cito):

> **Solución esperada**: desarrollar un sistema de IA capaz de `(i) procesar imágenes en 2-3 minutos (vs 2-3 horas)`, que `(ii) funcione en cualquier condición ambiental (lluvia, noche, nieve, sombras => ojo)`, que `(iii) detecte daños con una precisión superior al 95%`, que `(iv) genere descripciones automatizadas` y que `(v) se adapte a nuevos escenarios sin reentrenamiento`. Se espera ==**explicabilidad**== y métricas como entregable conjunto a la solución, a modo de documentación técnica a posteriori.

Para esto quieren que usemos modelos como [[SAM3]] (Segment Anything Model) o **Grounding DINO**.

![[SAM3_arquitectura.png|Arquitectura de SAM3|700x338]]

SAM3 es actualmente el modelo SOTA (State Of The Art o estado del arte) para segmentación de objetos [[[Modelos Zero-shot|zero-shot]]. Esto implica que no va a ser necesario (para nuestro caso) un re-entrenamiento del modelo con nueva información, ya que de por sí es muy bueno generalizando debido a que en su entrenamiento original se usaron cantidades **masivas** de datos.

---

Entonces, para que nos hagamos una idea; por fases sería:

- EDA (Exploratory Data Analysis) del dataset usando python (Jupyter Notebooks)
- Probar que tal rinde el modelo *[[Modelos Zero-shot|zero-shot]]*
	==En esta fase va a ser fundamental entender la arquitectura completa de un modelo como SAM3.==
- Diseñar y programar la herramienta usando [[FastAPI]] y Docker.

La respuesta de la API debe ser algo en la siguiente línea:
```json
{   // objeto json que va a guardar todos los campos
  "vehicle_damage_assessment": {
    "damage_class": "scratch", // qué clase de desperfecto ha sido detectado
    "car_part": "Front Bumper", // en qué zona del coche se encuentra el daño
    // una breve descripción
    "description": "Surface scratches across the front bumper with paint damage",
    // una especificación más detallada de dónde se encuentran los daños
    "affected_area": "Front bumper panel (upper, mid, and lower sections)",
    // un grado de severidad de daños (heat-map de daños puede ser interesante??)
    "severity": "Moderate",
    // acción recomendada a tomar por el VLLM
    "recommended_action": "Paint correction / respray"
  }
}
```

La mayoría de estos datos van a ser generados por un VLLM (Visual Large Language Model) que recibirá la clase predicha por el modelo de segmentación además de la imagen con el bounding box y área marcada.
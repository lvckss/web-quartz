---
class: uni/nota
fecha: 2026-03-22
hora: 13:13
asignatura: Proyecto Integrador
tags:
  - piia1
---
> [!info]
> En esta nota se asume que se está evaluando **segmentación por instancias** en formato tipo **COCO**, es decir:
>
> * cada predicción tiene **máscara**, **categoría** y **score**
> * cada objeto real tiene **máscara** y **categoría**
>
> Para métricas como `mask AP`, `AP75`, `AP_S` y `AP por clase`, la referencia estándar es **COCOeval / pycocotools**. Detectron2 reporta precisamente las métricas `AP`, `AP50`, `AP75`, `APs`, `APm` y `APl` para `segm`. [1.](detectron2.readthedocs.io)

---

## 0. Idea base: IoU, TP, FP, precision y recall

Antes de entrar en cada métrica, hay cuatro conceptos que mandan casi todo:
### IoU

El **IoU** (*Intersection over Union*) mide cuánto se solapa una predicción con la máscara real.
$$
IoU = \frac{\text{intersección}}{\text{unión}}
$$
- Si una máscara predicha coincide mucho con la real, el IoU es alto.
- Si coincide poco, el IoU es bajo.
### TP y FP

Una predicción se considera:

* **TP** (*true positive*) si:
  * la categoría es correcta
  * y el IoU supera el umbral exigido

* **FP** (*false positive*) si:
  * no encuentra ningún objeto real compatible
  * o es una detección duplicada
  * o tiene la clase equivocada

### Precision
$$
Precision = \frac{TP}{TP + FP}
$$

Responde a:

**“De todo lo que predije, cuánto era correcto.”**

### Recall
$$
Recall = \frac{TP}{TP + FN}
$$

Responde a:

**“De todo lo que existía realmente, cuánto fui capaz de encontrar.”**

En `pycocotools`, el evaluador:
* ordena las detecciones por `score`,
* calcula coincidencias predicción–ground truth,
* acumula TP y FP,
* calcula `recall = tp / npig`,
* calcula `precision = tp / (tp + fp)`,
* y luego guarda una rejilla de precisiones por umbral de IoU, recall, clase, tamaño y máximo de detecciones. ([GitHub][2])

---
## 1. `mask AP@[.50:.95]`

## Qué es

Es la **métrica principal** de segmentación por instancias.

`AP@[.50:.95]` significa:
* calcular el rendimiento con **IoU = 0.50**
* luego con **IoU = 0.55**
* luego con **0.60**
* ...
* hasta **0.95**
* y hacer la media

En COCO, eso es justo lo que se entiende por `AP`: un promedio sobre IoUs entre **0.50 y 0.95**. El benchmark de robustez para COCO también lo resume así. 
## Qué mide realmente

Mide, en conjunto:
* si el modelo **encuentra** los daños
* si los encuentra con la **clase correcta**
* si los encuentra con una **máscara suficientemente precisa**
* y cómo se comporta cuando cambias el umbral de confianza

No es solo “qué tal el contorno”.
Tampoco es solo “cuántos encuentra”.
Es una métrica **global de detección + segmentación**.
## Cómo se calcula

Conceptualmente:
1. Tomas todas las predicciones de una clase.
2. Las ordenas por `score` de mayor a menor.
3. Para cada IoU de evaluación (`0.50, 0.55, ..., 0.95`):
   * decides cuáles son TP y cuáles FP
4. Con eso construyes una curva **precision–recall**
5. Tomas el área bajo esa curva
6. Repites para todos los IoUs y haces la media

En `pycocotools`, los umbrales de IoU se crean con:

* `np.linspace(.5, 0.95, ... )`

y los niveles de recall con:

* `np.linspace(.0, 1.00, ... )`

Además, la matriz de precisión final tiene forma:

* `[IoU, Recall, Clase, Área, MaxDets]`

## Cómo interpretarlo

* **Más alto = mejor**
* Si sube `mask AP`, el pipeline en general:
  * detecta mejor
  * segmenta mejor
  * y mete menos errores graves

> [!tip]
> Si solo pudieras elegir **una métrica de calidad**, esta sería la más importante.

## Cómo implementarlo en Python

### Opción estándar con `pycocotools`

```python
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def run_coco_segm_eval(gt_json_path, pred_json_path, img_ids=None):
    coco_gt = COCO(gt_json_path)
    coco_dt = coco_gt.loadRes(pred_json_path)

    coco_eval = COCOeval(coco_gt, coco_dt, iouType="segm")

    if img_ids is not None:
        coco_eval.params.imgIds = img_ids

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval

# Uso:
# coco_eval = run_coco_segm_eval("gt.json", "pred.json")
# mask_ap = coco_eval.stats[0]
print("mask AP@[.50:.95] =", float(coco_eval.stats[0]))
```

### Qué devuelve `stats`

En evaluación tipo COCO para detección/segmentación:

* `stats[0]` → `AP`
* `stats[1]` → `AP50`
* `stats[2]` → `AP75`
* `stats[3]` → `APs`
* `stats[4]` → `APm`
* `stats[5]` → `APl`

> [!warning]
> Según la herramienta, puede aparecer en rango `0–1` o `0–100`.
> Detectron2 lo reporta multiplicado por 100.

---

## 2. `AP75`

## Qué es

Es exactamente el mismo concepto que `AP`, pero exigiendo un único umbral:
$$
IoU = 0.75
$$

O sea, no preguntas “¿el modelo lo detectó más o menos?”, sino:

**“¿lo detectó con bastante precisión?”**

Detectron2 lo reporta como una de las métricas estándar de `segm`.

## Qué mide

Mide la **calidad fina** de la segmentación.

Un pipeline puede tener un `AP50` aceptable porque “da con la zona general”, pero un `AP75` mediocre porque:

* se come fondo
* la máscara es demasiado gorda
* corta partes del daño
* desplaza ligeramente el contorno

## Cómo se calcula

Es el mismo proceso que en AP general, pero usando solo:
$$
IoU = 0.75
$$
En el resumen COCO, `stats[2]` corresponde justo a ese valor.

## Cómo interpretarlo

* **Más alto = mejor precisión espacial**
* Si dos pipelines tienen AP parecido, pero uno tiene mucho mejor `AP75`, ese suele segmentar con más finura

## Implementación en Python

```python
ap75 = float(coco_eval.stats[2])
print("AP75 =", ap75)
```

No necesitas recalcular nada adicional si ya has corrido `COCOeval`.

---

## 3. `AP_S`

## Qué es

Es el `AP` calculado **solo sobre objetos pequeños**.

En COCO, los rangos por tamaño se definen por área:

* `small`: `0^2` a `32^2`
* `medium`: `32^2` a `96^2`
* `large`: `96^2` en adelante

## Qué mide

Responde a:

**“¿Qué tal detecta y segmenta mi pipeline los daños pequeños?”**

Esto es crucial en daños finos como:
* scratches
* cracks
* pequeños desconchones o roturas locales

## Cómo se calcula

Es la misma AP de antes, pero filtrando solo las instancias cuyo tamaño cae en la franja `small`.

Importante para segmentación:

> Cuando Detectron2 evalúa `segm`, elimina el `bbox` de los resultados para que el tamaño `small/medium/large` se calcule usando **el área de la máscara**, no el área de la caja. ([GitHub][3])

Eso es importante porque, en segmentación, el área relevante de verdad es la de la **máscara del daño**.

## Cómo interpretarlo

* **Más alto = mejor en daños pequeños**
* Un pipeline puede tener AP global decente y, aun así, ser malo en `AP_S`
* Si tu caso de uso incluye muchos daños sutiles, `AP_S` es importantísimo

## Implementación en Python

```python
ap_small = float(coco_eval.stats[3])
print("AP_S =", ap_small)
```

---

## 4. `AP` por clase

## Qué es

Es la AP calculada **separadamente para cada categoría**.

No te da un solo número, sino uno por clase:

* `dent`
* `scratch`
* `crack`
* etc.

Detectron2 lo calcula usando la matriz `precision`, cuya forma es:

* `[IoU, Recall, Clase, Área, MaxDets]`

y, para cada clase, toma:

* todas las IoUs
* todos los recalls
* área = `all`
* `maxDets` final

y hace la media de las precisiones válidas.

## Qué mide

Responde a:

**“¿En qué tipos de daño es bueno o malo mi pipeline?”**

Esto evita autoengañarte con el promedio global.

Ejemplo típico:

* el modelo va muy bien en `glass shatter`
* regular en `dent`
* y fatal en `crack`

El AP global puede parecer razonable, pero el pipeline quizá no sirve para la clase que más te interesa.

## Cómo se calcula

Para cada clase `k`:

1. coges `precision[:, :, k, 0, -1]`

   * todas las IoUs
   * todos los recalls
   * esa clase
   * área `all`
   * máximo número de detecciones final
2. eliminas los `-1`
3. haces la media

Eso es exactamente lo que hace Detectron2. ([detectron2.readthedocs.io][1])

## Cómo interpretarlo

* **Más alto = mejor en esa clase**
* Te ayuda a responder:

  * quién detecta mejor grietas
  * quién segmenta mejor arañazos
  * quién falla más en abolladuras

## Implementación en Python

```python
import numpy as np

def compute_per_class_ap(coco_eval):
    precisions = coco_eval.eval["precision"]  # [T, R, K, A, M]
    cat_ids = coco_eval.params.catIds
    cats = coco_eval.cocoGt.loadCats(cat_ids)
    cat_id_to_name = {c["id"]: c["name"] for c in cats}

    results = {}

    for k, cat_id in enumerate(cat_ids):
        precision = precisions[:, :, k, 0, -1]   # all IoUs, all recalls, area=all, maxDets=last
        precision = precision[precision > -1]

        ap = float(np.mean(precision)) if precision.size else float("nan")
        results[cat_id_to_name[cat_id]] = ap

    return results

per_class = compute_per_class_ap(coco_eval)
for cls_name, ap in per_class.items():
    print(cls_name, ap)
```

> [!tip]
> Si quieres presentar resultados como porcentaje:
> `ap * 100`

---

## 5. `FP/image`

## Qué es

`FP/image` significa:

$$
\frac{\text{falsos positivos totales}} {\text{número de imágenes}}
$$

Responde a:

**“¿Cuántas falsas alarmas mete el sistema por imagen, de media?”**

No es una métrica oficial de COCO, pero es muy útil en práctica.

## Qué mide

Mide cuánto **alucina** el pipeline.

Ejemplos de falsos positivos:

* **reflejos que confunde con arañazos!!!!!!!**
* sombras o suciedad que parecen grietas
* duplicados de un mismo daño
* pequeños blobs sin sentido

## Cómo se calcula

Tienes que fijar una regla. La más normal es:

1. trabajar por imagen y por clase
2. ordenar predicciones por `score`
3. hacer matching greedy con ground truth no usada
4. si una predicción no encuentra una GT con `IoU >= umbral`, cuenta como `FP`
5. al final:

$$
FP/image = \frac{FP_{total}}{N_{imágenes}}
$$

## Cómo interpretarlo

* **Más bajo = mejor**
* Un valor como `0.15` significa:

  > 0.15 falsas alarmas por imagen
  >   o, aproximadamente, 1 falsa alarma cada 6-7 imágenes

> [!tip]
> Esta métrica es muy buena para comparar pipelines con mucho prompt ensembling o TTA, porque a veces suben recall pero también meten más basura.

## Implementación en Python

Aquí te dejo una versión sencilla para **máscaras binarias** en NumPy:

```python
import numpy as np
from collections import defaultdict

def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(inter / union) if union > 0 else 0.0

def fp_per_image(gt_by_image, pred_by_image, iou_thr=0.5):
    """
    gt_by_image:
        {
          image_id: [
            {"category_id": int, "mask": np.ndarray(bool)},
            ...
          ]
        }

    pred_by_image:
        {
          image_id: [
            {"category_id": int, "mask": np.ndarray(bool), "score": float},
            ...
          ]
        }
    """
    total_fp = 0
    image_ids = sorted(set(gt_by_image.keys()) | set(pred_by_image.keys()))

    for image_id in image_ids:
        gts = gt_by_image.get(image_id, [])
        preds = pred_by_image.get(image_id, [])

        # agrupar GT por clase
        gt_per_cat = defaultdict(list)
        for gt in gts:
            gt_per_cat[gt["category_id"]].append(gt)

        # agrupar predicciones por clase
        pred_per_cat = defaultdict(list)
        for pred in preds:
            pred_per_cat[pred["category_id"]].append(pred)

        for cat_id, pred_list in pred_per_cat.items():
            pred_list = sorted(pred_list, key=lambda x: x["score"], reverse=True)
            gt_list = gt_per_cat.get(cat_id, [])
            matched = [False] * len(gt_list)

            for pred in pred_list:
                best_iou = 0.0
                best_j = -1

                for j, gt in enumerate(gt_list):
                    if matched[j]:
                        continue
                    iou = mask_iou(pred["mask"], gt["mask"])
                    if iou > best_iou:
                        best_iou = iou
                        best_j = j

                if best_iou >= iou_thr:
                    matched[best_j] = True
                else:
                    total_fp += 1

    return total_fp / max(len(image_ids), 1)
```

---

## 6. `mPC`

## Qué es

`mPC` significa **mean Performance under Corruption**.

Es la media del rendimiento del modelo sobre:

* varios **tipos de corrupción**
* y varios **niveles de severidad**

El benchmark clásico de robustez lo define así:

$$
mPC = \frac{1}{N_c} \sum_{c=1}^{N_c}
\left(
\frac{1}{N_s} \sum_{s=1}^{N_s} P_{c,s}
\right)
$$

donde:

* $N_c$ = número de corrupciones
* $N_s$ = número de severidades
* $P_{c,s}$ = rendimiento bajo la corrupción $c$ y severidad $s$

## Qué mide

Responde a:

**“Cuando el mundo se pone feo, ¿qué rendimiento medio mantiene mi pipeline?”**

Es una métrica de **robustez absoluta**.

## Cómo se calcula

Supón estas corrupciones:

* lluvia
* nieve
* noche

Y 5 severidades por cada una.

Entonces calculas tu métrica base `P` para:

* lluvia-1, lluvia-2, ..., lluvia-5
* nieve-1, nieve-2, ..., nieve-5
* noche-1, noche-2, ..., noche-5

Si tu métrica base es `mask AP`, entonces cada $P_{c,s}$ es un `mask AP`.

Luego haces la media de todos esos valores.

## Cómo interpretarlo

* **Más alto = mejor robustez absoluta**
* Si un pipeline tiene `mPC` alto, significa que sigue funcionando bastante bien en condiciones adversas

> [!info]
> El benchmark de robustez usa `mPC` como métrica de ranking principal, mientras que `Pclean` y `rPC` sirven para interpretar mejor el resultado. 

## Implementación en Python

```python
import numpy as np

def compute_mpc(corruption_scores):
    """
    corruption_scores:
        {
          "rain":  [0.42, 0.39, 0.34, 0.28, 0.21],
          "snow":  [0.40, 0.36, 0.30, 0.25, 0.18],
          "night": [0.38, 0.33, 0.27, 0.20, 0.14],
        }

    Cada lista contiene el rendimiento P_{c,s} para las severidades 1..S
    """
    all_scores = []
    for _, severities in corruption_scores.items():
        all_scores.extend(severities)
    return float(np.mean(all_scores))
```

---

## 7. `rPC`

## Qué es

`rPC` significa **relative Performance under Corruption**.

Se define como:

$$
rPC = \frac{mPC}{P_{clean}}
$$

donde `Pclean` es el rendimiento en datos limpios. El benchmark lo define exactamente así. 

## Qué mide

Responde a:

**“¿Qué fracción de su rendimiento limpio conserva el modelo cuando le meto corrupciones?”**

Es una métrica de **degradación relativa**.

## Por qué no basta con `mPC`

Porque dos pipelines pueden tener el mismo `mPC`, pero venir de situaciones distintas:

* modelo A:

  * `Pclean = 0.60`
  * `mPC = 0.36`
  * `rPC = 0.60`

* modelo B:

  * `Pclean = 0.45`
  * `mPC = 0.36`
  * `rPC = 0.80`

Los dos tienen el mismo `mPC`, pero B **cae menos** respecto a su base.

## Cómo interpretarlo

* **Más alto = mejor conservación del rendimiento**
* `rPC = 0.80` significa:

  * el modelo conserva el 80% de su rendimiento limpio

## Implementación en Python

```python
def compute_rpc(mpc, pclean):
    if pclean == 0:
        return float("nan")
    return float(mpc / pclean)

# ejemplo
pclean = 0.48
mpc = 0.34
rpc = compute_rpc(mpc, pclean)
print("rPC =", rpc)
```

O todo junto:

```python
def compute_mpc_rpc(corruption_scores, pclean):
    mpc = compute_mpc(corruption_scores)
    rpc = compute_rpc(mpc, pclean)
    return mpc, rpc
```

---

## 8. `ms/img`

## Qué es

Son los **milisegundos por imagen**.

Miden la **latencia media de inferencia por imagen**.

## Qué mide

Responde a:

**“¿Cuánto tarda el pipeline en procesar una imagen?”**

No mide calidad, sino **coste temporal**.

Esto importa mucho si tienes:

* TTA
* prompt ensembling
* refinamiento iterativo
* varias pasadas por imagen

## Cómo se calcula

La idea es:

$$
ms/img = \frac{\text{tiempo total}}{\text{número total de imágenes}} \times 1000
$$

En GPU hay que tener cuidado porque muchas operaciones CUDA son asíncronas.
`torch.cuda.synchronize()` espera a que terminen los kernels pendientes, así que conviene usarlo al medir tiempo. PyTorch lo documenta explícitamente.

## Cómo interpretarlo

* **Más bajo = más rápido**
* Si un pipeline tiene `40 ms/img` y otro `120 ms/img`, el primero es unas 3 veces más rápido

## Implementación en Python

Versión simple, suponiendo **una imagen por iteración**:

```python
import time
import torch

def benchmark_ms_per_image(model, input_batches, warmup=10, device="cuda"):
    """
    input_batches: iterable de inputs ya preparados, 1 imagen por elemento
    """
    model.eval()

    # warmup
    with torch.no_grad():
        for i, batch in enumerate(input_batches):
            if i >= warmup:
                break
            _ = model(**batch)
        if device.startswith("cuda"):
            torch.cuda.synchronize()

    total_time = 0.0
    total_images = 0

    with torch.no_grad():
        for batch in input_batches:
            if device.startswith("cuda"):
                torch.cuda.synchronize()

            start = time.perf_counter()
            _ = model(**batch)

            if device.startswith("cuda"):
                torch.cuda.synchronize()

            end = time.perf_counter()

            total_time += (end - start)
            total_images += 1

    ms_per_img = (total_time / max(total_images, 1)) * 1000.0
    return ms_per_img
```

> [!tip]
> Si usas batch > 1, divide por el número total de imágenes, no por el número de batches.

---

## 9. `VRAM pico`

## Qué es

Es la **máxima memoria GPU ocupada** durante la ejecución.

En PyTorch, `torch.cuda.memory.max_memory_allocated()` devuelve el pico de memoria ocupada por tensores, en bytes. PyTorch también aclara que esto no es exactamente lo mismo que lo que ves en `nvidia-smi`, porque usa un **caching allocator** y puede haber memoria reservada pero no ocupada activamente por tensores.

## Qué mide

Responde a:

**“¿Cuál es el máximo consumo real de memoria de este pipeline durante la inferencia?”**

Esto es especialmente importante si trabajas con una GPU ajustada, por ejemplo 10 GB.

## Cómo se calcula

La forma práctica es:

1. resetear estadísticas pico
2. ejecutar la inferencia
3. leer el valor máximo

PyTorch recomienda usar `reset_peak_memory_stats()` para reiniciar el punto de partida del seguimiento del pico.

## Cómo interpretarlo

* **Más bajo = mejor eficiencia de memoria**
* Si el pico pasa de tu VRAM disponible, el pipeline no entra
* Si queda muy cerca del límite, cualquier subida de resolución o batch puede romperlo

## Implementación en Python

```python
import torch

def benchmark_peak_vram(model, input_batches, device="cuda"):
    model.eval()

    if device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        for batch in input_batches:
            _ = model(**batch)

    if device.startswith("cuda"):
        torch.cuda.synchronize()
        peak_bytes = torch.cuda.max_memory_allocated()
        peak_mb = peak_bytes / (1024 ** 2)
        peak_gb = peak_bytes / (1024 ** 3)
        return peak_bytes, peak_mb, peak_gb

    return 0, 0.0, 0.0
```

> [!warning]
> `max_memory_allocated()` mide memoria ocupada por tensores.
> `nvidia-smi` suele reflejar también memoria reservada por el allocator, así que ambas cifras no tienen por qué coincidir exactamente.

---

# Resumen muy corto de interpretación

## `mask AP@[.50:.95]`

La métrica global más importante.
Dice si el pipeline **detecta y segmenta bien en general**. 

## `AP75`

Versión más estricta.
Dice si las máscaras están **bien ajustadas de verdad**.

## `AP_S`

Dice si el pipeline funciona bien en **daños pequeños**.

## `AP por clase`

Dice **en qué clases gana o pierde** el pipeline.

## `FP/image`

Dice cuánto **molesta** el modelo con falsas alarmas.

## `mPC`

Dice qué rendimiento medio mantiene bajo **corrupciones**. 

## `rPC`

Dice qué fracción del rendimiento limpio **conserva** bajo corrupción. 

## `ms/img`

Dice cuánto **tarda** por imagen.

## `VRAM pico`

Dice cuánta **memoria máxima** necesita.

---

# Recomendación práctica para el proyecto

Yo dejaría estas dos capas:
## Capa 1 — calidad

* `mask AP@[.50:.95]`
* `AP75`
* `AP_S`
* `AP por clase`
## Capa 2 — robustez y coste

* `FP/image`
* `Pclean`
* `mPC`
* `rPC`
* `ms/img`
* `VRAM pico`

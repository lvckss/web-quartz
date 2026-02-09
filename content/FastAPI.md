---
class: uni/nota
fecha: 2026-02-09
hora: 11:46
asignatura: Proyecto Integrador
tags:
  - coding/webdev
  - math/ai/unsupervised
aliases:
  - fastapi
---
FastAPI es un framework web que permite crear de forma muy sencilla y rápida APIs web usando python.

---

- Ejemplo:
```python
from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: str | None = None):
    return {"item_id": item_id, "q": q}
```

- Para ejecutarlo (en terminal):
```bash
fastapi dev main.py  
╭────────── FastAPI CLI - Development mode ───────────╮  
│                                                     │  
│ Serving at: http://127.0.0.1:8000                   │  
│                                                     │  
│ API docs: http://127.0.0.1:8000/docs                │  
│                                                     │  
│ Running in development mode, for production use:    │  
│                                                     │  
│ fastapi run                                         │  
│                                                     │  
╰─────────────────────────────────────────────────────╯  
  
INFO: Will watch for changes in these directories: ['/home/user/code/awesomeapp']  
INFO: Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)  
INFO: Started reloader process [2248755] using WatchFiles  
INFO: Started server process [2248757]  
INFO: Waiting for application startup.  
INFO: Application startup complete.
```

Esto nos permite que nuestro navegador pueda ir a por ejemplo a la siguiente URL:
`http://127.0.0.1:8000/items/5?q=somequery` 

Y en esta URL se devolverá una respuesta JSON que en base a la lógica del código, nos servirá para guardar información diversa. En el caso del ejemplo puede ser que devuelva algo cómo:

```json
{"item_id": 5, "q": "somequery"}
```
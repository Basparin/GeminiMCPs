# Tarea Técnica: Corregir Error de Importación en `list_undocumented_functions_tool`

## Descripción del Problema

La herramienta `list_undocumented_functions_tool` en `codesage_mcp/tools/codebase_analysis.py` tiene un error de ejecución cuando se encuentra con ciertos tipos de errores (por ejemplo, `FileNotFoundError`, `SyntaxError`). El error es:

```
NameError: name 'HTTPException' is not defined. Did you mean: 'BaseException'?
```

Esto ocurre porque la función `list_undocumented_functions_tool` intenta lanzar una `HTTPException` (de `fastapi`) en sus bloques `except`, pero esta excepción no está importada en el módulo.

## Ubicación del Error

**Archivo:** `codesage_mcp/tools/codebase_analysis.py`
**Líneas problemáticas:** 204, 206, 208

```python
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
    except SyntaxError as e:
        raise HTTPException(status_code=400, detail=f"Syntax error in {file_path}: {e}")
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {e}"
        )
```

## Solución Propuesta

Agregar la importación necesaria al inicio del archivo.

**Archivo:** `codesage_mcp/tools/codebase_analysis.py`
**Línea a agregar (después de las otras importaciones):**

```python
from fastapi import HTTPException
```

## Verificación

Después de aplicar el cambio, se debe ejecutar nuevamente la herramienta `list_undocumented_functions_tool` sobre un archivo que cause un error (por ejemplo, uno que no exista o tenga un error de sintaxis) para confirmar que el error `NameError` ya no ocurre y se lanza correctamente la `HTTPException` esperada.

Esta corrección es importante para la robustez del servidor MCP y para que los errores se manejen de forma consistente con el framework FastAPI.
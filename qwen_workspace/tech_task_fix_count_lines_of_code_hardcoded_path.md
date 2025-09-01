# [LOC_TOOL_001] `count_lines_of_code_tool` usa una ruta codificada

- **Fecha de Reporte:** 2025-09-01
- **Componente/Archivo:** `codesage_mcp/tools/codebase_analysis.py`
- **Severidad:** Media
- **Estado:** Nuevo

## Descripción del Error

La herramienta `count_lines_of_code_tool`, definida en `codesage_mcp/tools/codebase_analysis.py`, contiene una ruta de sistema de archivos codificada:
```python
current_codebase_path = os.path.abspath("/home/basparin/Escritorio/GeminiMCPs")
```
Esto hace que la herramienta sea dependiente del entorno y no funcione correctamente en otros sistemas o ubicaciones de directorio.

## Comportamiento Actual

La herramienta siempre intenta contar las líneas de código del directorio `/home/basparin/Escritorio/GeminiMCPs`, independientemente de dónde se esté ejecutando el servidor o qué codebase se haya indexado.

## Comportamiento Esperado

La herramienta debería contar las líneas de código de la codebase que está actualmente indexada o permitir al usuario especificar qué codebase analizar.

## Análisis

La ruta codificada hace que la herramienta:
1.  No sea portable.
2.  No funcione en entornos de despliegue como Docker.
3.  No funcione en máquinas de otros desarrolladores.
4.  No funcione si el proyecto se mueve a otra ubicación.

## Posibles Soluciones

#### Solución 1: Permitir que el usuario especifique la ruta de la codebase
- **Descripción:** Modificar la herramienta para que acepte un parámetro `codebase_path`.
- **Pros:**
  - Soluciona el problema de forma flexible.
  - Permite al usuario analizar cualquier codebase indexada.
- **Contras:**
  - Requiere modificar la definición de la herramienta en `main.py` para incluir el nuevo parámetro.
- **Complejidad:** Baja

#### Solución 2: Usar la codebase actualmente indexada
- **Descripción:** Modificar la herramienta para que analice la codebase que está actualmente indexada en `codebase_manager`, sin necesidad de una ruta codificada.
- **Pros:**
  - Soluciona el problema de portabilidad.
  - No requiere un nuevo parámetro.
- **Contras:**
  - Si hay múltiples codebases indexadas, puede ser ambiguo cuál se debe analizar.
- **Complejidad:** Baja

## Recomendación

Se recomienda implementar la **Solución 1** (permitir que el usuario especifique la ruta) ya que es la más flexible y clara. Si no es posible modificar la interfaz de la herramienta, se puede considerar la **Solución 2**.

## Historial de Cambios
- **2025-09-01:** Reporte inicial del error - Qwen Code
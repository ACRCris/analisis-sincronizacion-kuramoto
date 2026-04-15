# Análisis de Sincronización en Redes Neuronales con Dinámica de Kuramoto

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Este repositorio contiene la implementación completa del análisis de sincronización de osciladores de Kuramoto aplicado a redes neuronales, desarrollado como parte de la investigación sobre **"Efecto de la sincronización en el proceso de aprendizaje de una red neuronal"**.

## 📋 Descripción

El proyecto explora cómo la dinámica de sincronización basada en el modelo de Kuramoto afecta el aprendizaje en redes neuronales artificiales. Se analiza el comportamiento crítico del sistema mediante:

- **Transiciones de fase** entre regímenes desincronizados y sincronizados
- **Distribuciones estadísticas** del parámetro de orden y acoplamientos críticos
- **Análisis de criticalidad** en función del parámetro de acoplamiento externo

El código implementa la arquitectura **AKOrN (Artificial Kuramoto Oscillatory Neurons)** sobre el dataset MNIST, permitiendo estudiar cómo la sincronización emerge y afecta la capacidad de clasificación de la red.

## 🗂️ Estructura del Proyecto

```
Analisis_Sincronizacion/
├── analisis_distribuciones/
│   ├── generar_metricas_kuramoto_train.py      # Genera métricas de 60k imágenes MNIST
│   ├── plot_distribuciones_R_stationary.py     # Visualiza distribuciones de R estacionario
│   └── resultados_kuramoto_TRAIN_MAC_60k/      # Datos: métricas temporales completas
│
├── analisis_alpha_c/
│   ├── generar_c_critico_mnist.py              # Calcula c_crítico por imagen
│   ├── generar_R_critico_mnist.py              # Calcula R_crítico por imagen  
│   ├── plot_distribuciones_c_y_R_critico.py    # Distribuciones por clase
│   └── resultados_c_critical/                   # Datos: bases SQLite de criticalidad
│
└── analisis_RvsC/
    ├── generar_curvas_R_vs_C.py                # Genera curvas R(C) completas
    ├── plot_curvas_R_vs_C_todas_clases.py      # Visualiza R(C) por clase
    ├── plot_transicion_fase_promedio.py        # Transición de fase promedio
    └── R_vs_C/                                  # Datos: curvas por imagen/clase
```

## 🚀 Instalación

### Requisitos previos

- Python 3.8 o superior
- PyTorch 2.0+ (con soporte para CUDA/MPS opcional)
- CUDA Toolkit (opcional, para GPU NVIDIA)

### Instalación de dependencias

```bash
# Clonar el repositorio
git clone https://github.com/tu-usuario/analisis-sincronizacion-kuramoto.git
cd analisis-sincronizacion-kuramoto

# Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install torch torchvision numpy scipy matplotlib seaborn tqdm
```

### Preparación de MNIST (recomendado antes de ejecutar análisis)

Desde la carpeta `Analisis_Sincronizacion`, ejecutar:

```bash
python setup_mnist.py
```

Este script replica el flujo de descarga usado en `codigo`:
- crea la ruta local de datos,
- llama `datasets.MNIST(..., download=True)` para train y test,
- valida el dataset con un batch de prueba.

Además, si la descarga remota falla (por ejemplo, error SSL), usa un origen local de respaldo y deja MNIST listo en:

- `Analisis_Sincronizacion/data/MNIST/raw` (ruta principal)

Opcionalmente puedes desactivar sincronización de rutas de compatibilidad:

```bash
python setup_mnist.py --no-sync-compat
```

### Dependencias principales

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
scipy>=1.10.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
```

## 📊 Uso

### 1. Generación de métricas de Kuramoto (60,000 imágenes MNIST)

```bash
cd analisis_distribuciones
python generar_metricas_kuramoto_train.py
```

Este script:
- Procesa todas las imágenes del conjunto de entrenamiento MNIST
- Integra la dinámica de Kuramoto por T=100 pasos temporales
- Calcula series temporales del parámetro de orden r(t)
- Genera métricas de criticalidad (DFA, PSD, entropía espectral)
- **Salida:** `resultados_kuramoto_TRAIN_MAC_60k/metricas_completas_TRAIN_MAC_60k.pt` (~3 GB)

### 2. Análisis de valores críticos

```bash
cd analisis_alpha_c

# Calcular c_crítico (acoplamiento crítico)
python generar_c_critico_mnist.py --all

# Calcular R_crítico (parámetro de orden en criticalidad)
python generar_R_critico_mnist.py --all --use-ch4-n4
```

Opciones disponibles:
- `--all`: Procesar todas las clases (0-9)
- `--clases 0 1 2`: Procesar clases específicas
- `--limite 100`: Limitar número de imágenes por clase
- `--device cuda/mps/cpu`: Seleccionar dispositivo de cómputo

**Salidas:**
- `resultados_c_critical/mnist_c_critical.db` (SQLite)
- `resultados_c_critical/mnist_R_critico.db` (SQLite)

### 3. Curvas de transición de fase R(C)

```bash
cd analisis_RvsC

# Generar curvas R(C) completas
python generar_curvas_R_vs_C.py --clases 0 1 2 3 4 5 6 7 8 9

# O solo para clases específicas
python generar_curvas_R_vs_C.py --clases 9
```

**Salida:** `R_vs_C/R_vs_C.db` (curvas por imagen en c ∈ [0, 0.4])

### 4. Generación de visualizaciones

```bash
# Distribuciones de R estacionario
cd analisis_distribuciones
python plot_distribuciones_R_stationary.py

# Distribuciones de criticalidad
cd ../analisis_alpha_c
python plot_distribuciones_c_y_R_critico.py

# Curvas de transición de fase
cd ../analisis_RvsC
python plot_curvas_R_vs_C_todas_clases.py
python plot_transicion_fase_promedio.py
```

## 🔬 Fundamentos Teóricos

### Modelo de Kuramoto

El modelo describe la dinámica de N osciladores acoplados:

```
θ̇ᵢ = ωᵢ + Σⱼ Jᵢⱼ sin(θⱼ - θᵢ)
```

Donde:
- `θᵢ`: Fase del oscilador i
- `ωᵢ`: Frecuencia natural
- `Jᵢⱼ`: Matriz de acoplamientos

### Parámetro de Orden

Cuantifica el grado de sincronización global:

```
r(t) = |1/N Σⱼ exp(iθⱼ(t))|
```

- `r = 0`: Sistema completamente desincronizado
- `r = 1`: Sistema completamente sincronizado
- `0 < r < 1`: Sincronización parcial

### Implementación Computacional

#### Clase `KConv2d` (capa convolucional de Kuramoto)

Define la conectividad espacial y evolución del sistema:

```
ẋᵢ = Ωᵢxᵢ + Proj_{xᵢ⊥}(cᵢ + Σⱼ Jᵢⱼxⱼ)
```

- `Ωᵢ`: Matriz antisimétrica de frecuencia natural
- `Jᵢⱼ`: Pesos convolucionales de acoplamiento
- `Proj`: Proyección ortogonal (norma unitaria)
- `cᵢ`: Campo de acoplamiento externo

Energía de Lyapunov:
```
E = -Σᵢⱼ xᵢᵀ Jᵢⱼxⱼ - Σᵢ cᵢᵀxᵢ
```

#### Clase `KBlock` (integración numérica)

Método de Euler explícito con proyección:

```
xᵢ⁽ᵗ⁺¹⁾ = Π[xᵢ⁽ᵗ⁾ + γ Δt Δxᵢ⁽ᵗ⁾]
```

Parámetros típicos:
- `γ = 0.7`: Factor de acoplamiento
- `Δt = 0.9`: Paso temporal
- `T = 30/100`: Pasos de integración

## 📈 Resultados Principales

### Transición de Fase

El sistema exhibe una transición continua de segundo orden:
- **c < 0.15**: Régimen desincronizado (R ≈ 0)
- **c ≈ 0.15**: Punto crítico (transición abrupta)
- **c > 0.15**: Régimen parcialmente sincronizado (R ≈ 0.5)

### Distribuciones de Criticalidad

- **R_crítico**: Unimodal, media ≈ 0.035-0.07
- **c_crítico**: Bimodal, sugiere heterogeneidad intraclase
- Consistencia entre clases MNIST (independencia geométrica)

### Impacto en el Aprendizaje

Redes con AKOrN en el régimen crítico:
- Tasa de aprendizaje superior (+2 unidades de accuracy por época)
- Convergencia más rápida (90% accuracy en 10 épocas)
- Mejor generalización

## 🔧 Configuración Avanzada

### Aceleración por GPU

**NVIDIA (CUDA):**
```bash
python script.py --device cuda
```

**Apple Silicon (M1/M2/M3):**
```bash
python script.py --device mps
```

### Parámetros del Modelo Kuramoto

Editar en los scripts de generación:

```python
PARAMS = {
    'ch': 4,           # Canales del estado
    'n': 4,            # Dimensión del oscilador
    'T': 100,          # Pasos temporales
    'gamma': 0.7,      # Factor de acoplamiento
    'del_t': 0.9,      # Paso temporal
    'ksize': 7,        # Tamaño del kernel convolucional
    'init_omg': 0.1,   # Frecuencia natural inicial
}
```

### Reproducibilidad

Todos los scripts usan `SEED = 1` para garantizar reproducibilidad completa:
- Estado inicial `x_init` fijo y reutilizado
- Orden de procesamiento determinista
- Checkpoints automáticos cada 50-100 imágenes

## 📚 Referencias

1. **Kuramoto, Y.** (1984). *Chemical Oscillations, Waves, and Turbulence*. Springer.

2. **Strogatz, S. H.** (2000). "From Kuramoto to Crawford: exploring the onset of synchronization in populations of coupled oscillators." *Physica D*, 143, 1–20.

3. **Miyato, T., et al.** (2024). "Artificial Kuramoto Oscillatory Neurons (AKOrN)." arXiv:2410.13821.

4. **Shew, W. L., & Plenz, D.** (2013). "The functional benefits of criticality in the cortex." *The Neuroscientist*, 19(1), 88–100.

5. **Myrov, V., et al.** (2024). "Hierarchical whole-brain modeling of critical synchronization dynamics in human brains." bioRxiv 2024.05.08.593146.

## 👥 Autores

- **Laura Ximena Rodriguez Quintero**
- **Cristian Camilo Perez Puentes**

**Institución:** Universidad Nacional de Colombia  
**Fecha:** Diciembre 2025

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo [LICENSE](LICENSE) para más detalles.

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📧 Contacto

Para preguntas o colaboraciones, por favor contactar a través de:
- Issues de GitHub
- Email institucional (Universidad Nacional de Colombia)

## 🙏 Agradecimientos

- Grupo de Investigación en Física Estadística y Sistemas Complejos, UNAL
- Comunidad de desarrolladores de PyTorch y torchvision
- Proyecto AKOrN original de Miyato et al.

---

**Nota:** Este proyecto forma parte de una investigación académica sobre sincronización en redes neuronales. Los resultados y el código están disponibles para uso académico y de investigación.

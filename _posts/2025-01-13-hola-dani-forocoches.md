---
layout: post
title: "Hola Dani, forocoches es mi primer blog y este es mi segundo!"
date: 2025-01-13
categories: [machine-learning, finanzas]
---

Este es el primer post de mi blog donde documentaré el desarrollo de mi TFM: un rebalanceador de estrategias de inversión basado en régimen económico.

## El problema

Los mercados financieros no se comportan de manera uniforme. Existen diferentes **regímenes** caracterizados por distintos niveles de volatilidad y tendencia:

- **Bull market**: Tendencia alcista, volatilidad moderada
- **Bear market**: Tendencia bajista, volatilidad alta
- **Sideways**: Sin tendencia clara, volatilidad baja

La idea es: ¿podemos detectar automáticamente estos regímenes y adaptar nuestra estrategia?

## Hidden Markov Models (HMM)

Los HMM son una herramienta clásica para este problema. La idea es que los retornos observados \\(r_t\\) dependen de un estado oculto \\(s_t\\) que evoluciona según una cadena de Markov.

La probabilidad de transición entre estados se define como:

$$P(s_t = j | s_{t-1} = i) = a_{ij}$$

Y las emisiones (retornos observados) siguen una distribución condicional al estado:

$$r_t | s_t = k \sim \mathcal{N}(\mu_k, \sigma_k^2)$$

## Implementación básica en Python

Usando la librería `hmmlearn`:

```python
import numpy as np
from hmmlearn import hmm

# Datos de retornos (ejemplo)
returns = np.random.randn(1000, 1) * 0.02  # Simular retornos

# Definir modelo con 2 estados (bull/bear)
model = hmm.GaussianHMM(
    n_components=2,
    covariance_type="full",
    n_iter=100,
    random_state=42
)

# Entrenar
model.fit(returns)

# Predecir estados
hidden_states = model.predict(returns)

# Ver parámetros aprendidos
print("Medias por estado:", model.means_.flatten())
print("Volatilidades:", np.sqrt(model.covars_.flatten()))
```

## Visualización

Una vez detectados los regímenes, podemos visualizarlos sobre el precio:

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 6))

# Colorear según régimen
colors = ['green' if s == 0 else 'red' for s in hidden_states]
ax.scatter(range(len(returns)), returns, c=colors, alpha=0.5, s=10)

ax.set_xlabel('Tiempo')
ax.set_ylabel('Retorno')
ax.set_title('Retornos coloreados por régimen detectado')
plt.savefig('regimes.png', dpi=150)
plt.show()
```

![Ejemplo de regímenes detectados](/assets/images/ejemplo-regimes.png)

## Próximos pasos

1. Aplicar esto a datos reales del S&P 500
2. Comparar con otros métodos (switching regression, clustering)
3. Evaluar si la detección de régimen mejora el Sharpe ratio

---

*Este post es parte de mi documentación del TFM. El código completo estará disponible en mi GitHub.*

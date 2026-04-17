## Comparativa de Modelos

| Característica | Q-Learning 1.0 | Q-Learning 2.0 (Smart) | PPO (Deep RL) |
| :--- | :--- | :--- | :--- |
| **Entrada** | Coordenadas fijas | Vector de Contexto (8 vecinos) | Estado del tablero (Tensor) |
| **Acciones** | Solo Revelar | Revelar + Banderas | Espacio discreto optimizado |
| **Generalización** | Baja (ligado al tamaño) | Alta (basado en patrones) | Máxima (Red Neuronal) |
| **Rendimiento** | ~10% victorias | ~65% victorias | >85% victorias (estabilizado) |

## 🧪 Conclusiones y Aprendizaje

1. **La importancia del Feature Engineering:** El gran salto de calidad ocurrió entre la v1 y v2. Al pasar de coordenadas absolutas a un **Vector de Contexto**, el agente dejó de "memorizar" el tablero y empezó a "entender" la lógica del juego.
2. **Reward Shaping:** Ajustar las recompensas por colocar banderas fue clave. Sin una penalización clara por banderas falsas, el agente tendía a ser demasiado conservador o agresivo.
3. **Q-Learning vs PPO:** Mientras que Q-Learning es excelente para entender los fundamentos, PPO demostró ser mucho más estable y capaz de manejar la estocasticidad del Buscaminas gracias a su política de optimización proximal.

## 🚀 Próximos Pasos (Roadmap)
- Implementar una **CNN (Convolutional Neural Network)** para que el agente "vea" el tablero como una imagen.
- Crear una interfaz gráfica con **Streamlit** para jugar contra la IA en tiempo real.

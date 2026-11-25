# C++23 numerical computing spike

[Engineering Spike #9](https://github.com/kshakirov/Agnostic-Inference-Engine/issues/9)

- `inference_core/`: reusable numerical routines as they emerge.
- `experiments/`: standalone checks of accuracy, allocation, and performance.

First experiment: reproduce the Python global linear regression on the same
inputs and compare coefficients. CLIPS lives separately in [src/clips](../clips/).

Первый стенд — глобальная линейная регрессия. Берём те же входы, что в Python,
и сверяем коэффициенты. Структура пока рабочая, не финальная архитектура.

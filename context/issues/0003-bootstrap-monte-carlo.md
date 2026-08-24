# Issue #3: Empirical Bootstrap Monte Carlo Engine

- **State:** Open / IN_PROGRESS
- **Created:** 2026-08-11
- **Updated:** 2026-08-11
- **Assignee:** `kshakirov`
- **Source:** https://github.com/kshakirov/Agnostic-Inference-Engine/issues/3

## Goal

Создать чистый непараметрический empirical bootstrap для оценки SLA-квантилей на Python без внешних статистических библиотек.

```python
def bootstrap_SLA_quantile(
    sample: tuple,
    B: int = 1000,
    q: float = 0.99,
) -> tuple:
    ...
```

## Data contract

- Элемент: `(status: bool, metric: float)`.
- `status=True` означает ошибку, `False` — успех.
- `metric` — latency в миллисекундах.
- Выборка — неизменяемый кортеж из $N=10257$ записей `wrk`.

## Sampling mechanics

Каждая запись получает одинаковый эмпирический вес:

$$
p_i=\frac1N.
$$

Для каждой из $B$ итераций формируется bootstrap-реплика размера $N$ равновероятной выборкой с возвращением. Реплика сжимается в одно значение $q$-квантиля. Результатом является отсортированный неизменяемый кортеж из $B$ оценок.

## Constraints

1. Не использовать `numpy`, `scipy`, `pandas` и `sklearn`.
2. Чистый stateless-конвейер без ООП-состояния и побочных эффектов.
3. Структура должна прямо переноситься в плоские массивы C/C++ или Rust.

## Definition of Done

- [ ] Реализован `bootstrap_SLA_quantile` для POD-кортежей.
- [ ] Подтверждена равномерность выборки без систематического смещения.
- [ ] Возвращается отсортированный кортеж длины $B$.
- [ ] Выполнена проверка на базовом датасете $N=10257$.

# Issue #4: Prototype Maximum Likelihood Estimation Core Engine

- **State:** Open
- **Created:** 2026-08-17
- **Updated:** 2026-08-17
- **Assignee:** `kshakirov`
- **Source:** https://github.com/kshakirov/Agnostic-Inference-Engine/issues/4

## Description

Разработать прототип MLE-движка для валидации математических моделей, создания Ground Truth и последующего переноса алгоритмов на Rust/C++.

## Objectives

- Вычислять log-likelihood, score/gradient и Hessian.
- Реализовать методы оптимизации для поиска $\theta^*$.
- Генерировать синтетические данные и автоматически проверять сходимость.
- Подготовить эталонные входные и выходные векторы для Rust/C++ unit-тестов.

## Technical scope

- **Baseline:** Gaussian distribution, аналитическое и численное решения.
- **Regression:** logistic regression и Poisson regression.
- **Optimizers:** L-BFGS-B и Newton—Raphson.

## Acceptance criteria

- [ ] Реализованы функции log-likelihood для выбранных распределений.
- [ ] Реализован общий интерфейс оптимизатора.
- [ ] Оценки на синтетических данных сходятся к $\theta^*$ с заданной точностью.
- [ ] Экспортируются Hessian и стандартные ошибки:

$$
\operatorname{SE}(\hat\theta)
=
\sqrt{\operatorname{diag}\left(I^{-1}(\hat\theta)\right)}.
$$

- [ ] Сформированы CSV/JSON-фикстуры для C++/Rust.

## Comment: Use Case 01 — High-Throughput Traffic Anomaly Detection

Источник: https://github.com/kshakirov/Agnostic-Inference-Engine/issues/4#issuecomment-5311792926

На скользящем окне нормального RPS чистая MLE-функция оценивает профиль $\hat\theta=(\hat\mu,\hat\sigma^2)$. Для новых значений вычисляется log-likelihood; значения с низким правдоподобием рассматриваются как возможные DDoS-всплески или аномалии.

Требования:

- чистые stateless-функции;
- синтетический ряд с размеченными атаками;
- JSON Ground Truth;
- сигнатуры, переносимые в Rust/C++.

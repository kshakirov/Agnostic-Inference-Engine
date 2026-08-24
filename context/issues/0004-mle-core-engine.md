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

## Comment: Task 1.1 — Gaussian MLE and Fisher Information Validation

Источник: https://github.com/kshakirov/Agnostic-Inference-Engine/issues/4#issuecomment-5390794422

### Goal

Экспериментально проверить аналитические свойства MLE для нормальной модели и связать информацию Фишера с фактическим разбросом оценок серверной латентности.

### Stage A: Parametric Monte Carlo with Ground Truth

1. Зафиксировать параметры генератора: $\mu$, $\sigma$, размер выборки $n$ и число повторов $B$.
2. Сгенерировать $B$ независимых выборок непосредственно из $\operatorname{Normal}(\mu,\sigma^2)$.
3. Для каждой выборки вычислить MLE-оценки $\hat\mu$ и $\hat\sigma^2$.
4. Сохранить распределение оценок без хранения всех промежуточных выборок.
5. Сравнить эмпирическое стандартное отклонение $\hat\mu$ с прогнозом $\sigma/\sqrt n$.

### Stage B: Fisher Information

1. Вычислить информацию Фишера для $\mu$:

$$
I_n(\mu)=\frac{n}{\sigma^2}.
$$

2. Получить прогноз стандартной ошибки:

$$
\operatorname{SE}(\hat\mu)=\frac{1}{\sqrt{I_n(\mu)}}.
$$

3. Сопоставить прогноз с параметрическим Monte Carlo.
4. Проверить влияние $n$ и $\sigma$ на разрешающую способность эксперимента.

### Stage C: Empirical Bootstrap Comparison

1. Сохранить одну исходную выборку латентностей.
2. Построить bootstrap-реплики выборкой с возвращением из эмпирического распределения.
3. Оценить bootstrap-распределение $\hat\mu$.
4. Сравнить Fisher theory, parametric Monte Carlo и empirical bootstrap.

### Architectural constraints

- Чистые stateless-функции и явные POD-данные.
- Потоковое вычисление моментов там, где исходная выборка после агрегации не требуется.
- Исследовательский режим сохраняет данные, необходимые для проверки формы распределения и хвостов.
- Фиксированный seed для воспроизводимых Ground Truth fixtures.
- Результаты пригодны для проверки реализаций на Rust/C++.

### Definition of Done

- [ ] Аналитические MLE-оценки Gaussian-модели воспроизводят Ground Truth в пределах ожидаемой ошибки.
- [ ] Эмпирический разброс $\hat\mu$ согласуется с $\sigma/\sqrt n$.
- [ ] Обратная информация Фишера согласуется с Monte Carlo оценкой дисперсии.
- [ ] Bootstrap-оценка ошибки сопоставлена с параметрическим эталоном.
- [ ] Сохранены воспроизводимые JSON/CSV fixtures и параметры эксперимента.

# TOOLS-001 — Что уже лежит в ящике

Это не парадный API и не обещание стабильности. Это реестр того, что уже можно
взять в `inference_core`, какой контракт у функции и где пока торчат провода.

## Загрузка Vegeta

### `utils.prep_data(file_name, length)`

Читает первые `length` строк Vegeta CSV, берёт третью колонку `latency` в
наносекундах и возвращает NumPy-массив в миллисекундах.

```python
from inference_core.utils import prep_data

latency = prep_data("samples.csv", 600)
```

Это загрузчик с фиксированным размером, а не общий CSV parser. Статусы HTTP он
не фильтрует: валидность прогона должна быть проверена раньше.

## Эмпирика и plug-in оценки

### `ks_tools.empirical_cdf(sample)`

Возвращает пару `(x, y)`: отсортированную выборку и значения ECDF
`1/n, ..., 1`. Используется для графика, KS-расстояния и поиска хвостов.

### `nonparametric.calculate_plug_in_quantile(sample, q_value)`

Возвращает эмпирический квантиль исходной выборки. Принимает обычную числовую
последовательность или NumPy-массив и `q_value` от `0` до `1`.

```python
from inference_core.nonparametric import calculate_plug_in_quantile

q25 = calculate_plug_in_quantile(latency, 0.25)
q75 = calculate_plug_in_quantile(latency, 0.75)
iqr = q75 - q25
```

Используется сейчас для робастного масштаба в правиле Сильвермана.

### `nonparametric.calculate_empirically_cdf(sample, t)`

Старый точечный вариант ECDF для POD-кортежей: считает долю наблюдений, у
которых поле значения не больше порога `t`. Название и контракт ещё надо
унифицировать с `ks_tools.empirical_cdf`.

### `nonparametric.calculate_plug_in_conf_interval(sample, confidentiality)`

Берёт центральный интервал из готовой выборки оценок, обычно после bootstrap.
Это percentile interval, а не универсальный доверительный интервал на все
случаи жизни.

## KS и параметрический bootstrap

### `ks_tools.ks_distance(model_cdf, empirical_y)`

Считает максимальное вертикальное расстояние между модельной CDF и обеими
сторонами ступенчатой ECDF.

### Bootstrap-шаги

- `bootstrap_normal_step(mu, sigma, length, rng)`;
- `bootstrap_gamma_step(shape, scale, length, rng)`;
- `bootstrap_weibull_step(shape, scale, length, rng)`.

Каждый шаг генерит одну параметрическую выборку, заново оценивает её параметры
и возвращает её KS-расстояние. Это параметрический Monte Carlo под конкретной
гипотезой, не empirical bootstrap.

### `ks_tools.bootstrap(size, params, length, step)`

Повторяет выбранный bootstrap-шаг `size` раз и возвращает выборку расстояний
`D*`.

### `ks_tools.bootstrap_p_value(d_star_s, d_real)`

Считает долю bootstrap-расстояний не меньше наблюдаемого `D_real` с поправкой
`+1` в числителе и знаменателе.

## Wald для разницы средних

Модуль `wald_test_conf_interval`:

- `mean_difference_standard_error(data_1, data_2)` — стандартная ошибка
  разницы средних;
- `wald_test(data_1, data_2)` — разница средних в единицах её стандартной
  ошибки;
- `confidence_interval(data_1, data_2, critical_value=1.96)` — интервал для
  разницы средних в исходных единицах;
- `p_value(wald)` — двустороннее p-value;
- `power(val, data_1, data_2, critical_value=1.96)` — мощность для явно
  заданного сдвига `val`.

Это асимптотический прибор для средних. Он не отвечает на вопрос о равенстве
распределений целиком.

## Наивный empirical bootstrap

Модуль `bootstrap`:

- `get_metrics(filename)` читает старый двухколоночный формат
  `True,latency`;
- `bootstrap_naive(n, samples, cf)` делает `n` выборок с возвращением и
  применяет к каждой переданный функционал `cf`.

Работает как Ground Truth, но использует Python-списки и `random`; это пока не
быстрый движок.

## Bernoulli-стенд по Вассерману

Модуль `wasserman` содержит ранний стенд для доли успехов, поправки
Agresti–Coull и интервала. Контракт исторический и пока не считается чистым
публичным API. Перед повторным использованием его надо отдельно проверить.

## KDE — текущий стенд

`run_kde_2.py` уже умеет для небольшой выборки:

1. один раз построить матрицу попарных разностей;
2. батчево вычислить Gaussian kernel для выбранного `h`;
3. обнулить диагональ и получить leave-one-out плотности;
4. вернуть средний log-score;
5. получить робастный ориентир `h` по правилу Сильвермана.

Это ещё не библиотечная тула. Следующий долг — проверить сетку `h`, закрепить
контракт и только потом вынести расчёт из стенда в `inference_core`.

## Главное правило реестра

Если функция появилась в экспериментальном скрипте, это ещё не инструмент.
Инструмент начинается там, где понятны вход, выход, предположения, единицы и
способ проверки результата.

from dataclasses import dataclass
from typing import Generic, TypeVar

# Переменная типа для абстрактного измерения латентности (ms или наносекунды)
T = TypeVar('T', int, float)

@dataclass(frozen=True)
class SamplePoint(Generic[T]):
    """
    Чистый Степановский концепт (POD-структура).
    Уважает кэш-линии, не имеет скрытого состояния и побочных эффектов.
    """
    metric: T     # Время отклика прокси (float) ИЛИ тики CPU алгоритма (int)
    status: bool  # Предикат: True — Сбой/Drop/Cache-Miss, False — Успех


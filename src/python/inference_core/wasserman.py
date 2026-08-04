from math import sqrt

def inject_agresti_coal_feature(sample):
    """
    добавляем 4 эксперимента 2 =1 2=0
    """

    return   sample + [(True,1)*2] + [(False,0)*2]

def calculate_Z_advanced(n):
    """
    считаем Я Z 0.95  по методу Хила-Дэвиса

    """
    df = n - 1
    z = 1.96 + 2.38/df + 2.71 /(df**2)
    print(f"Z = {z}")
    return z

def caclulate_confidential_interval(p_hat, variance, n):
    """
    теперь на вход получаем квадратичное отклонени и несмещенное количество экспериментов
    """
    z = calculate_Z_advanced(n)
    std = z * sqrt(variance / n) #н здесь уже несмещенное
    return (p_hat - std, p_hat + std)



def calculate_empirical_moments(sample):
    """
    Принимает массив данных (sample), где каждый элемент имеет поле .status (True/False).
    Возвращает кортеж: (выборочное_среднее, несмещенная_дисперсия).
    """
    sample = inject_agresti_coal_feature(sample)
    n = len(sample)
    if n < 2:
        raise ValueError("Для расчета разброса (n-1) нужно минимум 2 точки.")
      
    # 1. Точечная доля сбоев (p_hat) — среднее арифметическое булевых предикатов
    success_sum = sum(1 for point in sample if (point[0]))
    p_hat = success_sum / n
    
    # 2. Несмещенная дисперсия Бернулли с поправкой на степень свободы (n-1)
    variance = p_hat * (1.0 - p_hat) * (n / (n - 1))
    conf_int =caclulate_confidential_interval(p_hat, variance, n)
    
    return p_hat, variance, conf_int

#result = ((True, 1),(True, 1))
#calculate_empirical_moments(result)




    

    

def count_hot_days(temperatures):
    hot_days = 0
    for temp in temperatures:
        if temp == 30:
            hot_days += 1
    return hot_days

temps = [30, 25, 30, 28, 32]
print(count_hot_days(temps))

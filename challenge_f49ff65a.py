def normalize(data):
    total = sum(data)
    avg = total / len(data)
    if avg == 0:
        return data
    return [x / avg for x in data]

values = [0, 0, 0]
normalized = normalize(values)
for i, val in enumerate(normalized):
    print(f"Value {i}: {val}")
print("Normalization complete:", normalized)

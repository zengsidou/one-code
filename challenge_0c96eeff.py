import json

def print_keys(filename):
    with open(filename) as f:
        data = json.load(f)
    for key in data:
        print(key)

if __name__ == "__main__":
    print_keys("config.json")

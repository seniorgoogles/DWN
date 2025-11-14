"""Quick check of pickle structure"""
import pickle

with open('pruned_connections_6inputs.pkl', 'rb') as f:
    data = pickle.load(f)

print(f"Type: {type(data)}")
print(f"Length: {len(data) if hasattr(data, '__len__') else 'N/A'}")

if isinstance(data, list):
    print(f"\nFirst element type: {type(data[0])}")
    print(f"First element: {data[0]}")
elif isinstance(data, dict):
    print(f"\nKeys: {data.keys()}")
    for k, v in list(data.items())[:3]:
        print(f"{k}: {type(v)}")

import pickle

with open('pruned_connections_6inputs.pkl', 'rb') as f:
    data = pickle.load(f)

connections = data['connections']
print(f"Number of layers: {len(connections)}")
print(f"\nFirst layer keys: {connections[0].keys() if isinstance(connections[0], dict) else 'Not a dict'}")
print(f"First layer type: {type(connections[0])}")
print(f"\nFirst layer content (first 200 chars):")
print(str(connections[0])[:200])

import argparse, json, urllib.request
AP = argparse.ArgumentParser(); AP.add_argument("--port", type=int, required=True); AP.add_argument("--out", required=True)
A = AP.parse_args()
URL = f"http://127.0.0.1:{A.port}/v1/chat/completions"
def call(content, mt=60):
    body = json.dumps({"model":"q3","messages":[{"role":"user","content":content}],"max_tokens":mt,"temperature":0}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type":"application/json"})
    return json.load(urllib.request.urlopen(req, timeout=300))["choices"][0]["message"]["content"] or ""
tiny = call("Say hello in one short sentence.")
mid  = call("Context: " + " ".join(f"item {i}" for i in range(80)) + ". What is 3+4? Reply briefly.")
long_ = call("Context: " + " ".join(f"fact {i} is {i*7%13}." for i in range(700)) + " What is 3+4? Reply briefly.")
print("TINY:", tiny[:100].replace("\n"," "))
print("MID :", mid[:100].replace("\n"," "))
print("LONG:", long_[:100].replace("\n"," "))
json.dump({"tiny":tiny,"mid":mid,"long":long_}, open(f"{A.out}/tiny_probe.json","w"))

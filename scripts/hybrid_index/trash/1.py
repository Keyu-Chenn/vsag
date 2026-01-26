import requests

url = "https://paiplusinferencepre.alipay.com/inference/cc20eb65be9218db_q2host_retrieval_host_offline_m3sparse_a10_addLogTest/bge_m3_sparse_fp32"
body = {"features":{},"tensorFeatures":{"query":{"shapes":[1],"stringValues":["123456"]},"max_length":{"longValues":[128],"shapes":[1]}}}
headers = {
    "Content-Type": "application/json;charset=utf-8",
    "MPS-app-name": "your-app-name",
    "MPS-http-version": "1.0",
    "MPS-trace-id": "your-trace-id"
}

r = requests.post(url=url, json=body, headers=headers)
res = r.json()
print(res)
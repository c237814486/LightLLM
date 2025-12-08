Invoke-RestMethod -Uri "http://127.0.0.1:10083/generate" `
    -Method Post `
    -ContentType "application/json" `
    -Body '{"inputs": "1+1=", "parameters": {"max_new_tokens": 20}}'
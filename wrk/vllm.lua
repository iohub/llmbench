
JSON = require("JSON")
wrk.method = "POST"
wrk.headers["Content-Type"] = "application/json"


local text = '介绍李白的诗词风格。'
local prompt = string.format('<s>[INST] You are an AI assisant, answer the follow question in Chinese： %s [/INST]', text)

local data = {
    model = "/home/do/ssd/modelhub/llama-2-7b-hf",
    prompt = prompt,
    max_tokens = 40,
}

local body = JSON:encode(data)
print(body)

wrk.body = body

-- 响应处理函数
response = function(status, headers, body)
    -- 打印HTTP状态码
    print("Status: " .. status)

    -- 打印响应体
    print("Body: " .. body)
end
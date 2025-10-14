from transformers import AutoTokenizer

# 模型名称：可以换成别的Qwen2.5变体，比如 "Qwen/Qwen2.5-7B-Instruct"
model_path = "/mnt/vlm-ks3/fangyiyang/model/Qwen2.5-VL-3B-Instruct"

# 加载本地 tokenizer
print("🚀 正在加载本地 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 输入测试文本
texts = [
    """<coarse_thought>The user said “The view from the top of the mountain was beyond anything I've ever seen.” The overall tone expresses amazement and admiration.</coarse_thought>
<coarse_answer>positive</coarse_answer>
<fine_thought>Words like “beyond anything I've ever seen” indicate astonishment and deep appreciation toward the scene, evoking a sense of wonder.</fine_thought>
<final_answer>amusement</final_answer>""",
    """<coarse_thought>The user said “That cat video made me laugh so hard I nearly dropped my phone.” The tone is cheerful and humorous.</coarse_thought>
<coarse_answer>positive</coarse_answer>
<fine_thought>The expression “laugh so hard” conveys enjoyment and humor, showing the user found</fine_thought>
<final_answer>amusement</final_answer>""",
    """<coarse_thought>The user said “I finally finished my work early and now I can relax with a cup of tea.” The tone feels calm and satisfied.</coarse_thought>
<coarse_answer>positive</coarse_answer>
<fine_thought>The phrases “finished my work early” and “relax with a cup of tea” indicate peaceful satisfaction and comfort, typical of a content state.</fine_thought>
<final_answer>contentment</final_answer>""",
    """<coarse_thought>The user said “I can’t believe I got the job! This is the best news ever!” The tone is extremely energetic and positive.</coarse_thought>
<coarse_answer>positive</coarse_answer>
<fine_thought>Exclamations like “best news ever” and “can’t believe I got the job” express intense joy and enthusiasm, characteristic of excitement.</fine_thought>
<final_answer>excitement</final_answer>"""
]

# 输出分词结果
for text in texts:
    tokens = tokenizer.tokenize(text)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    print("=" * 50)
    print(f"📝 原文: {text}")
    print(f"🔹 Tokens: {tokens}")
    print(f"🔢 Token IDs: {ids}")
    print(f"📊 Token 数量: {len(tokens)}")

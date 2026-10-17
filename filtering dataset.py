import json

with open("mentalchat.json", "r", encoding="utf-8") as f:
    data = json.load(f)

cleaned = []
for row in data["rows"]:
    conv = row.get("row", {}).get("conversation", [])
    if len(conv) >= 2:
        user_turns = [m["content"] for m in conv if m["role"] == "user"]
        assistant_turns = [m["content"] for m in conv if m["role"] == "assistant"]
        
        if user_turns and assistant_turns:
            cleaned.append({
                "user": user_turns[0],
                "assistant": assistant_turns[0]
            })

print(f"Kept {len(cleaned)} clean samples")

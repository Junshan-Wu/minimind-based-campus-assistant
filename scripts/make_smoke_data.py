import json, os
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(os.path.join(root, "data", "build"), exist_ok=True)
out = os.path.join(root, "data", "build", "smoke.jsonl")
qs = [("补考怎么报名？", "补考报名一般在成绩公布后一周内，在教务系统提交申请。"),
      ("图书馆几点开门？", "图书馆工作日 8:00 开馆，周末 9:00 开馆。"),
      ("校园卡丢了怎么办？", "先在校园卡服务平台挂失，再到卡务中心补办。")]
with open(out, "w", encoding="utf-8") as f:
    for i in range(20):
        q, a = qs[i % len(qs)]
        f.write(json.dumps({"conversations": [{"role": "user", "content": q},
                 {"role": "assistant", "content": a}]}, ensure_ascii=False) + "\n")
print("wrote", out, "20 samples")
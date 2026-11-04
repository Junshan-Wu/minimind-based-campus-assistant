"""把 {"instruction","output"} 数组的 JSON 数据集转成 MiniMind 的 JSONL 对话格式。

输入：任意 JSON 数组，元素形如 {"instruction": "...", "output": "..."}
输出：每行一个 {"conversations":[{"role":"user","content":...},{"role":"assistant","content":...}]}

只改结构，不改任何文字内容；转换后会做一次逐条回读校验。

用法：
    python scripts/convert_dataset.py
    python scripts/convert_dataset.py --src "C:\\path\\in.json" --dst "minimind-code\\dataset\\campus_qa.jsonl"
"""

import os
import json
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def convert(src, dst, dedup=False):
    with open(src, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise SystemExit(f"[FATAL] 顶层不是数组：{type(data)}")

    rows, first_idx, problems = [], {}, []
    dropped = diff_answers = 0
    for i, item in enumerate(data, 1):
        if not isinstance(item, dict):
            problems.append(f"#{i} 不是对象")
            continue
        q = item.get("instruction")
        a = item.get("output")
        if not isinstance(q, str) or not q.strip():
            problems.append(f"#{i} 缺少 instruction")
            continue
        if not isinstance(a, str) or not a.strip():
            problems.append(f"#{i} 缺少 output")
            continue

        key = q.strip()
        if key in first_idx:
            keep = first_idx[key]
            if not dedup:
                problems.append(f"#{i} 与 #{keep + 1} 问题重复：{q[:20]}")
            else:
                dropped += 1
                if rows[keep][1] != a:
                    diff_answers += 1
                continue
        else:
            first_idx[key] = len(rows)
        rows.append((q, a))

    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    with open(dst, "w", encoding="utf-8", newline="\n") as f:
        for q, a in rows:
            rec = {"conversations": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": a},
            ]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # 回读校验：行数一致 + 内容逐字相同
    with open(dst, encoding="utf-8") as f:
        back = [json.loads(line) for line in f if line.strip()]
    assert len(back) == len(rows), f"行数不一致 {len(back)} != {len(rows)}"
    for (q, a), rec in zip(rows, back):
        assert rec["conversations"][0]["content"] == q, "user 内容被改变"
        assert rec["conversations"][1]["content"] == a, "assistant 内容被改变"

    if dedup:
        qs = [rec["conversations"][0]["content"].strip() for rec in back]
        assert len(set(qs)) == len(qs), "去重后仍存在问题重复"

    ql = [len(q) for q, _ in rows]
    al = [len(a) for _, a in rows]
    return rows, problems, ql, al, dropped, diff_answers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=r"C:\Users\Jason Wu\Desktop\datasets-odN_0GQf33lA-custom-2026-11-02.json")
    ap.add_argument("--dst", default=os.path.join(ROOT, "minimind-code", "dataset", "campus_qa.jsonl"))
    ap.add_argument("--dedup", action="store_true", help="问题完全相同的只保留第一条，其余丢弃")
    args = ap.parse_args()

    rows, problems, ql, al, dropped, diff_answers = convert(args.src, args.dst, args.dedup)

    print(f"源文件 : {args.src}")
    print(f"输出   : {args.dst}")
    print(f"条目数 : {len(rows)}")
    if args.dedup:
        print(f"去重   : 丢弃 {dropped} 条重复问题（其中 {diff_answers} 条的答案与保留条不同）")
    print(f"问题长度   : min {min(ql)} / 平均 {sum(ql) // len(ql)} / max {max(ql)} 字")
    print(f"答案长度   : min {min(al)} / 平均 {sum(al) // len(al)} / max {max(al)} 字")
    over = sum(1 for x in al if x > 500)
    print(f"答案超过 500 字的有 {over} 条（max_seq_len=340 大约只放得下 500 字左右，超了会被截断）")
    if problems:
        print(f"\n发现 {len(problems)} 个问题：")
        for p in problems[:20]:
            print("  -", p)
    else:
        print("\n校验：全部通过（行数一致、内容逐字未变）")


if __name__ == "__main__":
    main()

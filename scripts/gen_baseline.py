"""生成 docs/baseline_base.md —— 基座模型（未加 LoRA）在校园问答题库上的原始输出。

用途：在训练 LoRA 之前拍一张"基线照片"。训练完成后用同一批问题、同一套解码参数
再跑一次，两边的差异就是 LoRA 带来的全部收益。

用法（在仓库根目录）：
    python scripts/gen_baseline.py
    python scripts/gen_baseline.py --limit 3 --max_new_tokens 64

输出：docs/baseline_base.md（每答完一题就落盘，中断也不丢已完成的）
"""

import os
import sys
import time
import argparse
import platform

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "minimind-code"))

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM  # noqa: E402

# 题库：全部来自《助理日常解答Q&A文件.docx》，覆盖"该转哪个部门"和"招生政策"两类。
# 每条都显式带上"电子科技大学"：模型没有对话上下文，问句里不出现主体时它无法判断在问哪所学校，
# 训练集也必须遵守同样的写法。
QUESTIONS = [
    "电子科技大学的国际本科、成人教育、自学考试问题应该咨询哪个部门？",
    "电子科技大学港澳台学生的录取规则是怎样的？",
    "电子科技大学可以进去参观吗？怎么预约？",
    "电子科技大学博士、硕士研究生的招生信息在哪里查询？",
    "企业想去电子科技大学做招聘宣讲，应该联系谁？",
    "电子科技大学的录取证明、高考档案、出国留学翻译件在哪里开具？",
    "电子科技大学的在校生学分、学籍问题应该找哪个部门？",
    "电子科技大学2025年本科招生章程什么时候公布？在哪里查？",
    "电子科技大学的少数民族预科生一年后怎么转入专业？",
    "电子科技大学现在还招高水平艺术团和高水平运动队吗？",
    "电子科技大学的保研比例和深造率大概是多少？",
    "色盲考生能报考电子科技大学的电子信息工程专业吗？",
    "电子科技大学大一可以转专业吗？有什么限制？",
    "电子科技大学的格拉斯哥学院一年学费多少？进校后能转专业吗？",
    "电子科技大学的清水河校区和沙河校区有什么区别？",
    "电子科技大学平行志愿进档后，服从专业调剂还会被退档吗？",
    "电子科技大学的学费和住宿费分别是多少？",
    "电子科技大学调剂会不会被调到中外合作办学专业？",
    "电子科技大学的强基计划是怎么录取的？",
    "电子科技大学的校园卡丢了怎么办？",  # 域外问题，用来观察模型会不会硬编
]


def build_model(lora=None, lora_rank=16):
    tok_dir = os.path.join(ROOT, "minimind-code", "model")
    out_dir = os.path.join(ROOT, "minimind-code", "out")
    ckpt = os.path.join(out_dir, "full_sft_768.pth")
    if not os.path.isfile(ckpt):
        sys.exit(f"[FATAL] 权重不存在: {ckpt}")

    tokenizer = AutoTokenizer.from_pretrained(tok_dir)
    config = MiniMindConfig(hidden_size=768, num_hidden_layers=8, use_moe=False)
    model = MiniMindForCausalLM(config)
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state, strict=True)

    if lora:
        from model.model_lora import apply_lora, load_lora

        lora_path = os.path.join(out_dir, f"{lora}_768.pth")
        if not os.path.isfile(lora_path):
            sys.exit(f"[FATAL] LoRA 权重不存在: {lora_path}")
        apply_lora(model, rank=lora_rank)
        load_lora(model, lora_path)

    # CPU 上用 fp32：fp16 的矩阵乘在 x86 上要么走慢速模拟、要么直接不支持
    model = model.float().eval()
    n = sum(p.numel() for p in model.parameters())
    return model, tokenizer, n


def answer(model, tokenizer, question, gen):
    messages = [{"role": "user", "content": question}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt")
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, **gen)
    cost = time.time() - t0
    text = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return text.strip(), cost, out.shape[1] - inputs.input_ids.shape[1]


def render(meta, rows, total):
    lines = [
        f"# {meta['title']}",
        "",
        "> 本文件由 `scripts/gen_baseline.py` 自动生成，**模型输出原样保留，未做任何修改**。",
        "> 训练 LoRA 之后，用同一批问题、同一套解码参数再跑一次，两张表并排就是效果证据。",
        "",
        "## 环境与参数",
        "",
        "| 项 | 值 |",
        "| --- | --- |",
        f"| 生成时间 | {meta['time']} |",
        f"| 权重 | {meta['weight']} |",
        f"| 结构 | hidden_size=768, num_hidden_layers=8, use_moe=False |",
        f"| 参数量 | {meta['params'] / 1e6:.2f} M |",
        f"| 设备 | {meta['device']} |",
        f"| 精度 | float32 |",
        f"| 解码 | {meta['decode']} |",
        f"| Python / torch | {meta['py']} / {meta['torch']} |",
        "",
        "## 结果一览",
        "",
        "| # | 问题 | 生成 token | 耗时(s) | 命中要点 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for i, r in enumerate(rows, 1):
        lines.append(f"| {i} | {r['q']} | {r['tokens']} | {r['cost']:.1f} |  |")
    if len(rows) < total:
        lines.append("")
        lines.append(f"> 进度：{len(rows)}/{total}（脚本运行中）")

    lines += [
        "",
        "## 判读标准",
        "",
        "- **命中**：给出本校真实信息（正确的部门 / 电话 / 政策结论），可直接作为答复使用",
        "- **部分**：方向对但缺关键信息（只说“找教务处”，没给电话或条件）",
        "- **未命中**：答非所问、编造事实、或退化成通用常识",
        "",
        "> 训练后跑同一批题目时，把本文件的输出与 LoRA 版本的输出逐条对照，",
        "> 统计“命中 / 部分 / 未命中”三档数量，就是本次微调的全部收益。",
        "",
        "## 逐题原始输出",
        "",
    ]
    for i, r in enumerate(rows, 1):
        lines += [
            f"### Q{i}. {r['q']}",
            "",
            f"**{meta['label']}输出（原样）：**",
            "",
            "```text",
            r["a"] if r["a"] else "(空输出)",
            "```",
            "",
        ]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 题（调试用）")
    ap.add_argument("--only", type=int, default=0, help="只跑第 N 题（1 起，调试用）")
    ap.add_argument("--temperature", type=float, default=0.85, help="采样温度；与上游 eval_llm.py 默认一致")
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--greedy", action="store_true", help="强制贪心解码（可复现，但小模型极易复读，仅用于对照）")
    ap.add_argument("--seed", type=int, default=42, help="采样随机种子；固定后结果可复现")
    ap.add_argument("--repetition_penalty", type=float, default=1.0, help="1.0=关闭，建议 1.1~1.3")
    ap.add_argument("--no_repeat_ngram_size", type=int, default=0, help="0=关闭，建议 3~4")
    ap.add_argument("--lora", default=None, help="LoRA 权重名，如 lora_campus（读取 out/lora_campus_768.pth）")
    ap.add_argument("--lora_rank", type=int, default=16)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.out is None:
        name = "eval_lora.md" if args.lora else "baseline_base.md"
        args.out = os.path.join(ROOT, "docs", name)

    if args.only:
        questions = [QUESTIONS[args.only - 1]]
    elif args.limit:
        questions = QUESTIONS[: args.limit]
    else:
        questions = QUESTIONS
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    if args.greedy:
        gen = {"do_sample": False}
        decode = "贪心（do_sample=False，易复读，仅作对照）"
    else:
        gen = {"do_sample": True, "temperature": args.temperature, "top_p": args.top_p}
        decode = f"采样 temperature={args.temperature}, top_p={args.top_p}, seed={args.seed}"
    gen["max_new_tokens"] = args.max_new_tokens
    if args.repetition_penalty != 1.0:
        gen["repetition_penalty"] = args.repetition_penalty
        decode += f", repetition_penalty={args.repetition_penalty}"
    if args.no_repeat_ngram_size:
        gen["no_repeat_ngram_size"] = args.no_repeat_ngram_size
        decode += f", no_repeat_ngram_size={args.no_repeat_ngram_size}"
    decode += f", max_new_tokens={args.max_new_tokens}"

    tag = f"LoRA({args.lora})" if args.lora else "基座"
    print(f"[1/2] 加载{tag}模型 ...")
    model, tokenizer, n_params = build_model(args.lora, args.lora_rank)
    print(f"      参数量 {n_params / 1e6:.2f} M")
    gen["pad_token_id"] = tokenizer.pad_token_id
    gen["eos_token_id"] = tokenizer.eos_token_id

    meta = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "params": n_params,
        "device": "cpu",
        "max_new_tokens": args.max_new_tokens,
        "decode": decode,
        "py": platform.python_version(),
        "torch": torch.__version__,
        "title": f"LoRA 微调后输出（{args.lora}）" if args.lora else "基座模型基线输出（LoRA 训练前）",
        "label": tag,
        "weight": (
            f"`minimind-code/out/full_sft_768.pth` + `minimind-code/out/{args.lora}_768.pth`"
            if args.lora
            else "`minimind-code/out/full_sft_768.pth`（未加载 LoRA）"
        ),
    }

    rows = []
    print(f"[2/2] 生成 {len(questions)} 条回答 ...")
    for i, q in enumerate(questions, 1):
        # 每题固定种子：既避免贪心的退化复读，又保证重跑结果一致
        torch.manual_seed(args.seed + i - 1)
        text, cost, ntok = answer(model, tokenizer, q, gen)
        rows.append({"q": q, "a": text, "cost": cost, "tokens": ntok})
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(render(meta, rows, len(questions)))
        print(f"      [{i}/{len(questions)}] {cost:5.1f}s {ntok:4d}tok  {q[:24]}")

    print(f"完成 -> {args.out}")


if __name__ == "__main__":
    main()

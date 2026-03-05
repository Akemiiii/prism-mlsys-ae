"""
改编自 https://github.com/chromecast56/sglang/blob/6f145d2eadb93a116134f703358ce76f15381045/benchmark/mtbench/bench_sglang.py

用于测试 SGLang EAGLE/EAGLE3 推测解码（Speculative Decoding）的基准测试脚本

"""
import argparse
import json
import os
import time
import uuid

import sglang as sgl
from sglang.test.test_utils import (
    add_common_sglang_args_and_parse,
    select_sglang_backend,
)


def load_questions(filename):
    """
    从 JSONL 文件中加载 MT-Bench 问题
    
    Args:
        filename: 问题文件路径
    
    Returns:
        问题列表，每个问题包含两个回合的对话
    """
    questions = []
    with open(filename, "r") as fin:
        for line in fin:
            obj = json.loads(line)
            questions.append(obj)
    return questions


def write_answers(filename, model_id, questions, answers):
    """
    将模型生成的答案写入 JSONL 文件
    
    Args:
        filename: 输出文件路径
        model_id: 模型标识符
        questions: 问题列表
        answers: 答案列表，每个答案包含两个回合的回复
    """
    with open(os.path.expanduser(filename), "w") as fout:
        for i in range(len(answers)):
            # 构建符合 MT-Bench 格式的答案 JSON
            ans_json = {
                "question_id": questions[i]["question_id"],
                "answer_id": uuid.uuid4().hex,
                "model_id": model_id,
                "choices": {
                    "index": 0,
                    "turns": [answers[i][0], answers[i][1]],  # 两轮对话的答案
                },
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json) + "\n")


@sgl.function
def answer_single_turn(s, question):
    """
    SGLang 函数：用于回答单轮问题
    
    Args:
        s: SGLang 状态对象
        question: 问题
    """
    # 设置系统提示词，定义助手的行为准则
    s += sgl.system(
        "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
    )
    s += sgl.user(question)
    s += sgl.assistant(sgl.gen("answer"))


@sgl.function
def answer_mt_bench(s, question_1, question_2):
    """
    SGLang 函数：用于回答 MT-Bench 的两轮对话问题
    
    Args:
        s: SGLang 状态对象
        question_1: 第一轮问题
        question_2: 第二轮问题（通常是对第一轮回答的追问）
    """
    # 设置系统提示词，定义助手的行为准则
    s += sgl.system(
        "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
    )
    # 第一轮对话
    s += sgl.user(question_1)
    s += sgl.assistant(sgl.gen("answer_1"))
    # 第二轮对话
    s += sgl.user(question_2)
    s += sgl.assistant(sgl.gen("answer_2"))


def main(args):
    """
    主函数：运行 MT-Bench 基准测试
    
    Args:
        args: 命令行参数，包含问题文件路径、并行度等配置
    """
    # 1. 构建提示词 - 加载问题并准备参数
    questions = load_questions(args.question_file)[: args.num_questions]
    
    # 检查是单轮还是双轮问答
    is_single_turn = len(questions[0]["turns"]) == 1
    
    if is_single_turn:
        # 单轮问答
        arguments = [{"question": q["turns"][0]} for q in questions]
        num_prompt_tokens = sum(len(q["turns"][0]) for q in questions)
    else:
        # 双轮问答
        arguments = [
            {"question_1": q["turns"][0], "question_2": q["turns"][1]} for q in questions
        ]
        num_prompt_tokens = sum(len(q["turns"][0]) + len(q["turns"][1]) for q in questions)
    # 2. 选择并设置 SGLang 后端（可能是 EAGLE 推测解码等）
    backend = select_sglang_backend(args)
    sgl.set_default_backend(backend)

    # 3. 批量运行推理请求
    tic = time.time()
    if is_single_turn:
        rets = answer_single_turn.run_batch(
            arguments,
            temperature=args.temperature, 
            max_new_tokens=2048,  
            num_threads=args.parallel,  
            progress_bar=True,
        )
        # 处理失败的请求，如果没有 answer 字段则使用空字符串
        answers = []
        failed_count = 0
        for i, s in enumerate(rets):
            if "answer" not in s:
                print(f"警告: 第 {i+1} 个请求失败，没有返回答案")
                failed_count += 1
                answers.append(["", ""])
            else:
                answers.append([s["answer"], ""])
        if failed_count > 0:
            print(f"总共 {failed_count}/{len(rets)} 个请求失败")

    else:
        rets = answer_mt_bench.run_batch(
            arguments,
            temperature=0, 
            max_new_tokens=2048, 
            num_threads=args.parallel, 
            progress_bar=True,
        )
        # 处理失败的请求，如果没有 answer 字段则使用空字符串
        answers = []
        failed_count = 0
        for i, s in enumerate(rets):
            if "answer_1" not in s or "answer_2" not in s:
                print(f"警告: 第 {i+1} 个请求失败，没有返回完整答案")
                failed_count += 1
                answers.append(["", ""])
            else:
                answers.append([s["answer_1"], s["answer_2"]])
        if failed_count > 0:
            print(f"总共 {failed_count}/{len(rets)} 个请求失败")

    # 4. 计算总延迟
    total_latency = time.time() - tic
    
    # # 5. 单独测量 prefill 时间
    # print("正在重新运行以测量 prefill 时间...")
    # prefill_tic = time.time()
    # if is_single_turn:
    #     prefill_rets = answer_single_turn.run_batch(
    #         arguments,
    #         temperature=0,
    #         max_new_tokens=1,  # 只生成 1 个 token，主要测量 prefill 时间
    #         num_threads=args.parallel,
    #         progress_bar=True,
    #     )
    # else:
    #     prefill_rets = answer_mt_bench.run_batch(
    #         arguments,
    #         temperature=0,
    #         max_new_tokens=1,  # 只生成 1 个 token，主要测量 prefill 时间
    #         num_threads=args.parallel,
    #         progress_bar=True,
    #     )
    # prefill_time = time.time() - prefill_tic
    
    # # 6. 计算 decode 时间（总延迟减去 prefill 时间）
    # latency = total_latency - prefill_time
    # print(f"总延迟: {total_latency:.3f}s, Prefill 时间: {prefill_time:.3f}s, Decode 时间: {latency:.3f}s")
    
    if is_single_turn:
        # 单轮问答
        num_output_tokens = sum(
            s.get_meta_info("answer")["completion_tokens"]
            for s in rets
        )
        # 检查是否有推测解码验证信息
        has_verify = "spec_verify_ct" in rets[0].get_meta_info("answer")
        if has_verify:
            num_verify_tokens = sum(
                s.get_meta_info("answer")["spec_verify_ct"]
                for s in rets
            )
        else:
            num_verify_tokens = num_output_tokens
    else:
        # 双轮问答
        num_output_tokens = sum(
            s.get_meta_info("answer_1")["completion_tokens"]
            + s.get_meta_info("answer_2")["completion_tokens"]
            for s in rets
        )
        # 检查是否有推测解码验证信息
        has_verify = "spec_verify_ct" in rets[0].get_meta_info("answer_1")
        if has_verify:
            num_verify_tokens = sum(
                s.get_meta_info("answer_1")["spec_verify_ct"]
                + s.get_meta_info("answer_2")["spec_verify_ct"]
                for s in rets
            )
        else:
            num_verify_tokens = num_output_tokens

    # 注意：接受长度 = 完成的 token 数 / 推测验证次数
    # 元信息示例: {'id': '3bb9c5ead109488d8ed5ee9cbecaec29', 'finish_reason': {'type': 'length', 'length': 256}, 
    #             'prompt_tokens': 37, 'spec_verify_ct': 101, 'completion_tokens': 256, 'cached_tokens': 0}

    output_throughput = num_output_tokens / total_latency  # 输出吞吐量（token/秒）
    # output_throughput_without_prefill = num_output_tokens / latency


    # 5. 计算推测解码的接受长度（仅在使用推测解码时有效）
    if has_verify:
        accept_length = num_output_tokens / num_verify_tokens  # 平均每次验证接受的 token 数
    else:
        accept_length = 1.0  # 非推测解码模式，接受长度为 1

    # 6. 打印性能统计
    print(
        f"#questions: {len(questions)}, prompt_tokens: {num_prompt_tokens}, num_output_tokens: {num_output_tokens}, num_verify_tokens: {num_verify_tokens}, Throughput: {output_throughput:.3f} token/s, Acceptance length: {accept_length:.3f}"
    )

    # 7. 写入答案文件（用于后续评估）
    model_id = backend.model_info["model_path"]
    answer_file = args.answer_file or f"tmp_output_{args.backend}.txt"
    write_answers(answer_file, model_id, questions, answers)

    # 8. 获取服务器配置信息
    try:
        server_info = backend.get_server_info()
        # 提取关键配置参数
        server_config = {
            "model_path": server_info.get("model_path"),
            "speculative_algorithm": server_info.get("speculative_algorithm"),
            "speculative_draft_model_path": server_info.get("speculative_draft_model_path"),
            "speculative_num_steps": server_info.get("speculative_num_steps"),
            "speculative_num_draft_tokens": server_info.get("speculative_num_draft_tokens"),
            "speculative_eagle_topk": server_info.get("speculative_eagle_topk"),
            "dtype": server_info.get("dtype"),
            "cuda_graph_max_bs": server_info.get("cuda_graph_max_bs"),
            "tp_size": server_info.get("tp_size"),
            "disable_radix_cache": server_info.get("disable_radix_cache"),
        }
    except Exception as e:
        # 如果获取失败，使用 model_info
        server_config = {
            "model_path": backend.model_info.get("model_path"),
        }
        print(f"Warning: Could not get full server info: {e}")
    
    # 9. 将基准测试结果追加到结果文件
    with open(args.result_file, "a") as fout:
        value = {
            # "task": "mtbench",
            # "backend": args.backend,
            # "num_gpus": 1,
            # "latency": round(total_latency, 3),  # 总延迟（秒）
            "temperature": args.temperature,
            "question_file": args.question_file,
            "throughput": round(output_throughput, 3),  # 吞吐量（token/秒）
            "accept_length": round(accept_length, 3),  # 推测解码接受长度
            # "num_requests": args.num_questions,
            # "question_file": args.question_file,
            # "server_config": server_config,  # 服务器配置参数
            # "other": {
            #     "num_questions": args.num_questions,
            #     "parallel": args.parallel,
            # },
        }
        fout.write(json.dumps(value) + "\n")


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--temperature", type=float, default=0,
                       help="temperature")
    parser.add_argument("--question-file", type=str, default="question.jsonl", 
                       help="MT-Bench 问题文件路径")
    parser.add_argument("--answer-file", type=str, default="answer.jsonl",
                       help="答案输出文件路径（可选）")
    parser.add_argument("--num-questions", type=int, default=80,
                       help="要测试的问题数量")
    # 添加 SGLang 通用参数（如后端选择、模型路径等）
    args = add_common_sglang_args_and_parse(parser)
    main(args)


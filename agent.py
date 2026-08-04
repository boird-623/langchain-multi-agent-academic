"""
agent.py — Agent 核心逻辑（LangChain 1.3 新版写法）

LangChain 1.3 已移除 AgentExecutor，改用 bind_tools + 手动工具调用循环。
这个文件实现了完整的 Agent 运行逻辑。
"""
import os
import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    HumanMessage, SystemMessage, AIMessage, ToolMessage
)
from langchain_core.chat_history import InMemoryChatMessageHistory

from tools import extract_abstract, extract_keywords, generate_outline, search_arxiv


# ─────────────────────────────────────────────────────────────────────────────
# 初始化 LLM（指向 DeepSeek，接口格式与 OpenAI 完全兼容）
# ─────────────────────────────────────────────────────────────────────────────
def _init_llm():
    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise ValueError("❌ 请先设置环境变量 DEEPSEEK_API_KEY")
    return ChatOpenAI(
        base_url="https://api.deepseek.com/v1",
        api_key=api_key,
        model="deepseek-chat",
        temperature=0.7,
    )


# 工具列表（4个）
TOOLS = [search_arxiv, extract_abstract, extract_keywords, generate_outline]

# 工具名 → 函数的映射字典（用于调用时查找对应函数）
TOOL_MAP = {t.name: t for t in TOOLS}

# 系统提示词
SYSTEM_PROMPT = """你是一个专业的学术研究助手，帮助研究人员高效处理文献。

你有四个工具可以使用：
- search_arxiv：在 arXiv 学术数据库上搜索相关论文（需要联网）
- extract_abstract：从论文文本中提取结构化摘要（研究目的/方法/结论）
- extract_keywords：从文本中提取5-10个核心学术关键词
- generate_outline：根据关键词和摘要生成完整的文献综述大纲

处理规则：
1. 用户给出研究主题或关键词时，先用 search_arxiv 检索相关论文
2. 用户提供论文文本时，先提取关键词，再提取摘要，最后生成综述大纲
3. 用户只要摘要时，直接调用 extract_abstract
4. 用户只要关键词时，直接调用 extract_keywords
5. 用户要求"帮我检索/查找/搜索"时，调用 search_arxiv
6. 用户追问或要求修改时，根据对话历史和已有信息回答
7. 始终使用工具完成任务，不要凭空生成内容
8. search_arxiv 的 query 参数必须是英文，如果用户输入中文关键词，先翻译成英文再检索

保持专业学术风格，回答简洁清晰。"""


# ─────────────────────────────────────────────────────────────────────────────
# 对话管理（每个 session_id 独立维护一段对话历史）
# ─────────────────────────────────────────────────────────────────────────────
_session_store: dict[str, InMemoryChatMessageHistory] = {}


def get_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in _session_store:
        _session_store[session_id] = InMemoryChatMessageHistory()
    return _session_store[session_id]


def clear_history(session_id: str):
    """清空指定 session 的对话历史"""
    if session_id in _session_store:
        _session_store[session_id].clear()


# ─────────────────────────────────────────────────────────────────────────────
# 核心处理函数
# ─────────────────────────────────────────────────────────────────────────────
def process_message(user_input: str, session_id: str = "default") -> str:
    """
    处理用户输入，返回 Agent 回复。
    
    参数：
        user_input: 用户输入的文本（可以是论文内容或问题）
        session_id: 会话 ID，同一 ID 共享对话历史
    
    返回：
        str: Agent 的最终文字回复
    """
    llm = _init_llm()
    llm_with_tools = llm.bind_tools(TOOLS)   # 把三个工具绑定到 LLM

    history = get_history(session_id)
    history.add_user_message(user_input)

    # 构建完整消息列表：系统提示 + 历史消息
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(history.messages)

    # ── Agent 推理循环（最多执行 5 次工具调用）────────────────────────────
    max_iterations = 5
    for iteration in range(max_iterations):
        # 让 LLM 决定：直接回答，还是调用某个工具
        response = llm_with_tools.invoke(messages)

        if not response.tool_calls:
            # LLM 决定直接回答（不需要工具），退出循环
            break

        # LLM 要调用工具：把 LLM 的回应追加到消息列表
        messages.append(response)

        # 执行 LLM 要调用的每一个工具
        tool_results = []
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            # 查找并调用对应的工具函数
            if tool_name in TOOL_MAP:
                tool_fn = TOOL_MAP[tool_name]
                result = tool_fn.invoke(tool_args)
            else:
                result = f"❌ 未找到工具：{tool_name}"

            # 把工具结果包装成 ToolMessage 格式（LangChain 规定的格式）
            tool_results.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call["id"],
                )
            )

        # 把工具结果追加到消息列表，让 LLM 继续推理
        messages.extend(tool_results)

    # 获取最终回复文字
    final_reply = response.content if response.content else "处理完成，请查看上方工具返回的结果。"

    # 只把最终文字回复存入历史（不存中间的工具调用过程，保持历史整洁）
    history.add_ai_message(final_reply)
    return final_reply

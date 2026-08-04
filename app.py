"""
app.py — FastAPI 后端 + Gradio 前端（一体化启动）

运行方式：python app.py
访问地址：
  - Gradio 聊天界面：http://localhost:7860
  - FastAPI 接口文档：http://localhost:7860/docs
"""
import os
from dotenv import load_dotenv

# override=True 强制用 .env 里的值，无视系统环境变量
load_dotenv(override=True)

# 启动时打印正在使用的 Key 的后 6 位（方便确认读到了正确的 Key）
_key = os.getenv("DEEPSEEK_API_KEY", "")
print(f"[调试] 当前读到的 DEEPSEEK_API_KEY 末6位: ...{_key[-6:] if _key else '未设置'}")

import gradio as gr
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from agent import process_message, clear_history


# ─────────────────────────────────────────────────────────────────────────────
# 一、FastAPI 后端（提供 HTTP API 接口）
# ─────────────────────────────────────────────────────────────────────────────
api = FastAPI(
    title="学术助手 API",
    description="基于 LangChain 的多 Agent 学术助手后端接口",
    version="1.0.0",
)


class ChatRequest(BaseModel):
    """聊天请求的数据结构"""
    message: str                        # 用户输入的文本
    session_id: str = "default"         # 会话 ID（不同 ID 独立记忆）


class ChatResponse(BaseModel):
    """聊天响应的数据结构"""
    reply: str                          # Agent 的回复文本
    session_id: str                     # 会话 ID
    status: str = "success"             # 状态标记


class ClearRequest(BaseModel):
    """清空对话请求的数据结构"""
    session_id: str = "default"


@api.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    """
    聊天接口 —— 接收用户消息，返回 Agent 回复。

    请求示例：
        POST /chat
        {"message": "帮我搜索 RAG 相关论文", "session_id": "user1"}

    响应示例：
        {"reply": "找到以下论文...", "session_id": "user1", "status": "success"}
    """
    try:
        reply = process_message(req.message, req.session_id)
        return ChatResponse(reply=reply, session_id=req.session_id)
    except ValueError as e:
        return ChatResponse(reply=str(e), session_id=req.session_id, status="error")
    except Exception as e:
        return ChatResponse(
            reply=f"❌ 出错了：{str(e)}",
            session_id=req.session_id,
            status="error",
        )


@api.post("/clear")
def clear_endpoint(req: ClearRequest):
    """清空指定会话的对话历史"""
    clear_history(req.session_id)
    return {"message": "对话已清空", "session_id": req.session_id}


@api.get("/health")
def health_check():
    """健康检查接口（用于部署监控）"""
    return {"status": "ok", "service": "academic-assistant"}


# ─────────────────────────────────────────────────────────────────────────────
# 二、Gradio 前端（聊天界面，挂载到 FastAPI 上）
# ─────────────────────────────────────────────────────────────────────────────
SESSION_ID = "gradio_user"   # Gradio 界面固定使用这个 session ID


def gradio_chat(user_input: str, history: list) -> tuple[str, list]:
    """Gradio 聊天回调：调用 FastAPI 的 /chat 接口逻辑"""
    if not user_input.strip():
        return "", history

    try:
        reply = process_message(user_input, SESSION_ID)
    except ValueError as e:
        reply = str(e)
    except Exception as e:
        reply = f"❌ 出错了：{str(e)}\n\n请检查 API Key 是否正确，或者网络是否可以访问 DeepSeek。"

    history.append({"role": "user",      "content": user_input})
    history.append({"role": "assistant", "content": reply})
    return "", history


def reset_chat():
    """清空对话"""
    clear_history(SESSION_ID)
    return [], []


# 构建 Gradio 界面
with gr.Blocks(title="🎓 学术助手") as demo:

    gr.Markdown("""
    # 🎓 基于 LangChain 的多 Agent 学术助手

    **支持功能：** arXiv 文献检索 | 论文摘要提取 | 关键词提取 | 综述大纲生成 | 多轮追问

    **使用方法：**
    1. 输入研究主题（如「帮我搜索 深度学习 医学影像」），系统自动联网检索 arXiv 论文
    2. 或者直接粘贴论文文本，系统自动提取关键词、摘要，并生成综述大纲
    3. 可以追问：「请帮我扩展第二章节」「搜索更多相关文献」等

    ---
    """)

    chatbot = gr.Chatbot(
        label="对话区",
        height=480,
        placeholder="输入论文文本或问题，按 Enter 发送...",
    )

    with gr.Row():
        txt_input = gr.Textbox(
            label="输入框（支持粘贴大段论文文本）",
            placeholder="在这里输入论文内容或提问...",
            lines=4,
            scale=5,
        )
        with gr.Column(scale=1):
            btn_send  = gr.Button("📨 发送", variant="primary")
            btn_clear = gr.Button("🗑️ 清空对话")

    gr.Markdown("""
    **💡 使用示例：**
    - 「帮我搜索 大语言模型 检索增强生成 相关的论文」→ 自动联网检索 arXiv
    - 「请帮我分析这篇论文：[粘贴论文文本]」→ 提取摘要+关键词+生成大纲
    - 「帮我提取关键词：[粘贴一段文本]」→ 提取核心关键词

    ---
    📡 **API 接口文档：** [点击查看 /docs](http://localhost:7860/docs)
    """)

    txt_input.submit(gradio_chat, [txt_input, chatbot], [txt_input, chatbot])
    btn_send.click(gradio_chat,  [txt_input, chatbot], [txt_input, chatbot])
    btn_clear.click(reset_chat, outputs=[chatbot, chatbot])


# ─────────────────────────────────────────────────────────────────────────────
# 三、把 Gradio 挂载到 FastAPI 上，一起启动
# ─────────────────────────────────────────────────────────────────────────────
app = gr.mount_gradio_app(api, demo, path="/")


if __name__ == "__main__":
    print("=" * 58)
    print("  🎓 学术助手启动中...")
    print("  Gradio 聊天界面: http://localhost:7860")
    print("  FastAPI 接口文档: http://localhost:7860/docs")
    print("=" * 58)
    uvicorn.run(app, host="0.0.0.0", port=7860)

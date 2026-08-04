"""
tools.py — 四个学术工具函数
每个函数都用 @tool 装饰器变成 LangChain 工具

工具清单：
  1. search_arxiv      — 在 arXiv 上检索相关论文（联网）
  2. extract_abstract   — 从论文文本中提取结构化摘要
  3. extract_keywords   — 从文本中提取核心关键词
  4. generate_outline   — 根据关键词和摘要生成综述大纲
"""
import os
import arxiv
from openai import OpenAI
from langchain_core.tools import tool


def _get_client():
    """获取 DeepSeek API 客户端（兼容 OpenAI SDK）"""
    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise ValueError("❌ 请在 .env 文件里设置 DEEPSEEK_API_KEY")
    return OpenAI(
        base_url="https://api.deepseek.com/v1",
        api_key=api_key,
    )


@tool
def extract_abstract(paper_text: str) -> str:
    """
    从学术论文文本中提取结构化摘要。
    返回格式：研究目的、研究方法、主要结论。
    当用户提供了论文内容并需要摘要时调用此工具。
    """
    client = _get_client()
    prompt = f"""请从以下论文中提取关键信息，严格按照下面的格式返回，不要加任何多余内容：

研究目的：[用一句话概括研究要解决什么问题]
研究方法：[简述核心技术方法]
主要结论：[最重要的发现或贡献]

论文内容：
{paper_text[:3000]}
"""
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0.3,   # 低温度，摘要任务要准确不要发挥
    )
    return resp.choices[0].message.content


@tool
def extract_keywords(text: str) -> str:
    """
    从文本中提取5-10个核心学术关键词，用中文逗号分隔返回。
    当用户需要提取关键词时调用此工具。
    """
    client = _get_client()
    prompt = f"提取以下文本的核心学术关键词（5-10个），用中文逗号分隔，只输出关键词不要编号或解释：\n\n{text[:2000]}"
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0.3,
    )
    return resp.choices[0].message.content


@tool
def generate_outline(input_data: str) -> str:
    """
    根据关键词和文献摘要，生成完整的学术文献综述大纲。
    input_data 应包含：研究主题、关键词和摘要信息。
    当用户需要生成综述大纲时调用此工具。
    """
    client = _get_client()
    prompt = f"""你是一名资深学术研究员。
基于以下信息，生成一份标准的学术文献综述大纲：

{input_data}

大纲必须包含以下结构：
1. 引言（研究背景与意义）
2. 研究现状（分2-3个子主题展开）
3. 主要发现与学术争议
4. 研究空白与局限性
5. 未来研究方向

每个章节给出3-5个要点。
"""
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1500,
        temperature=0.7,   # 大纲生成可以有创意空间
    )
    return resp.choices[0].message.content


@tool
def search_arxiv(query: str) -> str:
    """
    在 arXiv 学术数据库上搜索相关论文，返回最相关的 5 篇论文信息。
    每篇包含：标题、作者、发表日期、摘要、arXiv链接。
    当用户要求搜索文献、查找相关论文、了解某领域最新研究时调用此工具。
    query 参数是英文搜索关键词（arXiv 是英文库，中文关键词会自动翻译）。
    """
    try:
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=5,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        results = list(client.results(search))

        if not results:
            return f"未找到与 '{query}' 相关的论文，请尝试更换关键词。"

        output_parts = []
        for i, paper in enumerate(results, 1):
            authors = ", ".join(str(a) for a in paper.authors[:3])
            if len(paper.authors) > 3:
                authors += f" 等（共{len(paper.authors)}位作者）"

            output_parts.append(
                f"【论文{i}】\n"
                f"标题：{paper.title}\n"
                f"作者：{authors}\n"
                f"日期：{paper.published.strftime('%Y-%m-%d')}\n"
                f"摘要：{paper.summary[:300]}...\n"
                f"链接：{paper.entry_id}\n"
            )

        return "\n".join(output_parts)

    except Exception as e:
        return f"arXiv 检索出错：{str(e)}。请检查网络连接或更换关键词重试。"

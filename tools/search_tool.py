import wikipedia
import logging
import time   

wikipedia.set_lang("en")
# 设置合法的开发者 User-Agent，绕过维基百科防火墙拦截
wikipedia.set_user_agent("LongHorizonAgent_Eval/1.0 (Contact: eval_test@example.com)")
def search(query: str, max_results: int = 2) -> str:
    """
    搜索维基百科，返回最相关词条的摘要。
    遇到网络异常或消歧义报错时，返回文本Observation。
    """
    try:
        # 在每次调用 Search 工具的开头，强制休眠 1.5 秒，防止两道题之间或者短时间内多次调用工具导致的并发过高
        time.sleep(1.5) 
        query = query.strip("\"'")
        
        # 1. 先搜索出最相关的词条标题
        titles = wikipedia.search(query, results=max_results)
        if not titles:
            return f"Observation: No Wikipedia results found for '{query}'. Please try different keywords."
            
        formatted_results =[]
        for i, title in enumerate(titles):
            try:
                # 休眠 1 秒
                time.sleep(1.0) 
                # 2. 获取具体页面的摘要
                page = wikipedia.page(title, auto_suggest=False)
                # 截断太长的摘要 
                summary = page.summary[:600] + "..." if len(page.summary) > 600 else page.summary
                formatted_results.append(
                    f"[Result {i+1}]\nTitle: {page.title}\nSnippet: {summary}"
                )
            except wikipedia.exceptions.DisambiguationError as e:
                # 如果遇到消歧义页面，把建议的词条返回给模型
                options = e.options[:3]
                formatted_results.append(
                    f"[Result {i+1}]\nTitle: {title} (Disambiguation)\nSnippet: The query is ambiguous. Please try more specific terms, e.g., {', '.join(options)}"
                )
            except wikipedia.exceptions.PageError:
                continue
            except Exception as e:
                continue
                
        if not formatted_results:
             return "Observation: Relevant titles found, but failed to extract page content."
             
        return "\n\n".join(formatted_results)
        
    except Exception as e:
        logging.warning(f"Wikipedia API Error: {str(e)}")
        return f"Observation: Search tool internal error ({str(e)}). Please refine your keywords or use another tool."



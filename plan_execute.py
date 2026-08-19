import os
import sqlite3
import operator
from typing import Annotated, TypedDict, Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from knowledge import search_knowledge
from langgraph.types import interrupt, Command

from tools import check_erp_balance, calculate_turnover, check_saft_ua_compliance, transfer_inventory

load_dotenv()

# Structured Output Models
class Plan(BaseModel):
    goal: str = Field(description='Головна ціль задачі')
    steps: list[str] = Field(description='Список кроків для досягнення цілі')

class ReplanDecision(BaseModel):
    action: Literal['continue', 'replan', 'finish'] = Field(
        description='continue=виконати наступний крок, replan=змінити план, finish=завершити'
    )
    updated_steps: list[str] | None = Field(
        default=None, description='Оновлені кроки (тільки якщо action=replan)'
    )
    reasoning: str = Field(description='Пояснення рішення')

# State
class PlanExecuteState(TypedDict):
    messages: Annotated[list, operator.add]
    plan: list[str]           
    current_step: int         
    results: list[str]        
    completed: bool           

# LLM 
llm = ChatOpenAI(
    model="google/gemini-2.5-flash",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0.1
)

planner_llm = llm.with_structured_output(Plan)
replanner_llm = llm.with_structured_output(ReplanDecision)

tools = [check_erp_balance, calculate_turnover, check_saft_ua_compliance, search_knowledge, transfer_inventory]
tools_by_name = {t.name: t for t in tools}
llm_with_tools = llm.bind_tools(tools)
RISKY_TOOLS = {'transfer_inventory'}

# Planner Node
def planner_node(state: PlanExecuteState) -> dict:
    user_msg = state['messages'][0].content if state['messages'] else ''
    tool_descriptions = "\n".join([f"- {t.name}: {t.description}" for t in tools])
    
    prompt = (
        f'Створи план для виконання задачі: {user_msg}\n\n'
        f'Ти маєш доступ до таких інструментів:\n{tool_descriptions}\n\n'
        f'Розбий задачу на конкретні кроки. Кожен крок має бути чіткою інструкцією.'
    )
    plan = planner_llm.invoke(prompt)
    return {
        'plan': plan.steps,
        'current_step': 0,
        'results': [],
        'messages': [AIMessage(content=f'Згенеровано план:\n- ' + '\n- '.join(plan.steps))],
        'completed': False
    }

# Executor Node з HITL
def executor_node(state: PlanExecuteState) -> dict:
    step_idx = state['current_step']
    plan = state['plan']

    if step_idx >= len(plan):
        return {'completed': True}

    current_step = plan[step_idx]
    response = llm_with_tools.invoke(
        f'Виконай цей крок: "{current_step}"\nПопередні результати: {state.get("results", [])}'
    )

    result_text = response.content
    if hasattr(response, 'tool_calls') and response.tool_calls:
        tool_outputs = []
        for tc in response.tool_calls:
            
            # Перевірка, чи інструмент є ризиковим
            if tc['name'] in RISKY_TOOLS:
                approval = interrupt({
                    'action': tc['name'],
                    'args': tc['args'],
                    'message': f"Агент намагається виконати ризикову дію: {tc['name']} з параметрами {tc['args']}."
                })
                
                if isinstance(approval, dict) and approval.get('approved'):
                    tool_fn = tools_by_name.get(tc['name'])
                    res = tool_fn.invoke(tc['args'])
                    tool_outputs.append(f"{tc['name']}: {res} (ПІДТВЕРДЖЕНО)")
                else:
                    reason = approval.get('reason', 'Без пояснень') if isinstance(approval, dict) else 'Відхилено'
                    tool_outputs.append(f"{tc['name']}: Скасовано ({reason})")
            
            # Якщо інструмент не ризиковий, виконується без підтвердження
            else:
                tool_fn = tools_by_name.get(tc['name'])
                if tool_fn:
                    tool_outputs.append(f'{tc["name"]}: {tool_fn.invoke(tc["args"])}')
                    
        result_text = " | ".join(tool_outputs)

    if not result_text: result_text = "Виконано (без текстового виводу)."
    step_result_msg = f'Крок {step_idx + 1} виконано: {result_text}'

    return {
        'current_step': step_idx + 1,
        'results': [*state.get('results', []), step_result_msg],
        'messages': [AIMessage(content=step_result_msg)],
    }

# Replanner Node
def replanner_node(state: PlanExecuteState) -> dict:
    plan = state['plan']
    step_idx = state['current_step']
    results = state.get('results', [])

    if step_idx >= len(plan):
        return {'completed': True, 'messages': [AIMessage(content='Всі кроки виконано.')]}

    prompt = (
        f'Початковий план: {plan}\nВиконано кроків: {step_idx}/{len(plan)}\n'
        f'Результати: {results}\nЗалишилось виконати: {plan[step_idx:]}\n'
        f'Виріши: continue, replan або finish.'
    )
    decision = replanner_llm.invoke(prompt)

    if decision.action == 'finish':
        return {'completed': True, 'messages': [AIMessage(content=f'Завершено: {decision.reasoning}')]}
    elif decision.action == 'replan' and decision.updated_steps:
        return {'plan': decision.updated_steps, 'current_step': 0, 'messages': [AIMessage(content=f'План змінено: {decision.reasoning}')]}
    return {}

# Router
def should_end(state: PlanExecuteState) -> Literal['executor', '__end__']:
    if state.get('completed'): return '__end__'
    return 'executor'

# підключення SqliteSaver
graph = StateGraph(PlanExecuteState)
graph.add_node('planner', planner_node)
graph.add_node('executor', executor_node)
graph.add_node('replanner', replanner_node)

graph.add_edge(START, 'planner')
graph.add_edge('planner', 'executor')
graph.add_edge('executor', 'replanner')
graph.add_conditional_edges('replanner', should_end)

conn = sqlite3.connect('agent_state.db', check_same_thread=False)
saver = SqliteSaver(conn)

app_with_interrupt = graph.compile(checkpointer=saver, interrupt_before=['replanner']) # для Завдання 2
app_full_run = graph.compile(checkpointer=saver) # для Завдання 3
app_with_memory = graph.compile(checkpointer=saver) # для Завдання 4

# ЗАПУСК ТЕСТІВ
if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

    print("ТЕСТ ЗАВДАННЯ 2: ЗБЕРЕЖЕННЯ ТА ВІДНОВЛЕННЯ СТАНУ")
    
    config1 = {'configurable': {'thread_id': 'session-001'}}
    query1 = "Перевір залишки по рахунку 281 на складі Основний."

    print("\n[Session-001] СТАРТ (Граф зупиниться перед replanner)")
    app_with_interrupt.invoke(
        {'messages': [HumanMessage(content=query1)], 'plan': [], 'current_step': 0, 'results': [], 'completed': False},
        config=config1
    )
    
    state1 = app_with_interrupt.get_state(config1)
    print(f"[Session-001] ГРАФ ЗУПИНЕНО НА ЧЕКПОІНТІ")
    print(f"Поточний крок (step): {state1.values.get('current_step')}")
    print(f"Згенерований план: {state1.values.get('plan')}")

    print("\n[Session-001] ВІДНОВЛЕННЯ ТА ПРОДОВЖЕННЯ")
    app_with_interrupt.invoke(None, config=config1)
    final_state1 = app_with_interrupt.get_state(config1)
    print(f"Статус завершення (completed): {final_state1.values.get('completed')}")

    print("\n[Session-002] ПЕРЕВІРКА НЕЗАЛЕЖНОСТІ СЕСІЙ")
    config2 = {'configurable': {'thread_id': 'session-002'}}
    state2 = app_with_interrupt.get_state(config2)
    if not state2.values:
        print("Стан для session-002 порожній! Сесії абсолютно незалежні")

    print("ТЕСТ ЗАВДАННЯ 3: AGENTIC RAG VS ACTION TOOLS")

    queries = [
        "Яка версія 1С потрібна для SAF-T UA і які вимоги до контрагентів?",
        "Як розрахувати коефіцієнт оборотності запасів і що він показує?",
        "Перевір залишки на рахунку 281 на складі Основний."
    ]

    for i, query in enumerate(queries):
        print(f"\n--- ЗАПИТ {i+1} ---")
        print(f"Користувач: {query}")
        
        config_rag = {'configurable': {'thread_id': f'rag_test_session_{i}'}}
    
        app_full_run.invoke(
            {'messages': [HumanMessage(content=query)], 'plan': [], 'current_step': 0, 'results': [], 'completed': False},
            config=config_rag
        )

        state_rag = app_full_run.get_state(config_rag)
        print("\nВикористаний план:")
        for step in state_rag.values.get('plan', []):
            print(f" - {step}")
        print("\nРезультати виконання:")
        for res in state_rag.values.get('results', []):
            print(res)

# Завдання 4 Human-in-the-Loop 
    query = "Перемісти 50000 грн по рахунку 281 зі складу Основний на склад Транзитний."

    print("\n" + "="*60)
    print("СЦЕНАРІЙ 1: ОПЕРАТОР ПІДТВЕРДЖУЄ ДІЮ (APPROVE)")
    print("="*60)
    
    config_approve = {'configurable': {'thread_id': 'hitl_approve_session'}}
    
    print("[Система] Запуск агента...")
    for event in app_with_memory.stream({'messages': [HumanMessage(content=query)]}, config_approve):
        if "__interrupt__" in event:
            interrupt_data = event["__interrupt__"][0].value
            print(f"\n ГРАФ ПРИЗУПИНЕНО!")
            print(f"Повідомлення від агента: {interrupt_data['message']}")
            break
            
    print("\n[Оператор] Перевіряє дані та натискає 'ПІДТВЕРДИТИ'...")
    app_with_memory.invoke(Command(resume={"approved": True}), config=config_approve)
    
    state_appr = app_with_memory.get_state(config_approve)
    print("\nФінальний результат у пам'яті:")
    print(state_appr.values.get('results')[-1])


    print("\n\n" + "="*60)
    print("СЦЕНАРІЙ 2: ОПЕРАТОР ВІДХИЛЯЄ ДІЮ (REJECT)")
    print("="*60)
    
    config_reject = {'configurable': {'thread_id': 'hitl_reject_session'}}
    
    print("[Система] Запуск агента (нова сесія)...")
    for event in app_with_memory.stream({'messages': [HumanMessage(content=query)]}, config_reject):
        if "__interrupt__" in event:
            interrupt_data = event["__interrupt__"][0].value
            print(f"\n ГРАФ ПРИЗУПИНЕНО!")
            print(f"Повідомлення від агента: {interrupt_data['message']}")
            break
            
    print("\n[Оператор] Помічає помилку і натискає 'ВІДХИЛИТИ'...")
    app_with_memory.invoke(Command(resume={"approved": False, "reason": "У нас заборонені переміщення на Транзитний у вихідні"}), config=config_reject)
    
    state_rej = app_with_memory.get_state(config_reject)
    print("\nФінальний результат у пам'яті:")
    print(state_rej.values.get('results')[-1])
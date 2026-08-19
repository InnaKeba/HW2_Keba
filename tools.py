from langchain_core.tools import tool
from pydantic import BaseModel, Field

# 1. Інструмент для фінансових розрахунків
class TurnoverInput(BaseModel):
    net_sales: float = Field(description="Чистий дохід або продажі")
    average_inventory: float = Field(description="Середня вартість запасів")

@tool(args_schema=TurnoverInput)
def calculate_turnover(net_sales: float, average_inventory: float) -> str:
    """Розраховує коефіцієнт оборотності запасів Turnover."""
    if average_inventory == 0:
        return "Помилка: середні запаси не можуть дорівнювати нулю."
    result = net_sales / average_inventory
    return f"Коефіцієнт оборотності: {result:.2f}"

# 2. Інструмент для запиту залишків з ERP
class ERPBalanceInput(BaseModel):
    account_code: str = Field(description="Код бухгалтерського рахунку, наприклад '281'")
    warehouse_name: str = Field(description="Назва складу")

@tool(args_schema=ERPBalanceInput)
def check_erp_balance(account_code: str, warehouse_name: str) -> str:
    """Отримує поточні залишки товарів по бухгалтерському рахунку на вказаному складі."""
    db_mock = {
        "281": {"Основний": 150000.00, "Транзитний": 25000.00},
        "282": {"Основний": 500.00}
    }
    balance = db_mock.get(account_code, {}).get(warehouse_name)
    if balance is not None:
        return f"Залишок ({account_code}, '{warehouse_name}'): {balance} грн."
    return f"Даних не знайдено."

# 3. Інструмент перевірки статусу SAF-T UA 
class TaxComplianceInput(BaseModel):
    period: str = Field(description="Період звітності, наприклад 'Q1 2026'")
    system: str = Field(description="ERP система, наприклад '1C Бухгалтерія' або 'ERP'")

@tool(args_schema=TaxComplianceInput)
def check_saft_ua_compliance(period: str, system: str) -> str:
    """Перевіряє статус формування та аудиту звітності SAF-T UA для вказаного періоду та вказаної системи."""
    mock_status = {
        "Q1 2026": {"1C Бухгалтерія": "Успішно сформовано, аудит пройдено", "ERP": "В процесі мепінгу даних"}
    }
    status = mock_status.get(period, {}).get(system, "Дані відсутні")
    return f"Статус SAF-T UA ({period}, {system}): {status}"

# 4. Ризиковий інструмент
class TransferInventoryInput(BaseModel):
    account_code: str = Field(description="Код рахунку (наприклад '281')")
    from_warehouse: str = Field(description="Склад відправник")
    to_warehouse: str = Field(description="Склад одержувач")
    amount: float = Field(description="Сума або кількість для переміщення")

@tool(args_schema=TransferInventoryInput)
def transfer_inventory(account_code: str, from_warehouse: str, to_warehouse: str, amount: float) -> str:
    """Переміщує товари між складами (РИЗИКОВА ДІЯ - ЗМІНЮЄ БАЗУ ДАНИХ).
    Використовується, коли користувач просить перенести, перемістити або відправити товар на інший склад.
    """
    return f"Успішно переміщено {amount} грн по рахунку {account_code} з '{from_warehouse}' на '{to_warehouse}'."
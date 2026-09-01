import numpy as np
import pytest
from cq.engine import StaticAttributeContainer


def test_static_reverse_lookup_concepts():
    mapping = {
        '000001.SZ': ['银行', '核心资产', '沪深300'],
        '000002.SZ': ['房地产', '万科概念'],
        '600000.SH': ['银行', '央企改革'],
        '600519.SH': ['白酒', '核心资产', '茅指数'],
    }
    all_syms = ['000001.SZ', '000002.SZ', '600000.SH', '600519.SH']
    container = StaticAttributeContainer(mapping=mapping, all_symbols=all_syms)

    bank_stocks = container.get_symbols('银行')
    assert set(bank_stocks) == {'000001.SZ', '600000.SH'}

    core_assets = container.get_symbols('核心资产')
    assert set(core_assets) == {'000001.SZ', '600519.SH'}

    non_exist = container.get_symbols('不存在的概念')
    assert non_exist == []

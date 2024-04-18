# 统计所有dataset的统计信息。
from decimal import Decimal
for loader_cls in [TicTacToe, Adult, Bank, Dota2, CreditCard]:
    loader = loader_cls()
    print(f"{loader.name}")
    loader.read(0.1, 123, cuda=False, nrows=None)
    print(f"num samples after encoding: {len(loader.X)}")
    print(f"num features before encoding: {loader.data.shape[1] - 1}")
    print(f"num features after encoding: {loader.X.shape[1]}")
    # print(f"num categorical: {len(data_manager.X_categorical_fields)}")
    print(f"num distinct values: {[len(set(loader.X))]}")
    distinct_counts = list(loader.data.nunique())
    # print(distinct_counts)
    product = Decimal(1)
    for num in distinct_counts:
        product *= Decimal(num)
    # print(distinct_counts)
    # print(size_domain)
    print(f"num domain: {format(product, '.2e')}")
    if loader_cls == CreditCard:
        print(f"num domain: {format(product / 568630, '.2e')}")
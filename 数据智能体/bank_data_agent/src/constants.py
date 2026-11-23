"""银行营销数据集的字段约定。"""

TARGET = "y"
POSITIVE_LABEL = "yes"
NEGATIVE_LABEL = "no"

NUMERIC_COLUMNS = [
    "age",
    "balance",
    "day",
    "duration",
    "campaign",
    "pdays",
    "previous",
]

CATEGORICAL_COLUMNS = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "poutcome",
]

EXPECTED_COLUMNS = [
    "age",
    "job",
    "marital",
    "education",
    "default",
    "balance",
    "housing",
    "loan",
    "contact",
    "day",
    "month",
    "duration",
    "campaign",
    "pdays",
    "previous",
    "poutcome",
    TARGET,
]

# unknown 是原始业务类别，不能放在这里。
MISSING_TOKENS = {"", "?", "??", "n/a", "na", "null", "none", "nan"}

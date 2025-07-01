# Elasticsearch查询转换器使用示例

## 简介

ESQueryConverter 是一个将 MetadataCondition 转换为 Elasticsearch 查询语句的转换器。它支持所有在 protobuf 定义中指定的比较操作符。

## 安装和导入

```python
from raghub_ext.storage_ext.utils.es_query_converter import ESQueryConverter
```

## 基本使用

### 1. 创建转换器实例

```python
converter = ESQueryConverter()
```

### 2. 转换元数据条件

```python
# 定义元数据条件
metadata_condition = {
    "logical_operator": "and",
    "conditions": [
        {
            "name": ["category"],
            "comparison_operator": "is",
            "value": "programming"
        },
        {
            "name": ["rating"],
            "comparison_operator": ">",
            "value": "4.0"
        }
    ]
}

# 转换为Elasticsearch查询
es_query = converter.convert_metadata_condition(metadata_condition)
print(es_query)
```

输出：
```json
{
    "bool": {
        "must": [
            {
                "term": {
                    "category.keyword": "programming"
                }
            },
            {
                "range": {
                    "rating": {
                        "gt": 4.0
                    }
                }
            }
        ]
    }
}
```

## 支持的操作符

### 文本操作符

1. **contains** - 包含
   ```python
   {
       "name": ["title"],
       "comparison_operator": "contains",
       "value": "python"
   }
   ```
   生成：
   ```json
   {
       "wildcard": {
           "title.keyword": "*python*"
       }
   }
   ```

2. **not contains** - 不包含
3. **start with** - 以...开头
4. **end with** - 以...结尾
5. **is** - 等于
6. **is not** - 不等于

### 数值比较操作符

1. **=** - 等于
2. **≠** - 不等于
3. **>** - 大于
4. **<** - 小于
5. **≥** - 大于等于
6. **≤** - 小于等于

### 存在性操作符

1. **empty** - 字段为空
2. **not empty** - 字段非空

### 日期操作符

1. **before** - 早于指定日期
2. **after** - 晚于指定日期

## 新增操作符支持

### 6. 简化比较操作符

1. **eq** - 等于（等同于 `is` 或 `=`）
   ```python
   {
       "name": ["status"],
       "comparison_operator": "eq",
       "value": "active"
   }
   ```

2. **ne** - 不等于（等同于 `is not` 或 `≠`）
   ```python
   {
       "name": ["status"],
       "comparison_operator": "ne",
       "value": "inactive"
   }
   ```

3. **lt** - 小于（等同于 `<`）
   ```python
   {
       "name": ["score"],
       "comparison_operator": "lt",
       "value": "80"
   }
   ```

4. **gt** - 大于（等同于 `>`）
   ```python
   {
       "name": ["rating"],
       "comparison_operator": "gt",
       "value": "4.0"
   }
   ```

5. **le** - 小于等于（等同于 `≤`）
   ```python
   {
       "name": ["age"],
       "comparison_operator": "le",
       "value": "65"
   }
   ```

6. **ge** - 大于等于（等同于 `≥`）
   ```python
   {
       "name": ["price"],
       "comparison_operator": "ge",
       "value": "100"
   }
   ```

### 7. 列表操作符

7. **in** - 包含在列表中
   ```python
   # 逗号分隔格式
   {
       "name": ["category"],
       "comparison_operator": "in",
       "value": "tech,science,programming"
   }
   
   # JSON数组格式
   {
       "name": ["priority"],
       "comparison_operator": "in",
       "value": '["high", "medium", "urgent"]'
   }
   ```

8. **notin** - 不包含在列表中
   ```python
   {
       "name": ["status"],
       "comparison_operator": "notin",
       "value": "deleted,archived,banned"
   }
   ```

## 高级使用

### 1. 多字段条件

```python
condition = {
    "logical_operator": "and",
    "conditions": [
        {
            "name": ["title", "description"],  # 多个字段
            "comparison_operator": "contains",
            "value": "machine learning"
        }
    ]
}
```

### 2. OR 逻辑操作

```python
condition = {
    "logical_operator": "or",
    "conditions": [
        {
            "name": ["category"],
            "comparison_operator": "is",
            "value": "AI"
        },
        {
            "name": ["category"],
            "comparison_operator": "is",
            "value": "ML"
        }
    ]
}
```

### 3. 构建完整ES查询

```python
# 构建包含文本搜索和元数据过滤的完整查询
full_query = converter.build_es_query(
    metadata_condition=metadata_condition,
    query_text="python tutorial",
    size=20,
    from_=0
)
```

## 实际应用示例

### 1. 博客文章搜索

```python
# 搜索编程类别下评分大于4的Python相关文章
blog_condition = {
    "logical_operator": "and",
    "conditions": [
        {
            "name": ["category"],
            "comparison_operator": "is",
            "value": "programming"
        },
        {
            "name": ["rating"],
            "comparison_operator": "≥",
            "value": "4.0"
        },
        {
            "name": ["tags"],
            "comparison_operator": "contains",
            "value": "python"
        },
        {
            "name": ["status"],
            "comparison_operator": "is",
            "value": "published"
        }
    ]
}

es_query = converter.build_es_query(
    metadata_condition=blog_condition,
    query_text="python programming tutorial",
    size=10
)
```

### 2. 文档管理系统

```python
# 搜索最近更新的非草稿状态文档
document_condition = {
    "logical_operator": "and",
    "conditions": [
        {
            "name": ["updated_at"],
            "comparison_operator": "after",
            "value": "2024-01-01"
        },
        {
            "name": ["status"],
            "comparison_operator": "is not",
            "value": "draft"
        },
        {
            "name": ["description"],
            "comparison_operator": "not empty"
        }
    ]
}

es_query = converter.convert_metadata_condition(document_condition)
```

### 3. 产品搜索

```python
# 搜索特定价格范围内的产品
product_condition = {
    "logical_operator": "and",
    "conditions": [
        {
            "name": ["price"],
            "comparison_operator": "≥",
            "value": "100"
        },
        {
            "name": ["price"],
            "comparison_operator": "≤",
            "value": "500"
        },
        {
            "name": ["brand", "manufacturer"],  # 多字段OR查询
            "comparison_operator": "contains",
            "value": "Apple"
        }
    ]
}
```

### 使用新操作符的复杂查询

```python
# 电商产品搜索示例
product_condition = {
    "logical_operator": "and",
    "conditions": [
        {
            "name": ["category"],
            "comparison_operator": "in",
            "value": "electronics,computers,phones"
        },
        {
            "name": ["price"],
            "comparison_operator": "ge",
            "value": "100"
        },
        {
            "name": ["price"],
            "comparison_operator": "le",
            "value": "1000"
        },
        {
            "name": ["status"],
            "comparison_operator": "eq",
            "value": "available"
        },
        {
            "name": ["brand"],
            "comparison_operator": "notin",
            "value": "unknown,generic"
        }
    ]
}

# 用户权限过滤示例
user_condition = {
    "logical_operator": "or",
    "conditions": [
        {
            "name": ["role"],
            "comparison_operator": "in",
            "value": "admin,moderator,editor"
        },
        {
            "name": ["experience_years"],
            "comparison_operator": "gt",
            "value": "5"
        },
        {
            "name": ["certification_level"],
            "comparison_operator": "ge",
            "value": "3"
        }
    ]
}
```

### Elasticsearch生成的查询示例

对于上述产品搜索条件，ES转换器会生成：

```json
{
    "bool": {
        "must": [
            {
                "terms": {
                    "category.keyword": ["electronics", "computers", "phones"]
                }
            },
            {
                "range": {
                    "price": {
                        "gte": 100
                    }
                }
            },
            {
                "range": {
                    "price": {
                        "lte": 1000
                    }
                }
            },
            {
                "term": {
                    "status.keyword": "available"
                }
            },
            {
                "bool": {
                    "must_not": [
                        {
                            "terms": {
                                "brand.keyword": ["unknown", "generic"]
                            }
                        }
                    ]
                }
            }
        ]
    }
}
```

### Qdrant生成的查询示例

对于相同条件，Qdrant转换器会生成相应的Filter对象，包含：
- `MatchAny` 用于 `in` 操作
- `Range` 用于数值比较操作
- `MatchValue` 用于等值匹配
- `MatchExcept` 用于排除操作

## 操作符对照表

| 操作符 | 别名 | 描述 | 示例值 |
|--------|------|------|--------|
| eq | is, = | 等于 | "active" |
| ne | is not, ≠, != | 不等于 | "inactive" |
| lt | < | 小于 | "80" |
| gt | > | 大于 | "4.0" |
| le | ≤, <= | 小于等于 | "65" |
| ge | ≥, >= | 大于等于 | "100" |
| in | - | 包含在列表中 | "a,b,c" 或 ["a","b","c"] |
| notin | not in | 不包含在列表中 | "x,y,z" |

## 注意事项

1. **字段类型**：
   - 文本字段自动使用 `.keyword` 后缀进行精确匹配
   - 数值字段直接使用字段名
   - 日期字段支持标准ISO格式

2. **数值处理**：
   - 自动识别数值类型并转换为相应的int或float
   - 支持负数和小数

3. **日期处理**：
   - 支持多种日期格式：YYYY-MM-DD, YYYY-MM-DD HH:MM:SS等
   - 自动识别日期字符串

4. **错误处理**：
   - 不支持的操作符会被忽略并记录警告
   - 无效的条件会被跳过
   - 空条件返回 `match_all` 查询

5. **数值类型自动识别**：转换器会自动识别数值并进行适当的类型转换
6. **列表格式灵活**：`in` 和 `notin` 操作符支持逗号分隔和JSON数组两种格式
7. **字段后缀处理**：ES转换器会自动为文本字段添加 `.keyword` 后缀进行精确匹配
8. **错误处理**：无效的操作符或值会被记录警告并跳过

## 扩展功能

转换器还提供了构建完整ES查询的功能，包括：

- 文本搜索（multi_match）
- 元数据过滤
- 分页（size, from）
- 字段排除（_source.excludes）

这使得它可以直接用于Elasticsearch客户端进行查询。

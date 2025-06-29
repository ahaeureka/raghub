"""
RAGHub Client 测试工具函数
提供测试中使用的通用工具和辅助函数
"""

import json
import logging
import time
import uuid
from typing import Any, Dict, List
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def generate_unique_id(prefix: str = "test") -> str:
    """生成唯一测试ID"""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def validate_url(url: str) -> bool:
    """验证URL格式"""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def measure_time(func):
    """时间测量装饰器"""

    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"{func.__name__} 执行时间: {end_time - start_time:.3f} 秒")
        return result

    return wrapper


def log_response_summary(response: Any, operation: str):
    """记录响应摘要"""
    if hasattr(response, "error") and response.error:
        logger.error(f"{operation} 失败: {response.error.error_msg}")
    else:
        logger.info(f"{operation} 成功")


def create_test_metadata(category: str, **kwargs) -> List[Dict[str, Any]]:
    """创建测试元数据"""
    metadata = {"category": category, "test_id": generate_unique_id(), "timestamp": str(int(time.time())), **kwargs}
    return metadata


def validate_retrieval_response(response, min_records: int = 0) -> bool:
    """验证检索响应的基本结构"""
    try:
        assert response is not None
        assert hasattr(response, "records")
        assert hasattr(response, "error")

        if response.error:
            logger.warning(f"响应包含错误: {response.error.error_msg}")
            return False

        if response.records is None:
            return min_records == 0

        assert len(response.records) >= min_records

        for record in response.records:
            assert hasattr(record, "content")
            assert hasattr(record, "score")
            assert hasattr(record, "title")
            assert record.content is not None
            assert 0.0 <= record.score <= 1.0

        return True
    except (AssertionError, AttributeError) as e:
        logger.error(f"响应验证失败: {e}")
        return False


def validate_chat_response(response) -> bool:
    """验证聊天响应的基本结构"""
    try:
        assert response is not None
        assert hasattr(response, "choices")
        assert response.choices is not None
        assert len(response.choices) > 0

        choice = response.choices[0]
        assert hasattr(choice, "message")
        assert choice.message is not None
        assert hasattr(choice.message, "role")
        assert hasattr(choice.message, "content")
        assert choice.message.role == "assistant"
        assert choice.message.content is not None

        return True
    except (AssertionError, AttributeError) as e:
        logger.error(f"聊天响应验证失败: {e}")
        return False


def extract_error_info(response) -> Dict[str, Any]:
    """提取错误信息"""
    if hasattr(response, "error") and response.error:
        return {
            "has_error": True,
            "error_code": getattr(response.error, "error_code", None),
            "error_message": getattr(response.error, "error_msg", None),
        }
    return {"has_error": False}


def compare_responses(response1, response2, tolerance: float = 0.001) -> Dict[str, Any]:
    """比较两个响应的相似性"""
    comparison = {"same_record_count": False, "similar_scores": False, "same_content": False}

    try:
        if hasattr(response1, "records") and hasattr(response2, "records") and response1.records and response2.records:
            comparison["same_record_count"] = len(response1.records) == len(response2.records)

            if comparison["same_record_count"]:
                scores1 = [r.score for r in response1.records]
                scores2 = [r.score for r in response2.records]
                comparison["similar_scores"] = all(abs(s1 - s2) <= tolerance for s1, s2 in zip(scores1, scores2))

                contents1 = [r.content for r in response1.records]
                contents2 = [r.content for r in response2.records]
                comparison["same_content"] = contents1 == contents2

    except Exception as e:
        logger.error(f"响应比较失败: {e}")

    return comparison


def create_test_summary(test_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """创建测试摘要"""
    total_tests = len(test_results)
    passed_tests = sum(1 for r in test_results if r.get("passed", False))
    failed_tests = total_tests - passed_tests

    summary = {
        "total_tests": total_tests,
        "passed_tests": passed_tests,
        "failed_tests": failed_tests,
        "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0,
        "details": test_results,
    }

    return summary


def save_test_results(results: Dict[str, Any], filename: str = "test_results.json"):
    """保存测试结果到文件"""
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"测试结果已保存到 {filename}")
    except Exception as e:
        logger.error(f"保存测试结果失败: {e}")


class TestTimer:
    """测试计时器"""

    def __init__(self, name: str):
        self.name = name
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = time.time()
        logger.info(f"开始 {self.name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        logger.info(f"{self.name} 完成，耗时: {duration:.3f} 秒")

    @property
    def duration(self) -> float:
        """获取执行时间"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0

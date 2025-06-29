#!/bin/bash
# RAGHub Client 测试快速启动脚本

echo "================================"
echo "RAGHub Client 测试套件"
echo "================================"

# 检查环境变量
if [ -z "$RAGHUB_TEST_BASE_URL" ]; then
    echo "⚠️  未设置 RAGHUB_TEST_BASE_URL 环境变量"
    echo "使用默认值: http://localhost:8000"
    export RAGHUB_TEST_BASE_URL="http://localhost:8000"
else
    echo "✅ 服务器地址: $RAGHUB_TEST_BASE_URL"
fi

if [ -z "$RAGHUB_TEST_TIMEOUT" ]; then
    export RAGHUB_TEST_TIMEOUT="60.0"
fi

echo "✅ 超时设置: ${RAGHUB_TEST_TIMEOUT}秒"
echo "--------------------------------"

# 检查依赖
echo "📦 检查依赖..."
python -c "import pytest, httpx, raghub_client" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ 缺少必要依赖，请安装："
    echo "   pip install pytest pytest-asyncio httpx"
    exit 1
fi
echo "✅ 依赖检查通过"

# 显示选项
echo "--------------------------------"
echo "请选择测试模式："
echo "1) 运行演示脚本 (推荐)"
echo "2) 运行完整测试套件"
echo "3) 运行交互式测试"
echo "4) 运行特定测试"
echo "5) 退出"
echo "--------------------------------"

read -p "请输入选择 (1-5): " choice

case $choice in
    1)
        echo "🚀 运行演示脚本..."
        python demo.py
        ;;
    2)
        echo "🧪 运行完整测试套件..."
        pytest -v --tb=short
        ;;
    3)
        echo "🎯 启动交互式测试..."
        python run_tests.py --interactive
        ;;
    4)
        echo "可用的测试文件："
        echo "  - test_retrieval.py (检索测试)"
        echo "  - test_chat.py (聊天测试)"
        echo "  - test_documents.py (文档测试)"
        echo "  - test_index.py (索引测试)"
        echo "  - test_integration.py (集成测试)"
        read -p "请输入测试文件名 (不含.py): " testfile
        pytest "test_${testfile}.py" -v
        ;;
    5)
        echo "👋 再见!"
        exit 0
        ;;
    *)
        echo "❌ 无效选择"
        exit 1
        ;;
esac

echo "================================"
echo "测试完成"
echo "================================"

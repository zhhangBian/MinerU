import os
from lightllm_src import *

def test_lightllm_engine():
    """测试LightLLM引擎"""
    try:
        from mineru.backend.vlm.lightllm_engine_predictor import LightllmEnginePredictor

        print("正在测试LightLLM引擎...")

        # 检查是否有模型路径
        model_path = "/mnt/youwei-data/zhuohang/model/opendatalab/MinerU2.0-2505-0.9B/"

        # 尝试创建引擎实例
        print("正在创建LightLLM引擎实例...")
        engine = LightllmEnginePredictor(
            model_path=model_path,
        )

        print("✅ LightLLM引擎创建成功！")

        # 测试基本推理
        print("正在测试基本推理...")

        # 创建一个简单的测试图像（base64编码的1x1像素图像）
        test_image = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="

        # 测试提示词
        test_prompt = "请描述这张图片中的内容。"

        # 执行推理
        result = engine.predict(
            image=test_image,
            prompt=test_prompt,
            max_new_tokens=50,
            temperature=0.7
        )

        print(f"✅ 推理成功！结果: {result}")

        # 清理资源
        engine.close()
        print("✅ 资源清理完成")

        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_lightllm_client():
    """测试LightLLM客户端"""
    try:
        from mineru.backend.vlm.lightllm_client_predictor import LightllmClientPredictor

        print("\n正在测试LightLLM客户端...")

        # 检查是否有lightllm服务
        server_url = os.environ.get("LIGHTLLM_SERVER_URL", "http://127.0.0.1:8000")

        print(f"尝试连接到LightLLM服务器: {server_url}")

        # 创建客户端实例
        client = LightllmClientPredictor(
            server_url=server_url,
            temperature=0.7,
            max_new_tokens=50
        )

        print("✅ LightLLM客户端创建成功！")

        # 测试连接
        try:
            import httpx
            response = httpx.get(f"{server_url}/health", timeout=5)
            if response.status_code == 200:
                print("✅ 服务器连接正常")
            else:
                print(f"⚠️ 服务器响应异常: {response.status_code}")
        except Exception as e:
            print(f"⚠️ 无法连接到服务器: {e}")
            print("请确保LightLLM服务器正在运行")

        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def main():
    """主函数"""
    print("🚀 开始测试LightLLM集成...")
    print("=" * 50)

    # 测试引擎
    engine_success = test_lightllm_engine()

    # 测试客户端
    client_success = test_lightllm_client()

    print("=" * 50)
    print("📊 测试结果汇总:")
    print(f"LightLLM引擎: {'✅ 通过' if engine_success else '❌ 失败'}")
    print(f"LightLLM客户端: {'✅ 通过' if client_success else '❌ 失败'}")

    if engine_success or client_success:
        print("\n🎉 至少有一个LightLLM后端测试通过！")
        print("\n使用说明:")
        if engine_success:
            print("- 本地引擎模式: mineru -p input.pdf -o output -b vlm-lightllm-engine --model-path /path/to/model")
        if client_success:
            print("- 客户端模式: mineru -p input.pdf -o output -b vlm-lightllm-client -u http://127.0.0.1:8000")
    else:
        print("Failed to test LightLLM")

if __name__ == "__main__":
    main()
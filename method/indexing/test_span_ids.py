#!/usr/bin/env python
"""
测试span_ids提取功能

验证：
1. extract_span_ids_from_graph函数是否正常工作
2. 构建的索引metadata是否包含span_ids
"""

import sys
import json
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from method.indexing.build_index import extract_span_ids_from_graph, Block


def test_extract_span_ids():
    """测试span_ids提取功能"""
    print("=" * 60)
    print("测试 span_ids 提取功能")
    print("=" * 60)
    
    # 检查Graph索引文件是否存在
    graph_index_file = "index_data/Loc-Bench_V1/graph_index_v2.3/UXARRAY_uxarray.pkl"
    if not Path(graph_index_file).exists():
        print(f"⚠️  Graph索引文件不存在: {graph_index_file}")
        print("   跳过测试")
        return False
    
    # 创建测试代码块
    test_blocks = [
        Block(
            file_path="uxarray/grid/connectivity.py",
            start=0,  # 0-based，对应第1行
            end=20,   # 0-based，对应第21行
            content="test content",
            block_type="fixed"
        ),
        Block(
            file_path="uxarray/grid/coordinates.py",
            start=100,
            end=120,
            content="test content",
            block_type="fixed"
        ),
    ]
    
    repo_path = "playground/locbench_repos/UXARRAY_uxarray"
    
    try:
        print(f"\n1. 测试提取span_ids...")
        print(f"   Graph索引: {graph_index_file}")
        print(f"   代码块数量: {len(test_blocks)}")
        
        span_ids_map = extract_span_ids_from_graph(
            test_blocks,
            graph_index_file,
            repo_path
        )
        
        print(f"   ✓ 提取成功")
        print(f"   找到span_ids的代码块: {len(span_ids_map)}/{len(test_blocks)}")
        
        if span_ids_map:
            for block_idx, span_ids in span_ids_map.items():
                block = test_blocks[block_idx]
                print(f"\n   代码块 {block_idx} ({block.file_path}, 行 {block.start+1}-{block.end+1}):")
                print(f"     span_ids: {span_ids[:3]}..." if len(span_ids) > 3 else f"     span_ids: {span_ids}")
        
        return True
        
    except Exception as e:
        print(f"   ✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_metadata_format():
    """测试metadata格式"""
    print("\n" + "=" * 60)
    print("测试 metadata 格式")
    print("=" * 60)
    
    # 检查是否有已构建的索引（带span_ids）
    test_index_dir = Path("index_data/dense_index_fixed_with_spans")
    if not test_index_dir.exists():
        print(f"⚠️  测试索引目录不存在: {test_index_dir}")
        print("   需要先构建一个带span_ids的索引")
        return None
    
    # 查找第一个仓库的metadata
    repo_dirs = [d for d in test_index_dir.iterdir() if d.is_dir()]
    if not repo_dirs:
        print("   索引目录为空")
        return None
    
    repo_dir = repo_dirs[0]
    metadata_file = repo_dir / "metadata.jsonl"
    
    if not metadata_file.exists():
        print(f"   metadata文件不存在: {metadata_file}")
        return None
    
    try:
        print(f"\n检查metadata文件: {metadata_file}")
        with open(metadata_file, 'r') as f:
            lines = f.readlines()
            if not lines:
                print("   metadata文件为空")
                return None
            
            # 检查前几行
            has_span_ids = 0
            total = min(10, len(lines))
            
            for i, line in enumerate(lines[:total]):
                data = json.loads(line)
                if "span_ids" in data and data["span_ids"]:
                    has_span_ids += 1
                    if has_span_ids == 1:
                        print(f"\n   示例（第{i+1}行）:")
                        print(f"     file_path: {data.get('file_path')}")
                        print(f"     start_line: {data.get('start_line')}, end_line: {data.get('end_line')}")
                        print(f"     span_ids: {data.get('span_ids')[:3]}..." if len(data.get('span_ids', [])) > 3 else f"     span_ids: {data.get('span_ids')}")
            
            print(f"\n   前{total}行中包含span_ids的代码块: {has_span_ids}/{total}")
            
            if has_span_ids > 0:
                print("   ✓ metadata格式正确，包含span_ids")
                return True
            else:
                print("   ⚠️  metadata中没有找到span_ids")
                return False
                
    except Exception as e:
        print(f"   ✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("开始测试span_ids提取功能...\n")
    
    results = []
    results.append(("span_ids提取功能", test_extract_span_ids()))
    results.append(("metadata格式", test_metadata_format()))
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    for name, result in results:
        if result is None:
            status = "⚠️  跳过"
        elif result:
            status = "✓ 通过"
        else:
            status = "✗ 失败"
        print(f"{name}: {status}")
    
    all_passed = all(r for _, r in results if r is not None)
    
    if all_passed:
        print("\n🎉 所有测试通过！")
    else:
        print("\n⚠️  部分测试失败或跳过")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

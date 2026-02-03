import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--index_type", required=True, choices=["dense", "sparse", "hybrid"])
    args, rest = parser.parse_known_args()

    if args.index_type == "dense":
        from method.retrieval import run_with_index
        sys.argv = ["run_with_index.py"] + rest
        run_with_index.main()
        return

    if args.index_type == "sparse":
        from method.retrieval import sparse_retriever
        sys.argv = ["sparse_retriever.py"] + rest
        sparse_retriever.main()
        return

    if args.index_type == "hybrid":
        from method.retrieval import hybrid_retriever
        sys.argv = ["hybrid_retriever.py"] + rest
        hybrid_retriever.main()
        return

    raise NotImplementedError(f"index_type={args.index_type} is not implemented yet")


if __name__ == "__main__":
    main()

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--index_type", required=True, choices=["dense", "sparse", "summary"])
    args, rest = parser.parse_known_args()

    if args.index_type == "dense":
        from method.indexing import batch_build_index
        sys.argv = ["batch_build_index.py"] + rest
        batch_build_index.main()
        return

    if args.index_type == "sparse":
        from method.indexing import batch_build_sparse_index
        sys.argv = ["batch_build_sparse_index.py"] + rest
        batch_build_sparse_index.main()
        return

    if args.index_type == "summary":
        raise NotImplementedError("summary index_type is not implemented yet")

    raise NotImplementedError(f"index_type={args.index_type} is not implemented yet")


if __name__ == "__main__":
    main()

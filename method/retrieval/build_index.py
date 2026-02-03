"""
Wrapper for building fixed-line dense indexes (non-overlapping blocks).
Defaults to nov3630/RLRetriever; supply --model_name to swap models.
"""
import sys
import method.RLCoder.build_dense_index as dense_index


if __name__ == "__main__":
    sys.exit(dense_index.build_dense_index(dense_index.parse_args()))

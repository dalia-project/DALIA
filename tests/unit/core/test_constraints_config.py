import pytest

from dalia.configs import constraints_config


@pytest.mark.parametrize("inline", [{"C": [[1.0, 2.0]]}, {"e": [1.0]}])
def test_linear_file_and_inline_raises(tmp_path, inline):
    with pytest.raises(ValueError, match="not both"):
        constraints_config.parse_config(
            {"type": "linear", "input_file": str(tmp_path / "constraints.npz"), **inline}
        )

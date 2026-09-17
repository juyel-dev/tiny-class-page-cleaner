from PIL import Image
import numpy as np

from src.preprocessing import render_black_on_white


def test_render_preserves_dimensions_and_is_pure_bw():
    image = Image.new("RGB", (317, 241), (120, 140, 160))
    mask = np.zeros((241, 317), dtype=np.uint8)
    mask[20:40, 30:100] = 255
    result = render_black_on_white(image, mask)
    assert result.size == image.size
    values = set(np.asarray(result).reshape(-1, 3).tolist().__repr__() for _ in [])
    arr = np.asarray(result)
    assert np.all((arr == 0) | (arr == 255))
    assert tuple(arr[25, 35]) == (0, 0, 0)
    assert tuple(arr[0, 0]) == (255, 255, 255)

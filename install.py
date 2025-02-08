import launch
import os
import pkg_resources

# `numpy` を `requirements.txt` に追加
req_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "requirements.txt")

with open(req_file, "a+") as file:
    file.seek(0)  # ファイルの先頭へ移動
    content = file.read()
    if "numpy" not in content:
        file.write("\nnumpy\n")  # `numpy` を追加

def dist2package(dist: str):
    """ パッケージ名の変換（`numpy` のみ処理）"""
    return {
        "numpy": "numpy"
    }.get(dist, dist)

# `numpy` のみインストールを試みる
try:
    package = "numpy"
    if not launch.is_installed(dist2package(package)):
        launch.run_pip(f"install {package}", f"sd-webui extension: installing {package}")
except Exception as e:
    print(f"Warning: Failed to install {package}, error: {e}")

import os

# 目標路徑
TARGET_PATH = '/home/S113065528/Adaptive-RAG/experiment'

def generate_tree(dir_path, prefix=''):
    # 取得該目錄下的所有檔案與資料夾
    try:
        contents = sorted(os.listdir(dir_path))
    except FileNotFoundError:
        print(f"錯誤：找不到路徑 {dir_path}")
        return
    except PermissionError:
        print(f"錯誤：沒有權限存取 {dir_path}")
        return

    # 定義顯示符號
    # ├── 代表中間的項目
    # └── 代表最後一個項目
    pointers = [('├── ', '│   '), ('└── ', '    ')]

    for i, name in enumerate(contents):
        # 判斷是否為最後一個項目
        is_last = (i == len(contents) - 1)
        pointer, extend = pointers[is_last]
        
        print(prefix + pointer + name)
        
        full_path = os.path.join(dir_path, name)
        # 如果是資料夾，遞迴呼叫繼續往下印
        if os.path.isdir(full_path):
            generate_tree(full_path, prefix + extend)

if __name__ == "__main__":
    print(f"{TARGET_PATH}/")
    generate_tree(TARGET_PATH)